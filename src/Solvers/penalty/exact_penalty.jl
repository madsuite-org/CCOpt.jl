function solve_homotopy!(nlp::AbstractMPCCPenaltyModel, solver::PenaltySolver; kwargs...)
    return solve_homotopy!(nlp, solver, CCOptExecutionStats(solver); kwargs...)
end

function solve_homotopy!(solver::PenaltySolver; kwargs...)
    return solve_homotopy!(solver.pnlp, solver; kwargs...)
end

function solve_homotopy!(
    nlp::AbstractMPCCPenaltyModel,
    solver::PenaltySolver,
    stats::CCOptExecutionStats;
    x = nothing,
    y = nothing,
    zl = nothing,
    zu = nothing,
    kwargs...,
)
    ipm = solver.ipm
    solver.cnt.start_time = time()
    if x != nothing
        MadNLP.full(ipm.x)[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        ipm.y[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        MadNLP.full(ipm.zl)[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        MadNLP.full(ipm.zu)[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        MadNLP.@warn(ipm.logger, "The options set during resolve may not have an effect")
        MadNLP.set_options!(ipm.opt, kwargs)
    end

    try
        if ipm.status == MadNLP.INITIAL
            MadNLP.@notice(
                solver.logger,
                "This is $(MadNLP.introduce()), using CCOpt Ell1 extension, running with $(MadNLP.introduce(ipm.kkt.linear_solver))\n"
            )
            MadNLP.print_init(ipm)
            # Also reset rho
            set_penalty(solver.pnlp, solver.opts.rho_0)
            ipm.status = MadNLP.initialize!(ipm)
        else # resolving the problem
            # Also reset rho
            set_penalty(solver.pnlp, solver.opts.rho_0)
            ipm.status = MadNLP.reinitialize!(ipm)
        end

        while ipm.status >= MadNLP.REGULAR
            ipm.status == MadNLP.REGULAR && (ipm.status = CCOpt.homotopy!(solver))
            ipm.status == MadNLP.RESTORE && (ipm.status = MadNLP.restore!(ipm))
            ipm.status == MadNLP.ROBUST && (ipm.status = MadNLP.robust!(ipm))
        end
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                ipm.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                ipm.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                ipm.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                ipm.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                ipm.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                ipm.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            ipm.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            ipm.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            ipm.opt.rethrow_error && rethrow(e)
        elseif e isa MadNLP.InterruptException
            ipm.status=MadNLP.USER_REQUESTED_STOP
            ipm.opt.rethrow_error && rethrow(e)
        else
            ipm.status=MadNLP.INTERNAL_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        end
    finally
        update!(stats, solver)
        ipm.cnt.total_time = time() - ipm.cnt.start_time
        if !(ipm.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(ipm)
        end

        MadNLP.@notice(
            solver.logger,
            "EXIT: $(MadNLP.get_status_output(ipm.status, ipm.opt))"
        )
        ipm.opt.disable_garbage_collector && (
            GC.enable(true);
            MadNLP.@warn(ipm.logger, "Julia garbage collector is turned back on")
        )
        MadNLP.finalize(ipm.logger)
    end

    return stats
end

function homotopy!(solver::PenaltySolver{T,VT}) where {T,VT}
    ipm = solver.ipm
    nlp = solver.pnlp
    mpcc = solver.mpcc
    while true
        # Set sigma to zero for constraint infeasibility calculations
        if (ipm.cnt.k!=0 && !ipm.opt.jacobian_constant)
            MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end
        MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLP.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        ipm.inf_pr = MadNLP.get_inf_pr(ipm.c)
        ipm.inf_du = MadNLP.get_inf_du(
            MadNLP.full(ipm.f),
            MadNLP.full(ipm.zl),
            MadNLP.full(ipm.zu),
            ipm.jacl,
            sd,
        )
        ipm.inf_compl = MadNLP.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            zero(T),
            sc,
        )
        inf_compl_mu = MadNLP.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            ipm.mu,
            sc,
        )
        inf_pr_comp_sum = get_inf_pr_cc_sum(solver) # Primal complementarity residual
        solver.inf_pr_cc = get_inf_pr_cc(solver)
        push!(solver.pr_comp_hist, inf_pr_comp_sum)

        MadNLP.print_iter(solver)

        # evaluate termination criteria
        MadNLP.@trace(ipm.logger, "Evaluating termination criteria.")
        if !(MadNLP.get_intermediate_callback(ipm)(ipm, MadNLP.UserCallbackRegular())::Bool)
            return MadNLP.USER_REQUESTED_STOP
        end
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc) <= ipm.opt.tol &&
            return MadNLP.SOLVE_SUCCEEDED
        max(
            ipm.inf_pr,
            ipm.inf_du,
            ipm.inf_compl,
            min(solver.inf_pr_cc, get_inf_pr_cc_prod(solver)),
        ) <= ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc) >=
        ipm.opt.diverging_iterates_tol && return MadNLP.DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter && return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED

        # Do dynamic penalty update:
        # First calculate primal comp epsilon
        eps_pr_comp = ipm.mu^solver.opts.gamma
        if solver.opts.dynamic_rho_update &&
           isfull(solver.pr_comp_hist) &&
           solver.inf_pr_cc > eps_pr_comp &&
           inf_pr_comp_sum > solver.opts.eta_dynamic_update*maximum(solver.pr_comp_hist) &&
           get_penalty(solver.pnlp) < solver.opts.rho_max
            set_penalty(solver.pnlp, solver.opts.rho_growth_rate*get_penalty(solver.pnlp))
            MadNLP.@trace(
                solver.logger,
                "Updating the penalty parameter dynamically to $(get_penalty(solver.pnlp))."
            )
            ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
            # Also clear the filter
            empty!(ipm.filter)
            push!(ipm.filter, (ipm.theta_max, -Inf))
        end

        MadNLP.@trace(solver.logger, "Evaluating the Lagrangian Hessian.")
        if (ipm.cnt.k!=0)
            MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        # update the barrier parameter
        MadNLP.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_updated = false
        mu_old = ipm.mu
        MadNLP.update_barrier!(ipm.opt.barrier, solver.ipm, sc)
        mu_updated = ipm.mu != mu_old
        # Standard check
        # FIXME(@anton) This update happening _after_ the matrix has been factorized means
        #               we are still using the old $\rho$ for one iteration. This in principle
        #               does not impact convergence guarantees, but is inaccurate.
        #               In principle this also means we need to possibly factorize twice in the case
        #               of the QualityFunctionUpdate (though this may not even make sense and we may
        #               only use the adaptive $\rho$ update in this case).
        if mu_updated
            # check for complementarity convergence when we decrease 𝜇
            # or if we already are at smallest mu increase penalty if we are not satisfying eps_pr_comp
            if solver.inf_pr_cc > eps_pr_comp
                set_penalty(
                    solver.pnlp,
                    solver.opts.rho_growth_rate*get_penalty(solver.pnlp),
                )
                MadNLP.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(get_penalty(solver.pnlp))."
                )
                ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
                MadNLP.@trace(
                    solver.logger,
                    "Evaluating the Lagrangian Hessian (again because penalty was updated)."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
                end
            end
        elseif ipm.mu ≤ max(ipm.opt.barrier.mu_min, ipm.opt.tol/10) &&
               max(ipm.inf_pr, ipm.inf_du, inf_compl_mu) <=
               ipm.opt.barrier_tol_factor*ipm.mu
            if solver.inf_pr_cc > ipm.opt.tol
                set_penalty(
                    solver.pnlp,
                    solver.opts.rho_growth_rate*get_penalty(solver.pnlp),
                )
                MadNLP.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(get_penalty(solver.pnlp))."
                )
                ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
                empty!(ipm.filter)
                push!(ipm.filter, (ipm.theta_max, -Inf))
                MadNLP.@trace(
                    solver.logger,
                    "Evaluating the Lagrangian Hessian (again because penalty was updated)."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
                end
            end
        end

        # compute the newton step
        MadNLP.@trace(ipm.logger, "Computing the newton step.")
        MadNLP.set_aug_diagonal!(ipm.kkt, solver.ipm)
        MadNLP.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        MadNLP.dual_inf_perturbation!(
            MadNLP.primal(ipm.p),
            ipm.ind_llb,
            ipm.ind_uub,
            ipm.mu,
            ipm.opt.kappa_d,
        )
        MadNLP.inertia_correction!(ipm.inertia_corrector, solver) || return MadNLP.ROBUST

        MadNLP.@trace(ipm.logger, "Backtracking line search initiated.")
        status = MadNLP.filter_line_search!(ipm)
        if status != MadNLP.LINESEARCH_SUCCEEDED
            return status
        end

        MadNLP.@trace(ipm.logger, "Updating primal-dual variables.")
        copyto!(MadNLP.full(ipm.x), MadNLP.full(ipm.x_trial))
        copyto!(ipm.c, ipm.c_trial)
        ipm.obj_val = ipm.obj_val_trial
        MadNLP.adjust_boundary!(ipm.x_lr, ipm.xl_r, ipm.x_ur, ipm.xu_r, ipm.mu)

        MadNLP.axpy!(ipm.alpha, MadNLP.dual(ipm.d), ipm.y)

        ipm.zl_r .+= ipm.alpha_z .* MadNLP.dual_lb(ipm.d)
        ipm.zu_r .+= ipm.alpha_z .* MadNLP.dual_ub(ipm.d)
        MadNLP.reset_bound_dual!(
            MadNLP.primal(ipm.zl),
            MadNLP.primal(ipm.x),
            MadNLP.primal(ipm.xl),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )
        MadNLP.reset_bound_dual!(
            MadNLP.primal(ipm.zu),
            MadNLP.primal(ipm.xu),
            MadNLP.primal(ipm.x),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )

        MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)

        ipm.cnt.k+=1
        MadNLP.@trace(ipm.logger, "Proceeding to the next interior point iteration.")
    end
end

# evaluate mpcc objective instead of ell1 objective (though they should be the same)
function update!(stats::CCOptExecutionStats, solver::PenaltySolver{T,VT}) where {T,VT}
    ipm = solver.ipm
    n, m = NLPModels.get_nvar(ipm.nlp), get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)

    stats.status = solver.ipm.status
    MadNLP.unpack_x!(stats.solution, ipm.cb, MadNLP.variable(ipm.x))
    MadNLP.unpack_y!(stats.multipliers, ipm.cb, ipm.y)
    MadNLP.unpack_z!(stats.multipliers_L, ipm.cb, MadNLP.variable(ipm.zl))
    MadNLP.unpack_z!(stats.multipliers_U, ipm.cb, MadNLP.variable(ipm.zu))

    MadNLP.update_z!(
        ipm.cb,
        stats.solution,
        stats.multipliers,
        stats.multipliers_L,
        stats.multipliers_U,
        ipm.jacl,
    )

    ind_cc1 = get_ind_cc1(solver.mpcc)
    ind_cc2 = get_ind_cc2(solver.mpcc)
    for ii in 1:ncc
        icc1 = ind_cc1[ii]
        icc2 = ind_cc2[ii]

        stats.multipliers_x1[ii] =
            stats.multipliers_L[icc1] -
            get_penalty(solver.pnlp) * (stats.solution[icc2] - get_lvar(solver.mpcc)[icc2])
        stats.multipliers_x2[ii] =
            stats.multipliers_L[icc2] -
            get_penalty(solver.pnlp) * (stats.solution[icc1] - get_lvar(solver.mpcc)[icc1])
    end
    # Report the MPCC objective, not the penalized one.
    stats.objective = NLPModels.obj(solver.mpcc, stats.solution)
    MadNLP.unpack_cons!(stats.constraints, ipm.cb, ipm.c)
    stats.constraints .+= ipm.rhs
    @views stats.constraints[ipm.ind_ineq] .+= MadNLP.slack(ipm.x)
    # Cut out scholtes constraints now we don't need them to calculate multipliers
    stats.dual_feas = ipm.inf_du
    stats.primal_feas = ipm.inf_pr
    stats.iter = ipm.cnt.k
    stats.inf_pr_cc = solver.inf_pr_cc
    solver.cnt.solve_time = time() - solver.cnt.start_time
    solver.cnt.total_time = solver.cnt.solve_time + solver.cnt.init_time
    solver.cnt.solver_time =
        solver.cnt.total_time - solver.cnt.init_time - ipm.cnt.linear_solver_time -
        ipm.cnt.eval_function_time
    return stats
end

function regularize_Q!(solver::PenaltySolver{T}) where {T}
    if solver.opts.q_regularization == :none ||
       solver.ipm.mu < solver.opts.min_reg_mu ||
       solver.ipm.mu > solver.opts.max_reg_mu
        return false
    end

    ipm = solver.ipm
    cb = ipm.cb
    pnlp = solver.pnlp
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = get_ncc(solver.mpcc)
    nnzh = get_nnzh(solver.mpcc)
    rho = get_penalty(solver.pnlp)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    A = Array{T}(undef, 2, 2)
    regularized = false
    for i in 1:ncc
        cc1 = ind_cc1[i]
        cc2 = ind_cc2[i]

        if solver.opts.q_regularization == :eigenvalue_decomposition
            A[1, 1] = kkt.pr_diag[cc1]
            A[2, 2] = kkt.pr_diag[cc2]
            A[2, 1] = rho*cb.obj_scale[]
            A[1, 2] = rho*cb.obj_scale[]
            E = eigen(Symmetric(A))
            if E.values[1] < 0 || E.values[2] > solver.opts.max_eig_value
                E.values[1] = solver.opts.min_eig_value
                E.values[2] = min(solver.opts.max_eig_value, E.values[2])
                A .= Symmetric(Matrix(E))
                kkt.reg[cc1] = A[1, 1] - kkt.pr_diag[cc1]
                kkt.reg[cc2] = A[2, 2] - kkt.pr_diag[cc2]
                kkt.pr_diag[cc1] = A[1, 1]
                kkt.pr_diag[cc2] = A[2, 2]
                kkt.hess_raw.V[nnzh+i] = A[1, 2]
                regularized = true
            end
        elseif solver.opts.q_regularization == :critical_rho
            rho_max = sqrt(kkt.pr_diag[cc1]*kkt.pr_diag[cc2])
            if rho*cb.obj_scale[] > rho_max
                kkt.hess_raw.V[end-ncc+i] =
                    solver.opts.critical_rho_factor*rho_max*(
                        pnlp.meta.minimize ? one(T) : -one(T)
                    )
                regularized = true
            end
        end
    end
    # We modify hess_raw so need to compress_hessian again.
    MadNLP.compress_hessian!(kkt)
    return regularized
end

function unregularize_Q!(solver::PenaltySolver{T}) where {T}
    ipm = solver.ipm
    cb = ipm.cb
    pnlp = solver.pnlp
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = get_ncc(solver.mpcc)
    nnzh = get_nnzh(solver.mpcc)
    rho = cb.obj_scale[]*get_penalty(solver.pnlp)
    ind_cc1 = get_ind_cc1(solver.mpcc)
    ind_cc2 = get_ind_cc2(solver.mpcc)
    A = Array{T}(undef, 2, 2)
    regularized = false
    kkt.pr_diag[ind_cc1] .-= kkt.reg[ind_cc1]
    kkt.pr_diag[ind_cc2] .-= kkt.reg[ind_cc2]
    kkt.reg[ind_cc1] .= 0
    kkt.reg[ind_cc2] .= 0
    kkt.hess_raw.V[(nnzh+1):(nnzh+ncc)] .= pnlp.meta.minimize ? rho : -rho
    # We modify hess_raw so need to compress_hessian again.
    MadNLP.compress_hessian!(kkt)
    return regularized
end

function MadNLP.inertia_correction!(
    inertia_corrector::MadNLP.InertiaBased,
    solver::PenaltySolver{T},
) where {T}
    ipm = solver.ipm
    n_trial = 0
    ipm.del_w = del_w_prev = zero(T)
    ipm.del_c = del_c_prev = zero(T)

    MadNLP.@trace(ipm.logger, "Inertia-based regularization started.")

    MadNLP.factorize_wrapper!(ipm)
    num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

    solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
    else
        false
    end

    # Try to regularize
    if !solve_status && regularize_Q!(solver)
        MadNLP.@trace(ipm.logger, "Trying to regularize out penalty.")
        MadNLP.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
            MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        if !solve_status
            # undo what we did and just do inertia correction
            unregularize_Q!(solver)
        end
    end
    while !solve_status
        MadNLP.@debug(ipm.logger, "Primal-dual perturbed.")

        if n_trial == 0
            ipm.del_w =
                ipm.del_w_last==zero(T) ? ipm.opt.first_hessian_perturbation :
                max(
                    ipm.opt.min_hessian_perturbation,
                    ipm.opt.perturb_dec_fact*ipm.del_w_last,
                )
        else
            ipm.del_w *=
                ipm.del_w_last==zero(T) ? ipm.opt.perturb_inc_fact_first :
                ipm.opt.perturb_inc_fact
            if ipm.del_w>ipm.opt.max_hessian_perturbation
                ipm.cnt.k+=1
                MadNLP.@debug(
                    ipm.logger,
                    "Primal regularization is too big. Switching to restoration phase."
                )
                return false
            end
        end
        ipm.del_c =
            num_zero == 0 ? zero(T) :
            ipm.opt.jacobian_regularization_value *
            ipm.mu^(ipm.opt.jacobian_regularization_exponent)
        MadNLP.regularize_diagonal!(ipm.kkt, ipm.del_w - del_w_prev, ipm.del_c - del_c_prev)
        del_w_prev = ipm.del_w
        del_c_prev = ipm.del_c

        MadNLP.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        n_trial += 1
    end

    ipm.del_w != 0 && (ipm.del_w_last = ipm.del_w)
    return true
end

function MadNLP.inertia_correction!(
    inertia_corrector::MadNLP.AbstractInertiaCorrector,
    solver::PenaltySolver{T},
) where {T}
    return MadNLP.inertia_correction!(inertia_corrector, solver.ipm)
end

function MadNLP.print_summary(solver::PenaltySolver)
    # TODO inquire this from nlpmodel wrapper
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    cnt_ipm = ipm.cnt
    cnt = solver.cnt
    MadNLP.@notice(ipm.logger, "")
    MadNLP.@notice(ipm.logger, "Number of Iterations....: $(cnt_ipm.k)\n")
    MadNLP.@notice(
        ipm.logger,
        "                                   (scaled)                 (unscaled)"
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Objective...............:  % 1.16e   % 1.16e",
            ipm.obj_val,
            ipm.obj_val/obj_scale
        )
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Dual infeasibility......:   %1.16e    %1.16e",
            ipm.inf_du,
            ipm.inf_du/obj_scale
        )
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Constraint violation....:   %1.16e    %1.16e",
            norm(ipm.c, Inf),
            ipm.inf_pr
        )
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Complementarity.........:   %1.16e    %1.16e",
            ipm.inf_compl*obj_scale,
            ipm.inf_compl
        )
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Primal Complementarity..:   %1.16e    %1.16e",
            solver.inf_pr_cc,
            solver.inf_pr_cc
        )
    )
    MadNLP.@notice(
        ipm.logger,
        @sprintf(
            "Overall NLP error.......:   %1.16e    %1.16e\n",
            max(ipm.inf_du*obj_scale, norm(ipm.c, Inf), ipm.inf_compl, solver.inf_pr_cc),
            max(ipm.inf_du, ipm.inf_pr, ipm.inf_compl, solver.inf_pr_cc)
        )
    )

    MadNLP.@notice(
        ipm.logger,
        "Number of objective function evaluations              = $(cnt_ipm.obj_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of objective gradient evaluations              = $(cnt_ipm.obj_grad_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of constraint evaluations                      = $(cnt_ipm.con_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of constraint Jacobian evaluations             = $(cnt_ipm.con_jac_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of Lagrangian Hessian evaluations              = $(cnt_ipm.lag_hess_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of KKT factorizations                          = $(cnt_ipm.factorization_cnt)"
    )
    MadNLP.@notice(
        ipm.logger,
        "Number of KKT backsolves                              = $(cnt_ipm.backsolve_cnt)\n"
    )
    MadNLP.@notice(
        ipm.logger,
        "Total wall secs in initialization                     = $(MadNLP.format_time(cnt.init_time))"
    )
    MadNLP.@notice(
        ipm.logger,
        "Total wall secs in linear solver                      = $(MadNLP.format_time(cnt_ipm.linear_solver_time))"
    )
    MadNLP.@notice(
        ipm.logger,
        "Total wall secs in NLP function evaluations           = $(MadNLP.format_time(cnt_ipm.eval_function_time))"
    )
    MadNLP.@notice(
        ipm.logger,
        "Total wall secs in solver (w/o init./fun./lin. alg.)  = $(MadNLP.format_time(cnt.total_time - cnt.init_time - cnt_ipm.linear_solver_time - cnt_ipm.eval_function_time))"
    )
    MadNLP.@notice(
        ipm.logger,
        "Total wall secs                                       = $(MadNLP.format_time(cnt.total_time))\n"
    )
end
