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
        MadNLPCore.full(ipm.x)[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        ipm.y[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        MadNLPCore.full(ipm.zl)[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        MadNLPCore.full(ipm.zu)[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        MadNLPCore.@warn(ipm.logger, "The options set during resolve may not have an effect")
        MadNLPCore.set_options!(ipm.opt, kwargs)
    end

    try
        if ipm.status == MadNLPCore.INITIAL
            MadNLPCore.@notice(
                solver.logger,
                "This is $(MadNLPCore.introduce()), using CCOpt Ell1 extension, running with $(MadNLPCore.introduce(ipm.kkt.linear_solver))\n"
            )
            MadNLPCore.print_init(ipm)
            # Also reset rho
            set_penalty(solver.pnlp, solver.opts.rho_0)
            ipm.status = MadNLPCore.initialize!(ipm)
        else # resolving the problem
            # Also reset rho
            set_penalty(solver.pnlp, solver.opts.rho_0)
            ipm.status = MadNLPCore.reinitialize!(ipm)
        end

        while ipm.status >= MadNLPCore.REGULAR
            ipm.status == MadNLPCore.REGULAR && (ipm.status = CCOpt.homotopy!(solver))
            ipm.status == MadNLPCore.RESTORE && (ipm.status = MadNLPCore.restore!(ipm))
            ipm.status == MadNLPCore.ROBUST && (ipm.status = MadNLPCore.robust!(ipm))
        end
    catch e
        if e isa MadNLPCore.InvalidNumberException
            if e.callback == :obj
                ipm.status=MadNLPCore.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                ipm.status=MadNLPCore.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                ipm.status=MadNLPCore.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                ipm.status=MadNLPCore.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                ipm.status=MadNLPCore.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                ipm.status=MadNLPCore.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLPCore.NotEnoughDegreesOfFreedomException
            ipm.status=MadNLPCore.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLPCore.LinearSolverException
            ipm.status=MadNLPCore.ERROR_IN_STEP_COMPUTATION;
            ipm.opt.rethrow_error && rethrow(e)
        elseif e isa MadNLPCore.InterruptException
            ipm.status=MadNLPCore.USER_REQUESTED_STOP
            ipm.opt.rethrow_error && rethrow(e)
        else
            ipm.status=MadNLPCore.INTERNAL_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        end
    finally
        update!(stats, solver)
        ipm.cnt.total_time = time() - ipm.cnt.start_time
        if !(ipm.status < MadNLPCore.SOLVE_SUCCEEDED)
            MadNLPCore.print_summary(ipm)
        end

        MadNLPCore.@notice(
            solver.logger,
            "EXIT: $(MadNLPCore.get_status_output(ipm.status, ipm.opt))"
        )
        ipm.opt.disable_garbage_collector && (
            GC.enable(true);
            MadNLPCore.@warn(ipm.logger, "Julia garbage collector is turned back on")
        )
        MadNLPCore.finalize(ipm.logger)
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
            MadNLPCore.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end
        MadNLPCore.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLPCore.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLPCore.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        ipm.inf_pr = MadNLPCore.get_inf_pr(ipm.c)
        ipm.inf_du = MadNLPCore.get_inf_du(
            MadNLPCore.full(ipm.f),
            MadNLPCore.full(ipm.zl),
            MadNLPCore.full(ipm.zu),
            ipm.jacl,
            sd,
        )
        ipm.inf_compl = MadNLPCore.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            zero(T),
            sc,
        )
        inf_compl_mu = MadNLPCore.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            ipm.mu,
            sc,
        )
        inf_pr_comp = comp_residual(mpcc, MadNLPCore.variable(ipm.x)) # Primal complementarity residual
        inf_pr_comp_sum = get_inf_pr_cc_sum(solver) # Primal complementarity residual
        solver.inf_pr_cc = get_inf_pr_cc(solver)
        push!(solver.pr_comp_hist, inf_pr_comp_sum)

        MadNLPCore.print_iter(solver)

        # evaluate termination criteria
        MadNLPCore.@trace(ipm.logger, "Evaluating termination criteria.")
        if !(MadNLPCore.get_intermediate_callback(ipm)(ipm, MadNLPCore.UserCallbackRegular())::Bool)
            return USER_REQUESTED_STOP
        end
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc) <= ipm.opt.tol &&
            return MadNLPCore.SOLVE_SUCCEEDED
        max(
            ipm.inf_pr,
            ipm.inf_du,
            ipm.inf_compl,
            min(solver.inf_pr_cc, get_inf_pr_cc_prod(solver)),
        ) <= ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLPCore.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc) >=
        ipm.opt.diverging_iterates_tol && return MadNLPCore.DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter && return MadNLPCore.MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLPCore.MAXIMUM_WALLTIME_EXCEEDED

        # Do dynamic penalty update:
        # First calculate primal comp epsilon
        eps_pr_comp = ipm.mu^solver.opts.gamma
        if solver.opts.dynamic_rho_update &&
           isfull(solver.pr_comp_hist) &&
           solver.inf_pr_cc > eps_pr_comp &&
           inf_pr_comp_sum > solver.opts.eta_dynamic_update*maximum(solver.pr_comp_hist) &&
           get_penalty(solver.pnlp) < solver.opts.rho_max
            set_penalty(solver.pnlp, solver.opts.rho_growth_rate*get_penalty(solver.pnlp))
            MadNLPCore.@trace(
                solver.logger,
                "Updating the penalty parameter dynamically to $(get_penalty(solver.pnlp))."
            )
            ipm.obj_val = MadNLPCore.eval_f_wrapper(ipm, ipm.x)
            # Also clear the filter
            empty!(ipm.filter)
            push!(ipm.filter, (ipm.theta_max, -Inf))
        end

        MadNLPCore.@trace(solver.logger, "Evaluating the Lagrangian Hessian.")
        if (ipm.cnt.k!=0)
            MadNLPCore.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        # update the barrier parameter
        MadNLPCore.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_updated = false
        mu_old = ipm.mu
        MadNLPCore.update_barrier!(ipm.opt.barrier, solver.ipm, sc)
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
                MadNLPCore.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(get_penalty(solver.pnlp))."
                )
                ipm.obj_val = MadNLPCore.eval_f_wrapper(ipm, ipm.x)
                MadNLPCore.@trace(
                    solver.logger,
                    "Evaluating the Lagrangian Hessian (again because penalty was updated)."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLPCore.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
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
                MadNLPCore.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(get_penalty(solver.pnlp))."
                )
                ipm.obj_val = MadNLPCore.eval_f_wrapper(ipm, ipm.x)
                empty!(ipm.filter)
                push!(ipm.filter, (ipm.theta_max, -Inf))
                MadNLPCore.@trace(
                    solver.logger,
                    "Evaluating the Lagrangian Hessian (again because penalty was updated)."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLPCore.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
                end
            end
        end

        # compute the newton step
        MadNLPCore.@trace(ipm.logger, "Computing the newton step.")
        MadNLPCore.set_aug_diagonal!(ipm.kkt, solver.ipm)
        MadNLPCore.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        MadNLPCore.dual_inf_perturbation!(
            MadNLPCore.primal(ipm.p),
            ipm.ind_llb,
            ipm.ind_uub,
            ipm.mu,
            ipm.opt.kappa_d,
        )
        MadNLPCore.inertia_correction!(ipm.inertia_corrector, solver) || return MadNLPCore.ROBUST

        MadNLPCore.@trace(ipm.logger, "Backtracking line search initiated.")
        status = MadNLPCore.filter_line_search!(ipm)
        if status != MadNLPCore.LINESEARCH_SUCCEEDED
            return status
        end

        MadNLPCore.@trace(ipm.logger, "Updating primal-dual variables.")
        copyto!(MadNLPCore.full(ipm.x), MadNLPCore.full(ipm.x_trial))
        copyto!(ipm.c, ipm.c_trial)
        ipm.obj_val = ipm.obj_val_trial
        MadNLPCore.adjust_boundary!(ipm.x_lr, ipm.xl_r, ipm.x_ur, ipm.xu_r, ipm.mu)

        MadNLPCore.axpy!(ipm.alpha, MadNLPCore.dual(ipm.d), ipm.y)

        ipm.zl_r .+= ipm.alpha_z .* MadNLPCore.dual_lb(ipm.d)
        ipm.zu_r .+= ipm.alpha_z .* MadNLPCore.dual_ub(ipm.d)
        MadNLPCore.reset_bound_dual!(
            MadNLPCore.primal(ipm.zl),
            MadNLPCore.primal(ipm.x),
            MadNLPCore.primal(ipm.xl),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )
        MadNLPCore.reset_bound_dual!(
            MadNLPCore.primal(ipm.zu),
            MadNLPCore.primal(ipm.xu),
            MadNLPCore.primal(ipm.x),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )

        MadNLPCore.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)

        ipm.cnt.k+=1
        MadNLPCore.@trace(ipm.logger, "Proceeding to the next interior point iteration.")
    end
end

# evaluate mpcc objective instead of ell1 objective (though they should be the same)
function update!(stats::CCOptExecutionStats, solver::PenaltySolver{T,VT}) where {T,VT}
    ipm = solver.ipm
    n, m = NLPModels.get_nvar(ipm.nlp), get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)

    stats.status = solver.ipm.status
    MadNLPCore.unpack_x!(stats.solution, ipm.cb, MadNLPCore.variable(ipm.x))
    MadNLPCore.unpack_y!(stats.multipliers, ipm.cb, ipm.y)
    MadNLPCore.unpack_z!(stats.multipliers_L, ipm.cb, MadNLPCore.variable(ipm.zl))
    MadNLPCore.unpack_z!(stats.multipliers_U, ipm.cb, MadNLPCore.variable(ipm.zu))

    MadNLPCore.update_z!(
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
    stats.objective = MadNLPCore.unpack_obj(ipm.cb, ipm.obj_val)
    MadNLPCore.unpack_cons!(stats.constraints, ipm.cb, ipm.c)
    stats.constraints .+= ipm.rhs
    @views stats.constraints[ipm.ind_ineq] .+= MadNLPCore.slack(ipm.x)
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
    MadNLPCore.compress_hessian!(kkt)
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
    MadNLPCore.compress_hessian!(kkt)
    return regularized
end

function MadNLPCore.inertia_correction!(
    inertia_corrector::MadNLPCore.InertiaBased,
    solver::PenaltySolver{T},
) where {T}
    ipm = solver.ipm
    n_trial = 0
    ipm.del_w = del_w_prev = zero(T)
    ipm.del_c = del_c_prev = zero(T)

    MadNLPCore.@trace(ipm.logger, "Inertia-based regularization started.")

    MadNLPCore.factorize_wrapper!(ipm)
    num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

    solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
    else
        false
    end

    # Try to regularize
    if !solve_status && regularize_Q!(solver)
        MadNLPCore.@trace(ipm.logger, "Trying to regularize out penalty.")
        MadNLPCore.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
            MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        if !solve_status
            # undo what we did and just do inertia correction
            unregularize_Q!(solver)
        end
    end
    while !solve_status
        MadNLPCore.@debug(ipm.logger, "Primal-dual perturbed.")

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
                MadNLPCore.@debug(
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
        MadNLPCore.regularize_diagonal!(ipm.kkt, ipm.del_w - del_w_prev, ipm.del_c - del_c_prev)
        del_w_prev = ipm.del_w
        del_c_prev = ipm.del_c

        MadNLPCore.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLPCore.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLPCore.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            MadNLPCore.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        n_trial += 1
    end

    ipm.del_w != 0 && (ipm.del_w_last = ipm.del_w)
    return true
end

function MadNLPCore.inertia_correction!(
    inertia_corrector::MadNLPCore.AbstractInertiaCorrector,
    solver::PenaltySolver{T},
) where {T}
    return MadNLPCore.inertia_correction!(inertia_corrector, solver.ipm)
end

function MadNLPCore.print_summary(solver::PenaltySolver)
    # TODO inquire this from nlpmodel wrapper
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    cnt_ipm = ipm.cnt
    cnt = solver.cnt
    MadNLPCore.@notice(ipm.logger, "")
    MadNLPCore.@notice(ipm.logger, "Number of Iterations....: $(cnt_ipm.k)\n")
    MadNLPCore.@notice(
        ipm.logger,
        "                                   (scaled)                 (unscaled)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Objective...............:  % 1.16e   % 1.16e",
            ipm.obj_val,
            ipm.obj_val/obj_scale
        )
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Dual infeasibility......:   %1.16e    %1.16e",
            ipm.inf_du,
            ipm.inf_du/obj_scale
        )
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Constraint violation....:   %1.16e    %1.16e",
            norm(ipm.c, Inf),
            ipm.inf_pr
        )
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Complementarity.........:   %1.16e    %1.16e",
            ipm.inf_compl*obj_scale,
            ipm.inf_compl
        )
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Primal Complementarity..:   %1.16e    %1.16e",
            solver.inf_pr_cc,
            solver.inf_pr_cc
        )
    )
    MadNLPCore.@notice(
        ipm.logger,
        @sprintf(
            "Overall NLP error.......:   %1.16e    %1.16e\n",
            max(ipm.inf_du*obj_scale, norm(ipm.c, Inf), ipm.inf_compl, solver.inf_pr_cc),
            max(ipm.inf_du, ipm.inf_pr, ipm.inf_compl, solver.inf_pr_cc)
        )
    )

    MadNLPCore.@notice(
        ipm.logger,
        "Number of objective function evaluations              = $(cnt_ipm.obj_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of objective gradient evaluations              = $(cnt_ipm.obj_grad_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of constraint evaluations                      = $(cnt_ipm.con_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of constraint Jacobian evaluations             = $(cnt_ipm.con_jac_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of Lagrangian Hessian evaluations              = $(cnt_ipm.lag_hess_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of KKT factorizations                          = $(cnt_ipm.factorization_cnt)"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Number of KKT backsolves                              = $(cnt_ipm.backsolve_cnt)\n"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Total wall secs in initialization                     = $(MadNLPCore.format_time(cnt.init_time))"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Total wall secs in linear solver                      = $(MadNLPCore.format_time(cnt_ipm.linear_solver_time))"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Total wall secs in NLP function evaluations           = $(MadNLPCore.format_time(cnt_ipm.eval_function_time))"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Total wall secs in solver (w/o init./fun./lin. alg.)  = $(MadNLPCore.format_time(cnt.total_time - cnt.init_time - cnt_ipm.linear_solver_time - cnt_ipm.eval_function_time))"
    )
    MadNLPCore.@notice(
        ipm.logger,
        "Total wall secs                                       = $(MadNLPCore.format_time(cnt.total_time))\n"
    )
end
