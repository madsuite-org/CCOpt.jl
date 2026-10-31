
function get_eta_heuristic(solver::RelaxationSolver)
    if solver.ipm.mu ≤ solver.opts.mu_thresh
        return solver.opts.eta_factor*solver.ipm.mu/(
            1+max(maximum(MadNLPCore.full(solver.ipm.zu)), maximum(MadNLPCore.full(solver.ipm.zl)))
        )
    else
        return 0.0
    end
end

function MadNLPCore.set_aug_diagonal!(
    kkt::MadNLPCore.AbstractKKTSystem{T},
    solver::RelaxationSolver{T,VT},
    eta::T,
) where {T,VT}
    ipm = solver.ipm
    n = length(ipm.x_ur)
    ncc = get_ncc(ipm.nlp.mpcc)

    fill!(kkt.reg, ipm.opt.default_primal_regularization)
    fill!(kkt.du_diag, -ipm.opt.default_dual_regularization)
    kkt.l_diag .= ipm.xl_r .- ipm.x_lr   # (Xˡ - X)
    kkt.u_diag .= ipm.x_ur .- ipm.xu_r   # (X - Xᵘ)
    copyto!(kkt.l_lower, ipm.zl_r)
    copyto!(kkt.u_lower, ipm.zu_r)

    # Regularize with 𝜂 using vicente-wright
    if solver.opts.kkt_regularization == :vicente_wright
        MadNLPCore.@debug(
            solver.logger,
            "Applying regularization to complementarity slacks eta = $(eta)"
        )
        kkt.u_diag[(n-ncc+1):n] .= @views min.(kkt.u_diag[(n-ncc+1):n], -eta)
        kkt.u_lower[(n-ncc+1):n] .= @views max.(kkt.u_lower[(n-ncc+1):n], eta)
        # Lower bounds
        kkt.l_diag[solver.ind_cc1_lb] .= @views min.(kkt.l_diag[solver.ind_cc1_lb], -eta)
        kkt.l_lower[solver.ind_cc1_lb] .= @views max.(kkt.l_lower[solver.ind_cc1_lb], eta)
        kkt.l_diag[solver.ind_cc2_lb] .= @views min.(kkt.l_diag[solver.ind_cc2_lb], -eta)
        kkt.l_lower[solver.ind_cc2_lb] .= @views max.(kkt.l_lower[solver.ind_cc2_lb], eta)
    end

    if solver.opts.kkt_regularization == :vicente_wright_sum
        MadNLPCore.@debug(
            solver.logger,
            "Applying regularization to complementarity slacks eta = $(eta)"
        )
        kkt.u_diag[(n-ncc+1):n] .-= @views eta .* kkt.u_lower[(n-ncc+1):n]

        # Lower bounds
        kkt.l_diag[solver.ind_cc1_lb] .-= @views eta .* kkt.l_lower[solver.ind_cc1_lb]
        kkt.l_diag[solver.ind_cc2_lb] .-= @views eta .* kkt.l_lower[solver.ind_cc2_lb]
    end

    MadNLPCore._set_aug_diagonal!(kkt)
    return
end

function MadNLPCore.set_aug_diagonal!(
    kkt::MadNLPCore.ScaledSparseKKTSystem{T},
    solver::RelaxationSolver{T,VT},
    eta::T,
) where {T,VT}
    ipm = solver.ipm
    n = length(ipm.x_ur)
    ncc = get_ncc(ipm.nlp.mpcc)

    fill!(kkt.reg, ipm.opt.default_primal_regularization)
    fill!(kkt.du_diag, -ipm.opt.default_dual_regularization)
    # Ensure l_diag and u_diag have only non negative entries
    kkt.l_diag .= ipm.x_lr .- ipm.xl_r   # (X - Xˡ)
    kkt.u_diag .= ipm.xu_r .- ipm.x_ur   # (Xᵘ - X)
    copyto!(kkt.l_lower, ipm.zl_r)
    copyto!(kkt.u_lower, ipm.zu_r)

    if solver.opts.kkt_regularization == :vicente_wright
        MadNLPCore.@debug(
            solver.logger,
            "Applying regularization to complementarity slacks eta = $(eta)"
        )
        kkt.u_diag[(n-ncc+1):n] .= @views max.(kkt.u_diag[(n-ncc+1):n], eta)
        kkt.u_lower[(n-ncc+1):n] .= @views max.(kkt.u_lower[(n-ncc+1):n], eta)
    end

    return MadNLPCore._set_aug_diagonal!(kkt)
end

function solve_homotopy!(
    nlp::ST,
    solver::RelaxationSolver;
    kwargs...,
) where {ST<:AbstractMPCCRelaxation}
    return solve_homotopy!(nlp, solver, CCOptExecutionStats(solver); kwargs...)
end

function solve_homotopy!(solver::RelaxationSolver; kwargs...)
    return solve_homotopy!(solver.rnlp, solver; kwargs...)
end

function solve_homotopy!(
    nlp::ST,
    solver::RelaxationSolver,
    stats::CCOptExecutionStats;
    x = nothing,
    y = nothing,
    zl = nothing,
    zu = nothing,
    kwargs...,
) where {ST<:AbstractMPCCRelaxation}
    ipm = solver.ipm
    ipm.cnt.start_time = time()
    solver.cnt.start_time = ipm.cnt.start_time
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
        # TODO(@anton) Probably should specialize `initialize!` as order of initalizing things is important.
        if ipm.status == MadNLPCore.INITIAL
            MadNLPCore.@notice(
                solver.logger,
                "This is $(MadNLPCore.introduce()), using CCOpt extension, running with $(MadNLPCore.introduce(ipm.kkt.linear_solver))\n"
            )
            MadNLPCore.print_init(ipm)
            # Also reset sigma
            ipm.status = MadNLPCore.initialize!(solver)
            init_sigma!(solver.opts.relaxation_update, solver.rnlp, solver)
            solver.inf_pr_cc = CCOpt.get_inf_pr_cc(solver)
        else # resolving the problem
            # Also reset sigma
            init_sigma!(solver.opts.relaxation_update, solver.rnlp, solver)
            ipm.status = MadNLPCore.reinitialize!(ipm)
            solver.inf_pr_cc = CCOpt.get_inf_pr_cc(solver)
        end
        # possibly fix complementarity variable upper bounds:
        if solver.opts.respect_comp_bounds
            MadNLPCore.variable(ipm.xl)[solver.ind_cc1] .= get_lvar(solver.mpcc)[solver.ind_cc1]
            MadNLPCore.variable(ipm.xl)[solver.ind_cc2] .= get_lvar(solver.mpcc)[solver.ind_cc2]
        end

        while ipm.status >= MadNLPCore.REGULAR
            ipm.status == MadNLPCore.REGULAR && (ipm.status = CCOpt.homotopy!(solver))
            ipm.status == MadNLPCore.RESTORE && (ipm.status = MadNLPCore.restore!(ipm))
            ipm.status == MadNLPCore.ROBUST && (ipm.status = MadNLPCore.robust!(ipm))
        end
        MadNLPCore.unpack_x!(solver.x, ipm.cb, MadNLPCore.variable(ipm.x))
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
        log_iter(solver.iterate_logger, solver)
        update!(stats, solver)
        ipm.cnt.total_time = time() - ipm.cnt.start_time
        if !(ipm.status < MadNLPCore.SOLVE_SUCCEEDED)
            MadNLPCore.print_summary(solver)
        end

        ipm.opt.disable_garbage_collector && (
            GC.enable(true);
            MadNLPCore.@warn(ipm.logger, "Julia garbage collector is turned back on")
        )
        MadNLPCore.finalize(ipm.logger)
        MadNLPCore.finalize(solver.logger)
        finalize(solver.iterate_logger)
    end

    return stats
end

function initialize_comps!(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    nlp = ipm.nlp
    opts = solver.opts

    if opts.center_complementarities
        # TODO(@anton): There are a lot of options here, for now lets assume proportional.
        #               The correct way is probably to do this at the first iterate or call
        #               update_mu! and update_sigma!, in initialize! (though this is maybe
        #               difficult due to the circular dependency of it all (particularly
        #               in the case of the quality function.

        # We initialize here by moving the initial values of the two complementarities and
        # the initial slack to be on the x1=x2 line and a factor (centering_factor) away
        # from the inequality constraint.
        x_vec = MadNLPCore.variable(ipm.x)
        s_vec = MadNLPCore.slack(ipm.x)
        sigma_0 = ipm.opt.barrier.mu_init
        ind_cc1 = solver.ind_cc1
        ind_cc2 = solver.ind_cc2
        ncc = get_ncc(mpcc)
        x_vec[ind_cc1] .=
            opts.centering_factor*sqrt(sigma_0) .+ get_lvar(mpcc)[get_ind_cc1(mpcc)]
        x_vec[ind_cc2] .=
            opts.centering_factor*sqrt(sigma_0) .+ get_lvar(mpcc)[get_ind_cc2(mpcc)]
        s_vec[(end-ncc+1):end] .= -sqrt(2*(1-opts.centering_factor*sigma_0))
    end
end

function MadNLPCore.initialize!(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    nlp = ipm.nlp
    opt = ipm.opt

    # Initializing variables
    MadNLPCore.@trace(ipm.logger, "Initializing variables.")
    MadNLPCore.initialize!(
        ipm.cb,
        ipm.x,
        ipm.xl,
        ipm.xu,
        ipm.y,
        ipm.rhs,
        ipm.ind_ineq;
        tol = opt.bound_relax_factor,
        bound_push = opt.bound_push,
        bound_fac = opt.bound_fac,
    )

    # Do custom initialization of the complementarity variables
    initialize_comps!(solver)

    fill!(ipm.jacl, zero(T))
    fill!(ipm.zl_r, one(T))
    fill!(ipm.zu_r, one(T))

    # Initializing scaling factors
    if opt.nlp_scaling
        MadNLPCore.set_scaling!(
            ipm.cb,
            ipm.x,
            ipm.xl,
            ipm.xu,
            ipm.y,
            ipm.rhs,
            ipm.ind_ineq,
            opt.nlp_scaling_max_gradient,
        )
    end

    # Initializing KKT system
    MadNLPCore.initialize!(ipm.kkt)

    # Initializing jacobian and gradient
    MadNLPCore.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    MadNLPCore.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)

    MadNLPCore.@trace(ipm.logger, "Initializing constraint duals.")
    if !ipm.opt.dual_initialized
        MadNLPCore.initialize_dual(ipm, opt.dual_initialization_method)
    end

    # Initializing
    ipm.obj_val = MadNLPCore.eval_f_wrapper(ipm, ipm.x)
    MadNLPCore.eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    MadNLPCore.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)

    theta = MadNLPCore.get_theta(ipm.c)
    ipm.theta_max = 1e4*max(1, theta)
    ipm.theta_min = 1e-4*max(1, theta)

    mu_init = ipm.opt.barrier.mu_init
    ipm.mu = mu_init
    ipm.tau = max(ipm.opt.tau_min, 1-mu_init)
    push!(ipm.filter, (ipm.theta_max, -Inf))

    return MadNLPCore.REGULAR
end

function homotopy!(solver::RelaxationSolver{T,VT}) where {T,VT}
    opts = solver.opts
    ipm = solver.ipm
    mpcc = solver.mpcc
    while true
        if (ipm.cnt.k!=0 && !ipm.opt.jacobian_constant)
            MadNLPCore.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end
        # Set σ to zero for constraint infeasibility calculations
        MadNLPCore.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLPCore.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLPCore.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        solver.inf_pr_cc = CCOpt.get_inf_pr_cc(solver)
        ipm.inf_pr =
            max(MadNLPCore.get_inf_pr(@view(ipm.c[1:get_ncon(mpcc)])), solver.inf_pr_cc)
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
        estimate_mpec_multipliers(solver)

        MadNLPCore.print_iter(solver)
        # evaluate termination criteria
        MadNLPCore.@trace(ipm.logger, "Evaluating termination criteria.")
        if !(MadNLPCore.get_intermediate_callback(ipm)(ipm, MadNLPCore.UserCallbackRegular())::Bool)
            return USER_REQUESTED_STOP
        end

        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) <= ipm.opt.tol &&
            return MadNLPCore.SOLVE_SUCCEEDED
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) <= ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLPCore.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) >= ipm.opt.diverging_iterates_tol &&
            return MadNLPCore.DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter && return MadNLPCore.MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLPCore.MAXIMUM_WALLTIME_EXCEEDED

        # Now go back to using relaxed inf_pr
        ipm.inf_pr = MadNLPCore.get_inf_pr(ipm.c)
        MadNLPCore.@trace(solver.logger, "Get eta.")
        eta_k = get_eta_heuristic(solver)

        # Evaluate the hessian of the lagrangian
        MadNLPCore.@trace(ipm.logger, "Evaluating nabla L.")
        if (ipm.cnt.k!=0)
            MadNLPCore.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        MadNLPCore.set_aug_diagonal!(ipm.kkt, solver, eta_k)

        # update the homotopy parameter
        # TODO(@anton) this should be done in a specific order depending on
        #              the types of ipm.opt.barrier, solver.opts.relaxation_update,
        #              and solver.endgame_strategy

        # First update mu.
        MadNLPCore.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_old = ipm.mu
        MadNLPCore.update_barrier!(ipm.opt.barrier, solver, sc)
        mu_updated = ipm.mu != mu_old
        MadNLPCore.@debug(
            solver.logger,
            "Updated the barrier parameter from mu=$(mu_old) to mu=$(ipm.mu)"
        )

        # Next update tau
        MadNLPCore.@trace(solver.logger, "Updating the relaxation parameter.")
        MadNLPCore.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        update_sigma!(solver.opts.relaxation_update, solver.rnlp, solver)

        # Then do endgame strategy.
        kkt_error = max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc)
        if kkt_error <= solver.opts.endgame_threshold
            MadNLPCore.@trace(solver.logger, "In endgame, using requested endgame strategy.")
            do_endgame!(solver, solver.rnlp, solver.opts.endgame_strategy)
        end
        log_iter(solver.iterate_logger, solver)

        if mu_updated && solver.opts.reset_slacks_on_update
            ind_cc1 = solver.ind_cc1
            ind_cc2 = solver.ind_cc2
            ncc = get_ncc(mpcc)
            @views begin
                x1 = MadNLPCore.variable(solver.ipm.x)[ind_cc1]
                x2 = MadNLPCore.variable(solver.ipm.x)[ind_cc2]
                MadNLPCore.slack(solver.ipm.x)[(end-ncc+1):end] .=
                    min.(.-(x1 .* x2 .- get_relaxation(solver.rnlp)), -ipm.mu)
            end
        end
        MadNLPCore.@trace(ipm.logger, "Calculating the newton step.")
        # TODO(@anton) update ipm.x ipm.zl, ipm.zu
        MadNLPCore.set_aug_diagonal!(ipm.kkt, solver, eta_k)
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

function update!(stats::CCOptExecutionStats, solver::RelaxationSolver{T,VT}) where {T,VT}
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
            stats.multipliers[end-ncc+ii] *
            (stats.solution[icc2] - get_lvar(solver.mpcc)[icc2])
        stats.multipliers_x2[ii] =
            stats.multipliers_L[icc2] -
            stats.multipliers[end-ncc+ii] *
            (stats.solution[icc1] - get_lvar(solver.mpcc)[icc1])
    end
    stats.objective = MadNLPCore.unpack_obj(ipm.cb, ipm.obj_val)
    MadNLPCore.unpack_cons!(stats.constraints, ipm.cb, ipm.c)
    stats.constraints .+= ipm.rhs
    @views stats.constraints[ipm.ind_ineq] .+= MadNLPCore.slack(ipm.x)
    # Cut out scholtes constraints now we don't need them to calculate multipliers
    resize!(stats.multipliers, m)
    resize!(stats.constraints, m)
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

function MadNLPCore.print_summary(solver::RelaxationSolver)
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
