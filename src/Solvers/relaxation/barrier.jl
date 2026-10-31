function _update_monotone!(
    barrier::MadNLPCore.AbstractBarrierUpdate{T},
    solver::RelaxationSolver{T},
    sc::T,
) where {T}
    ipm = solver.ipm
    rnlp = ipm.nlp
    ncc = get_ncc(solver.mpcc)
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
    inf_pr = ipm.inf_pr
    while (ipm.mu > max(barrier.mu_min, ipm.opt.tol/10)) &&
        (max(inf_pr, ipm.inf_du, inf_compl_mu) <= ipm.opt.barrier_tol_factor*ipm.mu)
        mu_new = MadNLPCore.get_mu(
            ipm.mu,
            barrier.mu_min,
            barrier.mu_linear_decrease_factor,
            barrier.mu_superlinear_decrease_power,
            ipm.opt.tol,
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
        ipm.tau = MadNLPCore.get_tau(ipm.mu, ipm.opt.tau_min)
        ipm.mu = mu_new

        # calculate new sigma
        sigma_candidate = sigma_from_mu(solver, solver.opts.relaxation_update, ipm.mu)
        inf_relaxed_cc = zero(T)
        sigma_old = get_relaxation(rnlp)
        for ii in 1:ncc
            inf_relaxed_cc =
                max(inf_relaxed_cc, abs(ipm.c[end-ncc+ii]+sigma_old[ii]-sigma_candidate))
        end
        inf_pr = max(inf_pr, inf_relaxed_cc)
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    return
end

function MadNLPCore.update_barrier!(
    barrier::MadNLPCore.MonotoneUpdate{T},
    solver::RelaxationSolver{T},
    sc::T,
) where {T}
    return _update_monotone!(barrier, solver, sc)
end

function MadNLPCore.get_adaptive_mu(
    solver::RelaxationSolver{T},
    barrier::MadNLPCore.LOQOUpdate{T},
) where {T}
    ipm = solver.ipm
    mu = MadNLPCore.get_average_complementarity(solver) # get average complementarity.
    min_cc = MadNLPCore.get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi), 2)^3
    return clamp(sigma * mu, barrier.mu_min, barrier.mu_max)
end

# Set RHS
function set_aug_rhs_aff!(
    solver::RelaxationSolver,
    kkt::MadNLPCore.AbstractKKTSystem,
    c::AbstractVector,
)
    ipm = solver.ipm
    px = MadNLPCore.primal(ipm.p)
    x = MadNLPCore.primal(ipm.x)
    f = MadNLPCore.primal(ipm.f)
    xl = MadNLPCore.primal(ipm.xl)
    xu = MadNLPCore.primal(ipm.xu)
    zl = MadNLPCore.full(ipm.zl)
    zu = MadNLPCore.full(ipm.zu)
    py = MadNLPCore.dual(ipm.p)
    pzl = MadNLPCore.dual_lb(ipm.p)
    pzu = MadNLPCore.dual_ub(ipm.p)

    ncon = get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)

    px .= .-f .+ zl .- zu .- ipm.jacl
    py .= .-c
    py[(ncon+1):(ncon+ncc)] .-= get_relaxation(solver.rnlp)
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r
    return
end

function MadNLPCore.set_centering_aug_rhs!(
    solver::RelaxationSolver,
    kkt::MadNLPCore.AbstractKKTSystem,
    mu,
)
    ipm = solver.ipm
    px = MadNLPCore.primal(ipm.p)
    py = MadNLPCore.dual(ipm.p)
    pzl = MadNLPCore.dual_lb(ipm.p)
    pzu = MadNLPCore.dual_ub(ipm.p)

    ncon = get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)

    px .= 0
    py .= 0
    py[(ncon+1):(ncon+ncc)] .+= mu
    pzl .= mu
    pzu .= -mu
    return
end

function MadNLPCore._check_progress(
    barrier::MadNLPCore.AbstractAdaptiveUpdate{T},
    solver::RelaxationSolver{T},
) where {T}
    # TODO(@anton): We need need to specialize the filter here if we want to be correct
    #               for now we try without.
    ipm = solver.ipm
    if !barrier.globalization
        return true
    end
    kappa_1 = T(1e-5) # filter margin width
    kappa_2 = T(1.0)  # filter margin maximum width
    # Check current progress using filter line search
    theta = MadNLPCore.get_theta(ipm.c)
    varphi = MadNLPCore.get_varphi(ipm.obj_val, ipm.x_lr, ipm.xl_r, ipm.xu_r, ipm.x_ur, ipm.mu)
    kkt_error = max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc)
    delta = kappa_1 * min(kappa_2, kkt_error)
    return MadNLPCore.is_filter_acceptable(ipm.filter, theta + delta, varphi + delta)
end

function MadNLPCore.get_fixed_mu(
    solver::RelaxationSolver{T},
    barrier::MadNLPCore.AbstractAdaptiveUpdate{T},
) where {T}
    # TODO(@anton) This maybe only makes sense if we have the proportional sigma update
    mu = T(0.8) * MadNLPCore.get_average_complementarity(solver)
    return clamp(mu, barrier.mu_min, barrier.mu_max)
end

function MadNLPCore._evaluate_quality_function(
    solver::RelaxationSolver,
    sigma,
    step_aff,
    step_cen,
    res_dual,
    res_primal,
)
    ipm = solver.ipm
    n, m = ipm.n, ipm.m
    ncc = get_ncc(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    nlb, nub = ipm.nlb, ipm.nub
    tau = ipm.tau
    d = ipm.d # Load buffer
    x_pr = MadNLPCore.variable(ipm.x)
    xl_pr = MadNLPCore.variable(ipm.xl)
    dx_pr = MadNLPCore.primal(ipm.d)

    # Δ(σ) = Δ(0) + σ (Δ(1) - Δ(0))
    MadNLPCore.full(d) .=
        MadNLPCore.full(step_aff) .+ sigma .* (MadNLPCore.full(step_cen) .- MadNLPCore.full(step_aff))
    # Primal step
    alpha_pr = MadNLPCore.get_alpha_max(
        MadNLPCore.primal(ipm.x),
        MadNLPCore.primal(ipm.xl),
        MadNLPCore.primal(ipm.xu),
        MadNLPCore.primal(d),
        tau,
    )
    # Dual step
    alpha_du =
        MadNLPCore.get_alpha_z(ipm.zl_r, ipm.zu_r, MadNLPCore.dual_lb(d), MadNLPCore.dual_ub(d), tau)

    # (x + αp Δx - xl)ᵀ (zl + αd Δzl)
    inf_compl_lb = mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        ipm.x_lr,
        ipm.xl_r,
        ipm.dx_lr,
        ipm.zl_r,
        MadNLPCore.dual_lb(d);
        init = 0.0,
    )
    # (xu - x - αp Δx)ᵀ (zu + αd Δzu)
    inf_compl_ub = mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        ipm.x_ur,
        ipm.xu_r,
        ipm.dx_ur,
        ipm.zu_r,
        MadNLPCore.dual_ub(d);
        init = 0.0,
    )
    @views begin
        inf_compl_pr =
            mapreduce(
                (x1, x1l, x2, x2l, dx1, dx2) ->
                    ((x1 + (alpha_pr * dx1) - x1l) * (x2 + (alpha_pr * dx2) - x2l))^2,
                +,
                x_pr[ind_cc1],
                xl_pr[ind_cc1],
                x_pr[ind_cc2],
                xl_pr[ind_cc2],
                dx_pr[ind_cc1],
                dx_pr[ind_cc2];
                init = 0.0,
            ) / ncc
    end

    # Primal infeasibility
    inf_pr = (1.0 - alpha_pr)^2 * res_primal^2 / n
    # Dual infeasibility
    inf_du = (1.0 - alpha_du)^2 * res_dual^2 / (m-ncc)
    # Complementarity infeasibility
    inf_compl = (inf_compl_lb + inf_compl_ub) / (nlb + nub)

    # Quality function qL defined in Eq. (4.2)
    return inf_du + inf_pr + inf_compl + inf_compl_pr
end

function MadNLPCore._run_golden_search!(
    solver::RelaxationSolver,
    barrier,
    sigma_lb,
    sigma_ub,
    step_aff,
    step_cen,
    res_primal,
    res_dual,
)
    ipm = solver.ipm
    gfac = 0.5 * (3.0 - sqrt(5.0))

    sigma_1, sigma_2 = sigma_lb, sigma_ub
    phi_1 = MadNLPCore._evaluate_quality_function(
        solver,
        sigma_1,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    phi_2 = MadNLPCore._evaluate_quality_function(
        solver,
        sigma_2,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )

    sigma_mid1 = sigma_lb + gfac * (sigma_ub - sigma_lb)
    sigma_mid2 = sigma_lb + (1.0 - gfac) * (sigma_ub - sigma_lb)
    phi_mid1 = MadNLPCore._evaluate_quality_function(
        solver,
        sigma_mid1,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    phi_mid2 = MadNLPCore._evaluate_quality_function(
        solver,
        sigma_mid2,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )

    # Golden search
    for i in 1:barrier.max_gs_iter
        if phi_mid1 > phi_mid2
            sigma_1 = sigma_mid1
            phi_1 = phi_mid1
            sigma_mid1 = sigma_mid2
            sigma_mid2 = sigma_1 + (1.0 - gfac) * (sigma_2 - sigma_1)
            phi_mid2 = MadNLPCore._evaluate_quality_function(
                solver,
                sigma_mid2,
                step_aff,
                step_cen,
                res_primal,
                res_dual,
            )
        else
            sigma_2 = sigma_mid2
            phi_2 = phi_mid2
            sigma_mid2 = sigma_mid1
            sigma_mid1 = sigma_1 + gfac * (sigma_2 - sigma_1)
            phi_mid1 = MadNLPCore._evaluate_quality_function(
                solver,
                sigma_mid1,
                step_aff,
                step_cen,
                res_primal,
                res_dual,
            )
        end

        if sigma_2 - sigma_1 < barrier.sigma_tol * sigma_2
            break
        end
    end
    # Compute final sigma
    sigma, phi = phi_mid1 < phi_mid2 ? (sigma_mid1, phi_mid1) : (sigma_mid2, phi_mid2)
    return sigma
end

function MadNLPCore.get_adaptive_mu(
    solver::RelaxationSolver,
    barrier::MadNLPCore.QualityFunctionUpdate,
)
    ipm = solver.ipm
    linear_solver = ipm.kkt.linear_solver
    step_aff = ipm._w1 # buffer 1
    step_cen = ipm._w2 # buffer 2
    # Affine step
    set_aug_rhs_aff!(solver, ipm.kkt, ipm.c)
    # Get primal and dual infeasibility directly 1from the values in RHS p
    res_primal = norm(@view(MadNLPCore.dual(ipm.p)[1:get_ncon(solver.mpcc)]))
    res_dual = norm(MadNLPCore.primal(ipm.p))

    # Get approximate solution without iterative refinement
    copyto!(MadNLPCore.full(step_aff), MadNLPCore.full(ipm.p))
    MadNLPCore.solve_kkt!(ipm.kkt, step_aff)

    # Get average complementarity
    mu = MadNLPCore.get_average_complementarity(solver)
    # Centering step
    MadNLPCore.set_centering_aug_rhs!(solver, ipm.kkt, mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    MadNLPCore.dual_inf_perturbation!(
        MadNLPCore.primal(ipm.p),
        ipm.ind_llb,
        ipm.ind_uub,
        mu,
        ipm.opt.kappa_d,
    )
    # Get (again) approximate solution without iterative refinement
    copyto!(MadNLPCore.full(step_cen), MadNLPCore.full(ipm.p))
    MadNLPCore.solve_kkt!(ipm.kkt, step_cen)

    # Refine the search interval using Ipopt's heuristics
    # First, check if sigma is greater than 1.
    phi1 = MadNLPCore._evaluate_quality_function(
        solver,
        1.0,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    sigma_1m = 1.0 - 1e-4
    phi1m = MadNLPCore._evaluate_quality_function(
        solver,
        sigma_1m,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    # Restrict search interval
    if phi1m > phi1
        sigma_min = 1.0
        sigma_max = min(barrier.sigma_max, barrier.mu_max / mu)
    else
        sigma_min = max(barrier.sigma_min, barrier.mu_min / mu)
        sigma_max = min(max(sigma_min, sigma_1m), barrier.mu_max / mu)
    end

    # Run Golden-section search (assume the quality function is unimodal)
    sigma_opt = MadNLPCore._run_golden_search!(
        solver,
        barrier,
        sigma_min,
        sigma_max,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    return clamp(sigma_opt * mu, barrier.mu_min, barrier.mu_max)
end

function MadNLPCore.update_barrier!(
    barrier::MadNLPCore.AbstractAdaptiveUpdate{T},
    solver::RelaxationSolver{T},
    sc::T,
) where {T}
    ipm = solver.ipm
    if !solver.opts.use_specialized_barrier_update
        return MadNLPCore.update_barrier!(barrier, ipm, sc)
    end
    is_barrier_updated = false
    progress = MadNLPCore._check_progress(barrier, solver)
    # Update state of barrier algorithm
    if !barrier.free_mode
        if progress
            MadNLPCore.@trace(solver.logger, "Moving adaptive barrier back to free mode.")
            barrier.free_mode = true
        else
            MadNLPCore._update_monotone!(barrier, ipm, sc)
        end
    else
        if !progress
            MadNLPCore.@trace(solver.logger, "Moving adaptive barrier to monotone mode.")
            barrier.free_mode = false
            # Reset barrier parameter using current average complementarity
            ipm.mu = MadNLPCore.get_fixed_mu(solver, barrier)
            is_barrier_updated = true
        else
            MadNLPCore.@trace(solver.logger, "Keeping adaptive barrier in free mode.")
        end
    end
    if barrier.free_mode
        ipm.mu = MadNLPCore.get_adaptive_mu(solver, barrier)
        is_barrier_updated = true
    end
    # Update tau and reset filter is barrier has been updated
    if is_barrier_updated
        ipm.tau = MadNLPCore.get_tau(ipm.mu, ipm.opt.tau_min)
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    return
end
