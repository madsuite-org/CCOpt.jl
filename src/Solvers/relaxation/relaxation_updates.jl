function update_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = sigma_from_mu(solver, relax, solver.ipm.mu)
    if relax.monotone
        set_relaxation(rnlp, min(solver.rnlp.σ[1], sigma_candidate))
    else
        set_relaxation(rnlp, sigma_candidate)
    end
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end

function update_sigma!(
    relax::RolloffRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = sigma_from_mu(solver, relax, solver.ipm.mu)
    if relax.monotone
        set_relaxation(rnlp, min(solver.rnlp.σ[1], sigma_candidate))
    else
        set_relaxation(rnlp, sigma_candidate)
    end
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end

function get_sigma(
    sigma,
    sigma_min,
    sigma_linear_decrease_factor,
    sigma_superlinear_decrease_power,
    tol,
)
    a = min(99.0 * sigma_min / tol, 0.01)
    return max(
        sigma_min,
        a * tol,
        min(sigma_linear_decrease_factor*sigma, sigma^sigma_superlinear_decrease_power),
    )
end

function get_accuracy(relax::ScheduledRelaxationUpdate{T}, sigma::T) where {T}
    if sigma >= relax.accuracy_shoulder
        return relax.base_accuracy
    else
        return sigma*(relax.base_accuracy/relax.accuracy_shoulder)
    end
end

function update_sigma!(
    relax::ScheduledRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    rnlp = ipm.nlp
    ncc = get_ncc(solver.mpcc)
    sc = MadNLPCore.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
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
    sigma = get_relaxation(rnlp)[1]
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    accuracy = get_accuracy(relax, sigma)
    while (sigma > max(relax.sigma_min, ipm.opt.tol/10)) &&
        (max(inf_pr, ipm.inf_du, inf_compl_mu) <= accuracy)
        sigma_new = get_sigma(
            sigma,
            relax.sigma_min,
            relax.sigma_linear_decrease_factor,
            relax.sigma_superlinear_decrease_power,
            ipm.opt.tol,
        )
        sigma = sigma_new
        accuracy = get_accuracy(relax, sigma)
        ipm.mu = accuracy # This is a hack but seems to work?
        @views begin
            inf_relaxed_cc =
                mapreduce((c)->abs(c-sigma_new), max, ipm.c[(end-ncc+1):end]; init = 0)
        end
        inf_pr = max(inf_pr, inf_relaxed_cc)
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
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    set_relaxation(rnlp, sigma)
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    return
end

function update_sigma!(
    relax::LOQORelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # Calculate mean primal complementarity
    cc_pr = @views dot(
        MadNLPCore.variable(ipm.x)[solver.ind_cc1] - MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2] - MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
    )
    mean_cc = cc_pr/ncc
    # Calculate minimum primal complementarity
    min_cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLPCore.variable(ipm.x)[solver.ind_cc1],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
        init = T(Inf),
    )
    # Calculate the factor to multiply the mean complementarity by.
    xi = min_cc_pr/mean_cc
    gamma_sigma = max(relax.gamma_min, relax.gamma*min((1-relax.r)*((1-xi)/xi), 2)^3)
    # TODO(@anton) in principle we would like to not reduce this too much depending on how close we are to the KKT conds
    set_relaxation(rnlp, max(gamma_sigma*mean_cc, relax.sigma_min, relax.mu_factor*ipm.mu))
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    # Throw out the filter as the barrier problem has changed
    empty!(ipm.filter)
    push!(ipm.filter, (ipm.theta_max, -Inf))
    return nothing
end

function kkt_residual_norm(
    rnlp::ScholtesRelaxation{T},
    solver::RelaxationSolver{T},
    δ1,
    δ2,
    σ,
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncon = get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    # Reset bounds:
    # NOTE(@anton): HERE BE DRAGONS! Due to using the original lvars we need to use two different
    #               index sets, due to fixed variable treatment
    MadNLPCore.variable(ipm.xl)[ind_cc1] .= @view(get_lvar(mpcc)[get_ind_cc1(mpcc)]) .- δ1
    MadNLPCore.variable(ipm.xl)[ind_cc2] .= @view(get_lvar(mpcc)[get_ind_cc2(mpcc)]) .- δ2
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
    c = ipm.c

    px .= .-f .+ zl .- zu .- ipm.jacl
    py .= .-c
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r

    return r
end

function update_sigma!(
    relax::TwoSidedScholtesUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    # TODO(@anton) This is inefficient
    r = kkt_residual_norm(rnlp, solver, rnlp.δ1opt, rnlp.δ2opt, rnlp.σopt) # kkt norm
    rl = r^(1+relax.tau)
    ru = r^(1-relax.tau)
    MadNLPCore.variable(ipm.xl)[ind_cc1] .= @view(get_lvar(mpcc)[get_ind_cc1(mpcc)]) .- rnlp.δ1
    MadNLPCore.variable(ipm.xl)[ind_cc2] .= @view(get_lvar(mpcc)[get_ind_cc2(mpcc)]) .- rnlp.δ2
    updated = false
    (@view ipm.c[(end-ncc+1):end]) .+= rnlp.σ
    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        cc1_orig = get_ind_cc1(mpcc)[ii]
        cc2_orig = get_ind_cc2(mpcc)[ii]

        nu1 = solver.multipliers_cc1[ii]
        nu2 = solver.multipliers_cc2[ii]
        # These are the rules from
        if nu1 > ru
            rnlp.δ1[ii] = min(relax.kappa*rnlp.δ1[ii], rl)
            MadNLPCore.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]
            rnlp.δ1opt[ii] = 0.0
            updated = true
        else
            rnlp.δ1opt[ii] = rnlp.δ1[ii]
        end

        if nu2 > ru
            rnlp.δ2[ii] = min(relax.kappa*rnlp.δ2[ii], rl)
            MadNLPCore.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]
            rnlp.δ2opt[ii] = 0.0
            updated = true
        else
            rnlp.δ2opt[ii] = rnlp.δ2[ii]
        end

        if nu1 < -ru || nu2 < -ru
            if min(relax.kappa*rnlp.σ[ii], rl) >= relax.sigma_min
                rnlp.σ[ii] = max(min(relax.kappa*rnlp.σ[ii], rl), relax.sigma_min)
                rnlp.σopt[ii] = 0.0
                updated = true
            end
        else
            rnlp.σopt[ii] = rnlp.σ[ii]
        end
    end
    if !updated && ipm.mu <= 1e-4 # Standard rules make no progress
        for ii in 1:ncc
            cc1 = ind_cc1[ii]
            cc2 = ind_cc2[ii]
            cc1_orig = get_ind_cc1(mpcc)[ii]
            cc2_orig = get_ind_cc2(mpcc)[ii]
            x1 = MadNLPCore.variable(ipm.x)[cc1] - get_lvar(mpcc)[cc1_orig]
            x2 = MadNLPCore.variable(ipm.x)[cc2] - get_lvar(mpcc)[cc2_orig]

            if x1 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLPCore.variable(ipm.x)[cc1] - MadNLPCore.variable(ipm.xl)[cc1])
                rnlp.δ1[ii] = max(relax.kappa*rnlp.δ1[ii], rnlp.δ1[ii]-max_decrease)
                MadNLPCore.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]
                updated = true
            end
            if x2 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLPCore.variable(ipm.x)[cc2] - MadNLPCore.variable(ipm.xl)[cc2])
                rnlp.δ2[ii] = max(relax.kappa*rnlp.δ2[ii], rnlp.δ2[ii]-max_decrease)
                MadNLPCore.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]
                updated = true
            end
            if x1 >= 0 && x2 >= 0 && x1*x2 >= solver.ipm.opt.tol
                rnlp.σ[ii] = max(relax.kappa*rnlp.σ[ii], relax.sigma_min)
                updated = true
            end
        end
    end
    (@view ipm.c[(end-ncc+1):end]) .-= rnlp.σ
    if updated
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
end

function init_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    set_relaxation(rnlp, sigma_from_mu(solver, relax, solver.ipm.mu))
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    return nothing
end

function init_sigma!(
    relax::RolloffRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    set_relaxation(rnlp, sigma_from_mu(solver, relax, solver.ipm.mu))
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    return nothing
end

function init_sigma!(
    relax::LOQORelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # Calculate mean primal complementarity
    cc_pr = @views dot(
        MadNLPCore.variable(ipm.x)[solver.ind_cc1] - MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2] - MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
    )
    mean_cc = cc_pr/ncc
    # Calculate minimum primal complementarity
    min_cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLPCore.variable(ipm.x)[solver.ind_cc1],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
        init = T(Inf),
    )
    # Calculate the factor to multiply the mean complementarity by.
    xi = min_cc_pr/mean_cc
    gamma_sigma = max(relax.gamma_min, relax.gamma*min((1-relax.r)*((1-xi)/xi), 2)^3)
    # TODO(@anton) in principle we would like to not reduce this too much depending on how close we are to the KKT conds
    set_relaxation(rnlp, max(gamma_sigma*mean_cc, relax.sigma_min, relax.mu_factor*ipm.mu))
    # update c
    (@view c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    # Throw out the filter as the barrier problem has changed
    empty!(ipm.filter)
    push!(ipm.filter, (ipm.theta_max, -Inf))
    return nothing
end

function init_sigma!(
    relax::TwoSidedScholtesUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    set_relaxation(rnlp, max(sigma_candidate, relax.sigma_min))
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    return nothing
end

function init_sigma!(
    relax::ScheduledRelaxationUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::RelaxationSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .+= get_relaxation(rnlp)
    # calculate new sigma
    set_relaxation(rnlp, relax.sigma_init)
    # update c
    (@view ipm.c[(end-ncc+1):end]) .-= get_relaxation(rnlp)
    return nothing
end

# TODO(@anton) this is a hack. Refactor parameter updates entirely at some point
function sigma_from_mu(
    solver::RelaxationSolver{T},
    relax::AbstractAdaptiveRelaxationUpdate{T},
    mu::T,
) where {T}
    return get_relaxation(solver.rnlp)[1] # Assume all the same for now
end

function sigma_from_mu(
    solver::RelaxationSolver{T},
    relax::ProportionalRelaxationUpdate{T},
    mu::T,
) where {T}
    sigma_candidate = relax.sigma_mu_ratio*(mu^relax.sigma_mu_exp)
    return max(sigma_candidate, relax.sigma_min)
end

function sigma_from_mu(
    solver::RelaxationSolver{T},
    relax::RolloffRelaxationUpdate{T},
    mu::T,
) where {T}
    sigma_candidate =
        relax.rolloff_max*(
            mu^relax.rolloff_slope
        )/(sqrt((mu^relax.rolloff_slope)^2) + relax.rolloff_point)
    return max(sigma_candidate, relax.sigma_min)
end
