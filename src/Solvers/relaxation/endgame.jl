# Default to do nothing
function do_endgame!(
    solver::RelaxationSolver,
    rnlp::RNLP,
    endgame::NoEndgameStrategy,
) where {RNLP<:AbstractMPCCRelaxation} end

## Lower bound relaxation endgame
# Do nothing for generic RNLPs
# TODO(@anton) we could do something.
function do_endgame(
    solver::RelaxationSolver,
    rnlp::RNLP,
    endgame::RelaxLBEndgameStrategy,
) where {RNLP<:AbstractMPCCRelaxation} end

function get_delta_candidate(nu, x, sigma, delta_max)
    if nu + x < 0.0
        return delta_max
    else
        return min(delta_max, sigma/(nu+x))
    end
end

# Calculate the new duals for a given relaxation
function calculate_relax_magic_step(x, x_other, z, zs, delta, nu, nu_other, mu)
    mu_r = mu + (x*z - mu)
    z_hat = inv(x+delta)*mu_r
    zs_hat = inv(x_other)*(-nu + z_hat)
    z_other_hat = x*zs_hat + nu_other

    return z_hat, z_other_hat, zs_hat
end

function set_relax_magic_step!(solver, ii, z1, z2, zs, x1, x2, delta_zs)
    cc1 = solver.ind_cc1[ii]
    cc2 = solver.ind_cc2[ii]
    ncc = get_ncc(solver.mpcc)
    ipm = solver.ipm
    cb = ipm.cb

    ## Set the new duals
    MadNLPCore.variable(ipm.zl)[cc1] = z1
    MadNLPCore.variable(ipm.zl)[cc2] = z2
    MadNLPCore.slack(ipm.zu)[end-ncc+ii] = zs
    ipm.y[end-ncc+ii] = zs

    ## Set the new J'y_c
    ipm.jacl[cc1] += x2*delta_zs*cb.con_scale[end-ncc+ii]
    ipm.jacl[cc2] += x1*delta_zs*cb.con_scale[end-ncc+ii]
    ipm.jacl[end-ncc+ii] -= delta_zs*cb.con_scale[end-ncc+ii]

    ## Set the multiplier contribution in the Hessian of the Lagrangian
    ipm.kkt.hess[end-ncc+ii] = zs*cb.con_scale[end-ncc+ii]
    return nothing
end

# If using Scholtes do LB relaxation
function do_endgame!(
    solver::RelaxationSolver{T},
    rnlp::ScholtesRelaxation{T},
    endgame::RelaxLBEndgameStrategy{T},
) where {T}
    ipm = solver.ipm
    cb = ipm.cb
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    mu = ipm.mu

    updated = false
    nu_bound = ipm.inf_du^endgame.tau
    # Check relaxation for each complementarity.
    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        cc1_orig = get_ind_cc1(mpcc)[ii]
        cc2_orig = get_ind_cc2(mpcc)[ii]
        nu1 = solver.multipliers_cc1[ii]
        nu2 = solver.multipliers_cc2[ii]

        x1 = MadNLPCore.variable(ipm.x)[cc1] - MadNLPCore.variable(ipm.xl)[cc1]
        z1 = MadNLPCore.variable(ipm.zl)[cc1]
        x2 = MadNLPCore.variable(ipm.x)[cc2] - MadNLPCore.variable(ipm.xl)[cc2]
        z2 = MadNLPCore.variable(ipm.zl)[cc2]
        zs = MadNLPCore.slack(ipm.zu)[end-ncc+ii]
        s = MadNLPCore.slack(ipm.x)[end-ncc+ii]

        nu1_inactive = nu1 <= -(nu_bound)
        if nu1_inactive
            delta_candidate = get_delta_candidate(nu1, x2, rnlp.σ[ii], endgame.delta_max)
            # If not enough change in delta candidate, it isn't worth perturbing newton.
            # In practice this means relaxation happens only once most of the time
            if delta_candidate < endgame.min_delta_inc_factor * rnlp.δ1[ii]
                continue
            end
            # Relax the lower bound, and take a magic step in the multipliers
            rnlp.δ1[ii] = delta_candidate
            updated = true
            MadNLPCore.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]

            # Calculate new values
            z1_hat, z2_hat, zs_hat =
                calculate_relax_magic_step(x1, x2, z1, zs, delta_candidate, nu1, nu2, mu)
            delta_zs = zs_hat - zs

            # Set new values
            ## Set the new duals
            set_relax_magic_step!(solver, ii, z1_hat, z2_hat, zs_hat, x1, x2, delta_zs)
        end

        nu2_inactive = nu2 <= -(nu_bound)
        if nu2_inactive
            delta_candidate = get_delta_candidate(nu2, x1, rnlp.σ[ii], endgame.delta_max)
            # If not enough change in delta candidate, it isn't worth perturbing newton.
            # In practice this means relaxation happens only once most of the time
            if delta_candidate < endgame.min_delta_inc_factor * rnlp.δ2[ii]
                continue
            end
            rnlp.δ2[ii] = delta_candidate
            updated = true
            MadNLPCore.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]

            # Calculate new values
            z2_hat, z1_hat, zs_hat =
                calculate_relax_magic_step(x2, x1, z2, zs, delta_candidate, nu2, nu1, mu)
            delta_zs = zs_hat - zs

            # Set new values
            set_relax_magic_step!(solver, ii, z1_hat, z2_hat, zs_hat, x1, x2, delta_zs)
        end
    end

    # Throw out the filter as the barrier problem has changed
    if updated
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    return nothing
end
