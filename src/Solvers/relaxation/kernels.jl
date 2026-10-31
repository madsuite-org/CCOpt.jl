function MadNLPCore.get_average_complementarity(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = ipm.nlb + ipm.nub + get_ncc(mpcc)
    cc_lb = dot(ipm.x_lr, ipm.zl_r) - dot(ipm.xl_r, ipm.zl_r)
    cc_ub = dot(ipm.xu_r, ipm.zu_r) - dot(ipm.x_ur, ipm.zu_r)
    # We know mpcc is in vertical form, therefore:
    # TODO(@anton) add view for x1, x2
    cc_pr = @views dot(
        MadNLPCore.variable(ipm.x)[solver.ind_cc1] - MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2] - MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
    )
    return (cc_lb + cc_ub + cc_pr)/ncc
end

function MadNLPCore.get_min_complementarity(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    cc_lb = mapreduce(
        (x_l, xl, zl) -> (x_l-xl)*zl,
        min,
        ipm.x_lr,
        ipm.xl_r,
        ipm.zl_r,
        init = T(Inf),
    )
    cc_ub = mapreduce(
        (x_u, xu, zu) -> (xu-x_u)*zu,
        min,
        ipm.x_ur,
        ipm.xu_r,
        ipm.zu_r,
        init = T(Inf),
    )
    # We know mpcc is in vertical form, therefore:
    cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLPCore.variable(ipm.x)[solver.ind_cc1],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc1],
        MadNLPCore.variable(ipm.x)[solver.ind_cc2],
        MadNLPCore.variable(ipm.xl)[solver.ind_cc2],
        init = T(Inf),
    )
    return min(cc_lb, cc_ub, cc_pr)
end

@inline function update_c!(c, σ, σ_old, ncc)
    return c[(end-ncc+1):end] .+= σ_old - σ
end

function estimate_mpec_multipliers(solver::RelaxationSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    ind_cc1_orig = mpcc.meta.ind_cc1
    ind_cc2_orig = mpcc.meta.ind_cc2

    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        cc1_orig = ind_cc1_orig[ii]
        cc2_orig = ind_cc2_orig[ii]
        x1 = MadNLPCore.variable(ipm.x)[cc1] - get_lvar(mpcc)[cc1_orig]
        z1 = MadNLPCore.variable(ipm.zl)[cc1]
        x2 = MadNLPCore.variable(ipm.x)[cc2] - get_lvar(mpcc)[cc2_orig]
        z2 = MadNLPCore.variable(ipm.zl)[cc2]
        zs = MadNLPCore.slack(ipm.zu)[end-ncc+ii]

        f1 = MadNLPCore.variable(ipm.f)[cc1]
        f2 = MadNLPCore.variable(ipm.f)[cc2]
        jac1 = ipm.jacl[cc1]
        jac2 = ipm.jacl[cc2]

        solver.multipliers_cc1[ii] = z1 - zs*x2
        solver.multipliers_cc2[ii] = z2 - zs*x1
    end
end
