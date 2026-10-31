function CCOptExecutionStats(solver::RelaxationSolver)
    n, m = get_nvar(solver.rnlp), get_ncon(solver.rnlp)
    ncc = get_ncc(solver.mpcc)
    VT = typeof(get_x0(solver.rnlp))
    x = similar(VT, n)
    zl = similar(VT, n)
    zu = similar(VT, n)
    zx1 = similar(VT, ncc)
    zx2 = similar(VT, ncc)
    c = similar(VT, m)
    y = similar(VT, m)
    n = MadNLPCore.get_nvar(solver.ipm.nlp)
    m = get_ncon(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2

    return CCOptExecutionStats(
        solver.ipm.opt,
        solver.opts,
        solver.ipm.status,
        MadNLPCore.unpack_obj(solver.ipm.cb, solver.ipm.obj_val),
        x,
        c,
        y,
        zl,
        zu,
        zx1,
        zx2,
        solver.ipm.inf_du,
        solver.ipm.inf_pr,
        solver.inf_pr_cc,
        0,
        solver.cnt,
    )
end

# TODO(@anton) this is a hack, fix it
function SolverCore.getStatus(result::CCOptExecutionStats)
    return MadNLPCore.get_status_output(result.status, result.options)
end

function log_iter(
    logger::IterateLogger,
    solver::RelaxationSolver{T,VT};
    magic = false,
) where {T,VT}
    if isnothing(logger.file)
        return nothing
    end
    ipm = solver.ipm
    ncc = get_ncc(solver.mpcc)

    k = ipm.cnt.k
    x0 = MadNLPCore.variable(ipm.x)[get_ind_x(solver.mpcc)]
    x1 = MadNLPCore.variable(ipm.x)[solver.ind_cc1]
    x2 = MadNLPCore.variable(ipm.x)[solver.ind_cc2]
    s = MadNLPCore.slack(ipm.x)[(end-ncc+1):end]

    z1 = MadNLPCore.variable(ipm.zl)[solver.ind_cc1]
    z2 = MadNLPCore.variable(ipm.zl)[solver.ind_cc2]
    zs = MadNLPCore.slack(ipm.zu)[(end-ncc+1):end]

    y = ipm.y

    p = ipm.p

    alpha_pr = ipm.alpha
    alpha_du = ipm.alpha_z

    ls = ipm.cnt.l

    obj = ipm.obj_val
    inf_pr = ipm.inf_pr
    inf_du = ipm.inf_du
    inf_compl = ipm.inf_compl
    inf_rnlp = MadNLPCore.get_inf_pr(ipm.c)
    inf_pr_cc = solver.inf_pr_cc

    theta = MadNLPCore.get_theta(ipm.c)
    varphi = MadNLPCore.get_varphi(ipm.obj_val, ipm.x_lr, ipm.xl_r, ipm.xu_r, ipm.x_ur, ipm.mu)

    mu = ipm.mu
    sigma = get_relaxation(solver.rnlp)
    nu1 = solver.multipliers_cc1
    nu2 = solver.multipliers_cc2
    nu1_filt = solver.multipliers_cc1_filt
    nu2_filt = solver.multipliers_cc2_filt
    delta1 = solver.rnlp.δ1
    delta2 = solver.rnlp.δ2

    W = ipm.kkt.aug_com
    K = Array(Symmetric(W, :L));
    KKT_s = eigvals(K);

    iter = RelaxationIterate(
        k,
        x0,
        x1,
        x2,
        s,
        z1,
        z2,
        zs,
        y,
        p,
        alpha_pr,
        alpha_du,
        ls,
        obj,
        inf_pr,
        inf_du,
        inf_compl,
        inf_rnlp,
        inf_pr_cc,
        theta,
        varphi,
        mu,
        sigma,
        nu1,
        nu2,
        nu1_filt,
        nu2_filt,
        delta1,
        delta2,
        solver.cnt.counters.factorization_cnt,
        KKT_s,
        magic,
    )
    serialize(logger.file, iter)

    return nothing
end

function finalize(logger::IterateLogger)
    if isnothing(logger.file)
        return nothing
    end

    return close(logger.file)
end

function get_inf_pr_cc(solver::RelaxationSolver{T}) where {T}
    @views begin
        map!(
            (a, la, b, lb) -> max((a-la)*(b-lb), la-a, lb-b),
            solver._cc1,
            MadNLPCore.variable(solver.ipm.x)[solver.ind_cc1],
            get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)],
            MadNLPCore.variable(solver.ipm.x)[solver.ind_cc2],
            get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)],
        )
    end
    return reduce(max, solver._cc1; init = zero(T))
end

function MadNLPCore.print_iter(solver::RelaxationSolver; is_resto = false)
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    mod(ipm.cnt.k, 10)==0 && MadNLPCore.@info(
        ipm.logger,
        @sprintf(
            "iter    objective    inf_pr  inf_rnlp   inf_du inf_comp lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls lg(σ)   inf_cc"
        )
    )
    if is_resto
        RR = ipm.RR::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = ipm.inf_du
        inf_pr = ipm.inf_pr
        inf_compl = ipm.inf_compl
        mu = log10(ipm.mu)
    end
    MadNLPCore.@info(
        ipm.logger,
        @sprintf(
            "%4i%s% 10.7e %6.2e %6.2e %6.2e %7.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i %5.1f  %6.2e",
            ipm.cnt.k,
            is_resto ? "r" : " ",
            ipm.obj_val/obj_scale,
            inf_pr,
            MadNLPCore.get_inf_pr(ipm.c),
            inf_du,
            inf_compl,
            mu,
            ipm.cnt.k == 0 ? 0.0 : norm(MadNLPCore.primal(ipm.d), Inf),
            ipm.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, ipm.del_w)),
            ipm.alpha_z,
            ipm.alpha,
            ipm.ftype,
            ipm.cnt.l,
            get_log_relaxation(solver.rnlp),
            solver.inf_pr_cc
        )
    )
    return
end
