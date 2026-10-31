function MadNLPCore.print_iter(solver::PenaltySolver{T,VT}; is_resto = false) where {T,VT}
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    mod(ipm.cnt.k, 10)==0 && MadNLPCore.@info(
        ipm.logger,
        @sprintf(
            "iter    objective    inf_pr   inf_du inf_compl lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls lg(τ)   inf_cc"
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
            "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i %5.1f  %6.2e",
            ipm.cnt.k,
            is_resto ? "r" : " ",
            ipm.obj_val/obj_scale,
            inf_pr,
            inf_du,
            inf_compl,
            mu,
            ipm.cnt.k == 0 ? 0.0 : norm(MadNLPCore.primal(ipm.d), Inf),
            ipm.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, ipm.del_w)),
            ipm.alpha_z,
            ipm.alpha,
            ipm.ftype,
            ipm.cnt.l,
            get_log_penalty(solver.pnlp),
            solver.inf_pr_cc
        )
    )
    return
end

function get_inf_pr_cc(solver::PenaltySolver{T,VT}) where {T,VT}
    inf_pr_cc = zero(T)
    @views begin
        a = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc1]
        la = get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)]
        b = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc2]
        lb = get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)]
        for ii in eachindex(a)
            inf_pr_cc = max(
                inf_pr_cc,
                max(min(a[ii]-la[ii], b[ii]-lb[ii]), la[ii]-a[ii], lb[ii]-b[ii]),
            )
        end
    end
    return inf_pr_cc
end

function get_inf_pr_cc_prod(solver::PenaltySolver{T,VT}) where {T,VT}
    inf_pr_cc_prod = zero(T)
    @views begin
        a = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc1]
        la = get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)]
        b = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc2]
        lb = get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)]
        for ii in eachindex(a)
            inf_pr_cc_prod = max(
                inf_pr_cc_prod,
                max((a[ii]-la[ii])*(b[ii]-lb[ii]), la[ii]-a[ii], lb[ii]-b[ii]),
            )
        end
    end
    return inf_pr_cc_prod
end

function get_inf_pr_cc_sum(solver::PenaltySolver{T,VT}) where {T,VT}
    inf_pr_cc_sum = zero(T)
    @views begin
        a = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc1]
        la = get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)]
        b = MadNLPCore.variable(solver.ipm.x)[solver.ind_cc2]
        lb = get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)]
        for ii in eachindex(a)
            inf_pr_cc_sum += max((a[ii]-la[ii])*(b[ii]-lb[ii]), la[ii]-a[ii], lb[ii]-b[ii])
        end
    end
    return inf_pr_cc_sum
end
