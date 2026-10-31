@kwdef struct PenaltyOptions{T} <: MadNLPCore.AbstractOptions
    penalty::Type = Ell1Relaxation
    # complementarity homotopy options
    rho_0::T = 1.0
    rho_max::T = 1e10
    rho_growth_rate::T = 10.0
    gamma::T = 0.4

    # Algorithm options
    dynamic_rho_update::Bool = true # Switch between classic and dynamic algorithm from
    # Leyffer2006 paper
    comp_history_length::Int = 10 # Length of history buffer (default from Leyffer2006)
    eta_dynamic_update::T = 0.99 # "sufficient decrease" parameter (default from Leyffer2006)

    # regularization
    q_regularization::Symbol = :critical_rho
    min_eig_value::T = 1e-8
    max_eig_value::T = Inf
    critical_rho_factor::T = 0.99
    min_reg_mu::T = 5e-6
    max_reg_mu::T = 1e-1

    # Output options
    output_file::String = ""
    print_level::MadNLPCore.LogLevels = MadNLPCore.INFO
    file_print_level::MadNLPCore.LogLevels = MadNLPCore.INFO
end

mutable struct PenaltySolver{
    T,
    VT,
    MPCC<:AbstractMPCCModel{T,VT},
    PNLP<:AbstractMPCCPenaltyModel{T,VT},
    SOLVER<:MadNLPCore.MadNLPSolver{T,VT},
}
    const mpcc::MPCC
    const pnlp::PNLP
    const ipm::SOLVER
    const cnt::CCOptCounters
    const logger::MadNLPCore.MadNLPLogger
    const opts::PenaltyOptions{T}

    inf_pr_cc::T
    const ind_cc1::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc2::Vector{Int} # fixed indices in case of MakeParameter

    const pr_comp_hist::CircularBuffer{T} # Complementarity history
end

function CCOptExecutionStats(solver::PenaltySolver)
    n, m = get_nvar(solver.pnlp), get_ncon(solver.pnlp)
    ncc = get_ncc(solver.mpcc)
    VT = typeof(get_x0(solver.pnlp))
    x = similar(VT, n)
    zl = similar(VT, n)
    zu = similar(VT, n)
    zx1 = similar(VT, ncc)
    zx2 = similar(VT, ncc)
    c = similar(VT, m)
    y = similar(VT, m)
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

"""
  This solver implements the two algorithms in the paper:
  Interior Methods for Mathematical Programs with Complementarity Constraints
  Sven Leyffer, Gabriel López-Calva, and Jorge Nocedal
  SIAM Journal on Optimization Vol. 17, Issue 1 (2006)
  epubs.siam.org/doi/10.1137/040621065
"""
function PenaltySolver(
    mpcc::AbstractMPCCModel{T,VT};
    solver_opts = PenaltyOptions(),
    ipm_options...,
) where {T,VT}
    start_time = time()
    pnlp = solver_opts.penalty(mpcc)
    ipm = MadNLPCore.MadNLPSolver(pnlp; ipm_options...)
    cnt = CCOptCounters(counters = ipm.cnt)
    logger = MadNLPCore.MadNLPLogger(
        print_level = solver_opts.print_level,
        file_print_level = solver_opts.file_print_level,
        file = solver_opts.output_file == "" ? nothing :
               open(solver_opts.output_file, "w+"),
    )

    pr_comp_hist = CircularBuffer{T}(solver_opts.comp_history_length)
    ind_cc1 = copy(get_ind_cc1(mpcc))
    ind_cc2 = copy(get_ind_cc2(mpcc))
    _adjust_cc_inds!(ipm.cb, ind_cc1, ind_cc2)
    solver = PenaltySolver(
        mpcc,
        pnlp,
        ipm,
        cnt,
        logger,
        solver_opts,
        0.0,
        ind_cc1,
        ind_cc2,
        pr_comp_hist,
    )
    cnt.init_time = time() - start_time
    return solver
end
include("utils.jl")
include("exact_penalty.jl")
