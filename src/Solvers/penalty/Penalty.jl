
"""
  Primary options structure for the `CCOpt` penalty based solver.
  This contains options related to the handling of complementarity constrants and `PenaltySolver` related logging.

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct PenaltyOptions{T<:Real} <: MadNLP.AbstractOptions
    """
    The penalty formulation to use. The `Ell1Relaxation` is the only non-experimental option.
    """
    penalty::Type = Ell1Relaxation
    # complementarity homotopy options

    raw"""
    Initial penalty parameter.

    Range ``[0.0,\infty]``
    """
    rho_0::T = 1.0

    raw"""
    Maximum penalty parameter.

    Range ``[0.0,\infty]``
    """
    rho_max::T = 1e10

    raw"""
    Factor by which the penalty is increased when stationarity is achieved without complementarity feasibility.

    Range ``(0.0,\infty]``
    """
    rho_growth_rate::T = 10.0

    raw"""
    Exponent applied to the current barrier parameter ``\mu`` to get the expected complementarity reduction.

    Range ``(0.0, 1.0]``
    """
    gamma::T = 0.4

    # Algorithm options
    """
    Whether the dynamic penalty method of Leyffer et al. is used.
    """
    dynamic_rho_update::Bool = true # Switch between classic and dynamic algorithm from

    # Leyffer2006 paper
    raw"""
    How many of the previous complementarity residuals to test for sufficient decrease.

    Range: Integers ``[1,\infty]``
    """
    comp_history_length::Int = 10 # Length of history buffer (default from Leyffer2006)

    raw"""
    Factor of sufficient decrease required to not increase the penalty value in the dynamic update algorithm.

    Range: ``(0.0,\infty)``
    """
    eta_dynamic_update::T = 0.99 # "sufficient decrease" parameter (default from Leyffer2006)

    # regularization
    """
    How to regularize the indefinite contibutiion of the complementarity constraints to the Hessian of the Lagrangian.

    - `:none`: Does no regularization.
    - `:critical_rho`: Clips the penalty for each complementarity to the critical value maintaining a threshold positive definiteness.
    - `:eigenvalue_decomposition`: Project each 2 by 2 matrix to the space of Positive semi-definite matrices, by cliping their eigenvalues to [`min_eig_value`, `max_eig_value`].
    """
    q_regularization::Symbol = :critical_rho

    raw"""
    Minimum allowed eigenvalue for the eigenvalue decomposition based `q_regularization`.

    Range: ``(0, \infty]``
    """
    min_eig_value::T = 1e-8

    raw"""
    Maximum allowed eigenvalue for the eigenvalue decomposition based `q_regularization`.

    Range: ``(0, \infty]``
    """
    max_eig_value::T = Inf

    """
    Factor of the critical multiplier to set for the regularized blocks of ``Q``.

    Range: ``(0.0, 1.0)``
    """
    critical_rho_factor::T = 0.99

    raw"""
    Lower bound of the barrier parameter below which `q_regularization` is disabled.

    Range: ``(0.0, \infty)``
    """
    min_reg_mu::T = 5e-6
    max_reg_mu::T = 1e-1

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
end

mutable struct PenaltySolver{
    T,
    VT,
    MPCC<:AbstractMPCCModel{T,VT},
    PNLP<:AbstractMPCCPenaltyModel{T,VT},
    SOLVER<:MadNLP.MadNLPSolver{T,VT},
}
    const mpcc::MPCC
    const pnlp::PNLP
    const ipm::SOLVER
    const cnt::CCOptCounters
    const logger::MadNLP.MadNLPLogger
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
        MadNLP.unpack_obj(solver.ipm.cb, solver.ipm.obj_val),
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
    logger = MadNLP.MadNLPLogger(
        print_level = solver_opts.print_level,
        file_print_level = solver_opts.file_print_level,
        file = solver_opts.output_file == "" ? nothing :
               open(solver_opts.output_file, "w+"),
    )
    # As in `RelaxationSolver`, default to no bound relaxation: with x₁, x₂ slightly below
    # their bounds the penalty term ρx₁x₂ becomes negative, which for large ρ creates
    # spurious stationary points and inaccurate MPCC multipliers.
    if haskey(ipm_options, :bound_relax_factor)
        ipm_options[:bound_relax_factor] == zero(T) || MadNLP.@warn(
            logger,
            "You have set a positive 'bound_relax_factor', this is probably a mistake! Allowing bound relaxation means complementarity variables are no longer strictly positive which can cause the algorithm to stall at non-physical, inaccurate solutions."
        )
        ipm = MadNLP.MadNLPSolver(pnlp; ipm_options...)
    else
        ipm = MadNLP.MadNLPSolver(pnlp; bound_relax_factor = zero(T), ipm_options...)
    end
    cnt = CCOptCounters(counters = ipm.cnt)

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
