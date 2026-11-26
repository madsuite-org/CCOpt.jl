@kwdef mutable struct IterateLogger
    file::Union{IOStream,Nothing} = nothing
end
# Relaxation updates

"""
  Method by which the Scholtes relaxation parameter σ is calculated at each iteration.
"""
abstract type AbstractRelaxationUpdate{T} end

"""
  Fixed relaxation update methods define a function ``σ(μ)`` which is used to calculate the
  Scholtes relaxation parameter from the barrier parameter.
"""
abstract type AbstractFixedRelaxationUpdate{T} <: AbstractRelaxationUpdate{T} end

"""
  Adaptive relaxation update methods update σ independently of μ.
"""
abstract type AbstractAdaptiveRelaxationUpdate{T} <: AbstractRelaxationUpdate{T} end

"""
  Proportional relaxation update which updates ``σ = aμ^b``

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct ProportionalRelaxationUpdate{T<:Real} <: AbstractFixedRelaxationUpdate{T}
    raw"""
    Multiplicative factor ``a``.

    Range: ``(0,\infty)``
    """
    sigma_mu_ratio::T = 1.0 # a

    raw"""
    Exponent ``b``.

    Range: ``(0,\infty)``
    """
    sigma_mu_exp::T = 1.0 # b

    """
    Enforce only monotone updates.
    """
    monotone::Bool = false

    raw"""
    Minimum relaxation factor.

    Range: ``(0,\infty)``
    """
    sigma_min::T = 1e-8
end

"""
  Rolloff Relaxation update which updates ``σ = cμ^a/(sqrt(μ^a^2)+b)``

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct RolloffRelaxationUpdate{T} <: AbstractFixedRelaxationUpdate{T}
    raw"""
    Slope of rolloff function ``a``.

    Range: ``(0,\infty)``
    """
    rolloff_slope::T = 2.0 # a

    raw"""
    Beginning of rolloff ``b``.

    Range: ``(0,\infty)``
    """
    rolloff_point::T = 1e-6 # b

    raw"""
    Initial relaxation value ``c``.

    Range: ``(0,\infty)``
    """
    rolloff_max::T = 1.0 # c

    """
    Enforce only monotone updates.
    """
    monotone::Bool = false

    raw"""
    Minimum relaxation factor.

    Range: ``(0,\infty)``
    """
    sigma_min::T = 1e-8
end

"""
  LOQO based algorithm for adaptively selecting the Scholtes relaxation parameter σ.

  ``ξ = (n_{cc}min(x₁,x₂))/(x₁ᵀx₂)``

  ``k = γ*min((1-r)(1-ξ)/ξ, 2)³``

  ``σ = k(x₁ᵀx₂)/ncc``

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct LOQORelaxationUpdate{T} <: AbstractAdaptiveRelaxationUpdate{T}
    raw"""
    Scale factor ``\gamma``.

    Range: ``(0,\infty)``
    """
    gamma::T = 0.05 # scale factor

    raw"""
    Minimum reduction factor.

    Range: ``(0,\infty)``
    """
    gamma_min::T = 1e-5 # smallest factor of reduction allowed
    mu_factor::T = 1e-5

    raw"""
    Step length parameter ``r``.

    Range: ``(0,1.0)``
    """
    r::T = 0.95 # Steplength param

    raw"""
    Minimum relaxation factor.

    Range: ``(0,\infty)``
    """
    sigma_min::T = 1e-8
end

"""
  Two-sided Scholtes relaxation attempts to drive either the scholtes bound x1*x2 - σ ≤ 0 to zero
  or the lower bounds of x1 and x2 choosing which to decrease by inspecting the estimated mpcc
  lagrange multipliers.

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct TwoSidedScholtesUpdate{T} <: AbstractAdaptiveRelaxationUpdate{T}
    kappa::T = 0.1
    k_ftb::T = 0.9
    tau::T = 0.3
    sigma_min::T = 1e-8
    # Initial relaxation σ₀ = sigma_mu_ratio * μ₀^sigma_mu_exp
    sigma_mu_ratio::T = 1.0
    sigma_mu_exp::T = 1.0
end

"""
  A scheduled update for σ, that is the Fiacco-McCormick rule is applied when the RNLP(σ) is solved to a target accuracy.
  The target accuracy is calculated as a piecewise function: base accuracy if σ > accuracy shoulder, and
  ((target accuaracy - base accuracy)/(target accuracy - accuracy shoulder))*σ otherwise.
  This is done independently of the μ (barrier parameter) update.

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct ScheduledRelaxationUpdate{T} <: AbstractAdaptiveRelaxationUpdate{T}
    base_accuracy::T = 1e-3
    target_accuracy::T = 1e-8
    accuracy_shoulder::T = 1e-4
    sigma_linear_decrease_factor::T = 0.1
    sigma_superlinear_decrease_power::T = 1.9
    sigma_init::T = 1e0
    sigma_min::T = 1e-8
end

"""
  Strategies taken once an accuracy threshold is reached in order to promote convergence.
"""
abstract type AbstractEndgameStrategy{T} end

"""
  No special strategy is used.
"""
struct NoEndgameStrategy{T} <: AbstractEndgameStrategy{T} end

"""
  Relaxes the complementarity lower bounds when:
    du_inf ≤ relax_threshold
    the corresponding estimated mpec multiplier is negative (the scholtes bound is active)
    identified by the multiplier being smaller than μ^tau.

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef mutable struct RelaxLBEndgameStrategy{T} <: AbstractEndgameStrategy{T}
    raw"""
    Maximum lower bound relaxation.

    Range: ``[0.0,\infty)``
    """
    delta_max::T = 1e-4

    raw"""
    Minimum factor of additional relaxation which is applied. If this is not exceeded then we do not modify the lower bound.

    Range: ``[0.0,\infty)``
    """
    min_delta_inc_factor::T = 1.1

    raw"""
    Exponent applied to the barrier parameter which is used as the threshold at which a bound is considered active.

    Range: ``(0.0,1.0)``
    """
    tau::T = 0.4
end

# Iterate saving structure
struct RelaxationIterate{T,VT}
    k::Int

    x0::VT
    x1::VT
    x2::VT
    s::VT

    z1::VT
    z2::VT
    zs::VT

    y::VT

    p::MadNLP.UnreducedKKTVector{T,VT}

    alpha_pr::T
    alpha_du::T

    ls::Int

    obj::T
    inf_pr::T
    inf_du::T
    inf_compl::T
    inf_rnlp::T
    inf_pr_cc::T
    theta::T
    varphi::T
    mu::T
    sigma::VT
    nu1::VT
    nu2::VT
    delta1::VT
    delta2::VT

    n_fact::Int

    KKT_s::VT

    magic::Bool
end

# Options struct
"""
  Primary options structure for the `CCOpt` relaxation based solver.
  This contains options related to the handling of complementarity constrants and `RelaxationSolver` related logging.

  # Fields:

  $(TYPEDFIELDS)
"""
@kwdef struct RelaxationOptions{
    T<:Real,
    RELAX<:AbstractRelaxationUpdate{T},
    EG<:AbstractEndgameStrategy{T},
} <: MadNLP.AbstractOptions
    """
    Type of relaxation to apply. The only non-experimental relaxation is the `ScholtesRelaxation`.
    """
    relaxation::Type = ScholtesRelaxation

    """
    Whether the adaptive barrier update rules implemented in `MadNLP` include upper level complementarity residuals.
    """
    use_specialized_barrier_update::Bool = false

    """
    Relaxation update type. See the section on [Relaxation parameter updates](@ref) for more details.
    """
    relaxation_update::RELAX = ProportionalRelaxationUpdate()

    """
    Minumum value the relaxation parameter is allowed to take.
    """
    sigma_min::T = 1e-8 # TODO(@anton) I think this should be probably be related to ipm tolerance

    """
    Initial lower bound relaxation for the TwoSidedRelaxation.
    """
    delta_init::T = 0.0

    # Endgame options
    """
    Primal-dual stationarity tolerance that needs to be achieved to activate the endgame strategy.
    """
    endgame_threshold::T = 1e-6

    """
    What endgame strategy to use to improve the robustness of solver convergence when near to the solution.
    """
    endgame_strategy::EG = NoEndgameStrategy{Float64}() # TODO(@anton) this will break for other precisions

    # initialization options
    """
    Whether the [`MadNLPOptions.bound_relaxation`](@extref MadNLP :std:doc:options) should be ignored for complementarity variables.
    """
    respect_comp_bounds::Bool = false # Essentially don't relax complementarity variables

    """
    Whether complementarity variables are initialized using a custom procedure instead of the standard bound-push.
    This places ``x_1`` and ``x_2`` on the line ``x_1 = x_2``.
    """
    center_complementarities::Bool = false

    raw"""
    At which fraction of the distance between ``(0,0)`` and ``(\sqrt{\tau},\sqrt{\tau}`` the centering initialization places the complementarity variables.

    Range: ``[0, 1.0]``
    """
    centering_factor::T = 0.5

    # regularization options
    """
    What KKT regularization to use near the solution.

    - `:vicente_wright`: adjusts variables away from their bounds to reduce the ill conditioning of the KKT matrix.
    - `:none`: applies no regularization.
    """
    kkt_regularization::Symbol = :vicente_wright # Options: :vicente_wright

    """
    How to regularize the indefinite contibutiion of the complementarity constraints to the Hessian of the Lagrangian.

    - `:none`: Does no regularization.
    - `:critical_rho`: Clips each Lagrange multiplier to the critical value maintaining a threshold positive definiteness.
    - `:eigenvalue_decomposition`: Project each 2 by 2 matrix to the space of Positive semi-definite matrices, by cliping their eigenvalues to [`min_eig_value`, `max_eig_value`].
    """
    q_regularization::Symbol = :none

    raw"""
    Threshold at which the `kkt_regularization` option is applied.

    Range: ``[0, \infty]``.
    """
    mu_thresh::T = 5e-6

    eta_factor::T = 0.1

    raw"""
    Minimum allowed eigenvalue for the eigenvalue decomposition based `q_regularization`.

    Range: ``(0, \infty]``
    """
    min_eig_value::T = 1e-4

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

    # Reset slacks
    reset_slacks_on_update::Bool = false

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Store Iterations
    iterates_fname::String = ""
end

@kwdef mutable struct RelaxationCounters
    counters::MadNLP.MadNLPCounters
    solver_time::Float64 = 0.0
end

# MadNLP-C algorithm
mutable struct RelaxationSolver{
    T,
    VT,
    MPCC<:AbstractMPCCModel{T,VT},
    RNLP<:AbstractMPCCRelaxation{T,VT},
    RELAX<:AbstractRelaxationUpdate{T},
    EG<:AbstractEndgameStrategy{T},
    SOLVER<:MadNLP.MadNLPSolver{T,VT},
}
    const mpcc::MPCC
    const rnlp::RNLP
    const ipm::SOLVER
    const logger::MadNLP.MadNLPLogger
    const iterate_logger::IterateLogger
    const opts::RelaxationOptions{T,RELAX,EG}
    const cnt::CCOptCounters
    inf_pr_cc::T
    const _cc1::VT
    const multipliers_cc1::VT
    const multipliers_cc2::VT
    const ind_cc1::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc2::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc1_lb::Vector{Int}
    const ind_cc2_lb::Vector{Int}
    const x::VT
    const b::Vector{Bool}
end

function RelaxationSolver(
    mpcc::AbstractMPCCModel{T,VT};
    solver_opts = RelaxationOptions(),
    ipm_options...,
) where {T,VT}
    start_time = time()
    rnlp = solver_opts.relaxation(mpcc)
    logger = MadNLP.MadNLPLogger(
        print_level = solver_opts.print_level,
        file_print_level = solver_opts.file_print_level,
        file = solver_opts.output_file == "" ? nothing :
               open(solver_opts.output_file, "w+"),
    )

    if haskey(ipm_options, :bound_relax_factor)
        ipm_options[:bound_relax_factor] == zero(T) || MadNLP.@warn(
            logger,
            "You have set a positive 'bound_relax_factor', this is probably a mistake! Allowing bound relaxation means complementarity variables are no longer strictly positive which can cause the algorithm to stall at non-physical, inaccurate solutions."
        )
        ipm = MadNLP.MadNLPSolver(rnlp; ipm_options...)
    else
        ipm = MadNLP.MadNLPSolver(rnlp; bound_relax_factor = zero(T), ipm_options...)
    end
    cnt = CCOptCounters(counters = ipm.cnt)
    initialize_relaxation(rnlp, ipm.opt.barrier.mu_init, solver_opts.delta_init)

    iterates_logger = IterateLogger(
        file = solver_opts.iterates_fname == "" ? nothing :
               open(solver_opts.iterates_fname, "w+"),
    )

    x = VT(undef, get_nvar(mpcc))
    _cc1 = VT(undef, get_ncc(mpcc))
    multipliers_cc1 = VT(undef, get_ncc(mpcc))
    multipliers_cc2 = VT(undef, get_ncc(mpcc))
    b = Vector{Bool}(undef, get_ncc(mpcc))

    # TODO(@anton) Can we do this nonquadratically
    ind_cc1 = copy(get_ind_cc1(mpcc))
    ind_cc2 = copy(get_ind_cc2(mpcc))
    _adjust_cc_inds!(ipm.cb, ind_cc1, ind_cc2)
    if solver_opts.kkt_regularization ∈ (:vicente_wright, :vicente_wright_sum)
        ind_cc1_lb = map((i)->findfirst((j)->i==j, ipm.kkt.ind_lb), ind_cc1)
        ind_cc2_lb = map((i)->findfirst((j)->i==j, ipm.kkt.ind_lb), ind_cc2)
    else
        ind_cc1_lb = Vector{Int}()
        ind_cc2_lb = Vector{Int}()
    end
    solver = RelaxationSolver(
        mpcc,
        rnlp,
        ipm,
        logger,
        iterates_logger,
        solver_opts,
        cnt,
        0.0,
        _cc1,
        multipliers_cc1,
        multipliers_cc2,
        ind_cc1,
        ind_cc2,
        ind_cc1_lb,
        ind_cc2_lb,
        x,
        b,
    )
    cnt.init_time = time() - start_time
    return solver
end

include("utils.jl")
include("kernels.jl")
include("barrier.jl")
include("relaxation_updates.jl")
include("inertia_correction.jl")
include("endgame.jl")
include("solver.jl")
