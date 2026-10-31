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
  Proportional Relaxation update which updates ``σ = aμ^b``
"""
@kwdef struct ProportionalRelaxationUpdate{T} <: AbstractFixedRelaxationUpdate{T}
    sigma_mu_ratio::T = 1.0 # a
    sigma_mu_exp::T = 1.0 # b
    monotone::Bool = false
    sigma_min::T = 1e-8
end

"""
  Rolloff Relaxation update which updates ``σ = c*μ^a/(sqrt(μ^a^2)+b)``
"""
@kwdef struct RolloffRelaxationUpdate{T} <: AbstractFixedRelaxationUpdate{T}
    rolloff_slope::T = 2.0 # a
    rolloff_point::T = 1e-6 # b
    rolloff_max::T = 1.0 # c
    monotone::Bool = false
    sigma_min::T = 1e-8
end

"""
  LOQO based algorithm for adaptively selecting the Scholtes relaxation parameter σ.

  ``ξ = (ncc*min(x₁,x₂))/(x₁ᵀx₂)``
  ``k = γ*min((1-r)(1-ξ)/ξ, 2)³``
  ``σ = k(x₁ᵀx₂)/ncc``
"""
@kwdef struct LOQORelaxationUpdate{T} <: AbstractAdaptiveRelaxationUpdate{T}
    gamma::T = 0.05 # scale factor
    gamma_min::T = 1e-5 # smallest factor of reduction allowed
    mu_factor::T = 1e-5 # smallest factor of reduction allowed
    r::T = 0.95 # Steplength param
    sigma_min::T = 1e-8
end

"""
  Two-sided Scholtes relaxation attempts to drive either the scholtes bound x1*x2 - σ ≤ 0 to zero
  or the lower bounds of x1 and x2 choosing which to decrease by inspecting the estimated mpcc
  lagrange multipliers.
"""
@kwdef struct TwoSidedScholtesUpdate{T} <: AbstractAdaptiveRelaxationUpdate{T}
    kappa::T = 0.1
    k_ftb::T = 0.9
    tau::T = 0.3
    sigma_min::T = 1e-8
end

"""
  A scheduled update for σ, that is the Fiacco-McCormick rule is applied when the RNLP(σ) is solved to a target accuracy.
  The target accuracy is calculated as a piecewise function: base accuracy if σ > accuracy shoulder, and
  ((target accuaracy - base accuracy)/(target accuracy - accuracy shoulder))*σ otherwise.
  This is done independently of the μ (barrier parameter) update.
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
"""
@kwdef mutable struct RelaxLBEndgameStrategy{T} <: AbstractEndgameStrategy{T}
    delta_max::T = 1e-4
    min_delta_inc_factor::T = 1.1
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

    p::MadNLPCore.UnreducedKKTVector{T,VT}

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
    nu1_filt::VT
    nu2_filt::VT
    delta1::VT
    delta2::VT

    n_fact::Int

    KKT_s::VT

    magic::Bool
end

# Options struct
@kwdef struct RelaxationOptions{
    T,
    RELAX<:AbstractRelaxationUpdate{T},
    EG<:AbstractEndgameStrategy{T},
} <: MadNLPCore.AbstractOptions
    # Relaxation type
    relaxation::Type = ScholtesRelaxation

    # adaptive mu update parameters
    use_specialized_barrier_update::Bool = false

    # complementarity homotopy options
    relaxation_update::RELAX = ProportionalRelaxationUpdate()
    sigma_min::T = 1e-8 # TODO(@anton) I think this should be probably be related to ipm tolerance
    delta_init::T = 0.0

    # Endgame options
    endgame_threshold::T = 1e-6
    endgame_strategy::EG = NoEndgameStrategy{Float64}() # TODO(@anton) this will break for other precisions

    # initialization options
    respect_comp_bounds::Bool = false # Essentially don't relax complementarity variables
    center_complementarities::Bool = false
    centering_factor::T = 0.5

    # regularization options
    kkt_regularization::Symbol = :vicente_wright # Options: :vicente_wright
    q_regularization::Symbol = :none
    mu_thresh::T = 5e-6
    eta_factor::T = 0.1
    min_eig_value::T = 1e-4
    max_eig_value::T = Inf
    critical_rho_factor::T = 0.99
    min_reg_mu::T = 5e-6

    # Reset slacks
    reset_slacks_on_update::Bool = false

    # Output options
    output_file::String = ""
    print_level::MadNLPCore.LogLevels = MadNLPCore.INFO
    file_print_level::MadNLPCore.LogLevels = MadNLPCore.INFO

    # Store Iterations
    iterates_fname::String = ""
end

@kwdef mutable struct RelaxationCounters
    counters::MadNLPCore.MadNLPCounters
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
    SOLVER<:MadNLPCore.MadNLPSolver{T,VT},
}
    const mpcc::MPCC
    const rnlp::RNLP
    const ipm::SOLVER
    const logger::MadNLPCore.MadNLPLogger
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
    logger = MadNLPCore.MadNLPLogger(
        print_level = solver_opts.print_level,
        file_print_level = solver_opts.file_print_level,
        file = solver_opts.output_file == "" ? nothing :
               open(solver_opts.output_file, "w+"),
    )

    if haskey(ipm_options, :bound_relax_factor)
        ipm_options[:bound_relax_factor] == zero(T) || MadNLPCore.@warn(
            logger,
            "You have set a positive 'bound_relax_factor', this is probably a mistake! Allowing bound relaxation means complementarity variables are no longer strictly positive which can cause the algorithm to stall at non-physical, inaccurate solutions."
        )
        ipm = MadNLPCore.MadNLPSolver(rnlp; ipm_options...)
    else
        ipm = MadNLPCore.MadNLPSolver(rnlp; bound_relax_factor = zero(T), ipm_options...)
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
