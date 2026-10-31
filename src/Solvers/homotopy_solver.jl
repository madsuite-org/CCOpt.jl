######################### Types #########################
mutable struct HomotopySolverStats{T,VT}
    # TODO(@anton) what needs to live here
    # TODO(@anton) Should subclass AbstractExecutionStats probably
    status::MadNLPCore.Status           # Return status from the HomotopySolver
    solution::VT             # solution for primal variables x
    objective::T             # objective achieved
    multipliers::VT          # multipliers for nonlinear constraints (including relaxed complementarities)
    inf_cc::T                # Complementarity infeasibility calculated as max(x₁⊙x₂)

    wall_time::Float64          # Total wall time take by solver
    eval_function_time::Float64 # Total time spent in function eval (only for MadNLP)
    linear_solver_time::Float64 # Total time spent in linear solver (only for MadNLP)
    iter::Int                   # Total number of nlp solver iterations

    nlp_stats::Vector{Any}      # Vector of nlp solver stats.
end

function HomotopySolverStats(mpcc::AbstractMPCCModel{T,VT}) where {T,VT}
    return HomotopySolverStats(
        MadNLPCore.INITIAL,
        VT(undef, get_nvar(mpcc)),
        zero(T),
        VT(undef, get_ncon(mpcc)),
        zero(T),
        0.0,
        0.0,
        0.0,
        0,
        [],
    )
end

@kwdef mutable struct HomotopySolverOptions{T}
    σ₀::T = 1.0 # Initial value of complementarity relaxation
    𝛼::T = 0.1  # Linear component of relaxation tightening law
    𝛽::T = 1.0  # Superlinear component of relaxation tightening law

    comp_tol::T = 1e-8 # Target complementarity tolerence in terms of max(x₁⊙x₂)

    N_homotopy::Int = 10 # Max Number of outer iterations

    max_inner_iter::Int = 9000

    # TODO(@anton) Do we want an Int here? A symbol (e.g. `:info`, `:warn`, `:debug`, etc.)?
    print_level::MadNLPCore.LogLevels = MadNLPCore.INFO
    file_print_level::MadNLPCore.LogLevels = MadNLPCore.INFO
    output_file::String = ""

    # Decreased bound push for followup iterations
    warm_start_bound_push::Union{Float64,Nothing} = nothing

    nlp_solver_options::Dict{Symbol,Any} = Dict(
        :print_level=>0,
        :sb=>"yes",
        :mu_strategy=>"adaptive",
        :mu_oracle=>"quality-function",
        :bound_relax_factor=>0.0,
    )
end

"""
    HomotopySolver

Implementation of a Scholtes relaxation homotopy solver for MPCCs. It reformulates

  min f(x)
  s.t lc ≤ c(x) ≤ uc
      lx ≤ x₀   ≤ ux
      0  ≤ x₁ ⟂ x₂ ≥ 0

into the NLP
  min f(x)
  s.t lc ≤ c(x) ≤ uc
      lx ≤ x₀   ≤ ux
      0 ≤ x₁
      0 ≤ x₂
      0 ≥ x₁⊙x₂ - σ

And solve a sequence of these NLPs with σ→0.
"""
mutable struct HomotopySolver{M,S,T,VT} <: AbstractMPCCSolver{M,S,T,VT}
    mpcc::M
    nlp::ScholtesRelaxation{T,VT}
    solver::S

    opts::HomotopySolverOptions
    stats::HomotopySolverStats
    logger::MadNLPCore.MadNLPLogger

    start_time::Float64

    k::Int
    x_k::VT
    y_k::VT
    f_k::T
    inf_cc::T

    σ::T
end

function HomotopySolver(mpcc::AbstractMPCCModel, S::Type, opts::HomotopySolverOptions)
    nlp = ScholtesRelaxation(mpcc)

    if S<:MadNLPCore.AbstractMadNLPSolver
        solver = S(nlp; opts.nlp_solver_options...)
    else
        solver = S(nlp)
    end
    stats = HomotopySolverStats(mpcc)

    logger = MadNLPCore.MadNLPLogger(
        print_level = opts.print_level,
        file_print_level = opts.file_print_level,
        file = opts.output_file == "" ? nothing : open(opts.output_file, "w+"),
    )

    x_k = nlp.meta.x0
    y_k = nlp.meta.y0

    σ = opts.σ₀

    return HomotopySolver(
        mpcc,
        nlp,
        solver,
        opts,
        stats,
        logger,
        0.0,
        0,
        x_k,
        y_k,
        0.0,
        0.0,
        σ,
    )
end

######################### Helpers #########################
function solve_rnlp(
    solver::HomotopySolver{M,S,T,VT},
    n::Int,
) where {M,S<:SolverCore.AbstractOptimizationSolver,T,VT}
    # TODO(@anton) copying here seems expensiveish, perhaps store a "working settings" copy.
    nlp_opts_i = copy(solver.opts.nlp_solver_options)

    # Preprocess nlp_opts
    if n > 1
        nlp_opts_i[:warm_start_init_point] = "yes"
        nlp_opts_i[:warm_start_bound_push] = something(
            solver.opts.warm_start_bound_push,
            get(nlp_opts_i, :warm_start_bound_push, nothing),
            0.001,
        )
    end
    max_iter = min(
        something(get(nlp_opts_i, :max_iter, nothing), typemax(Int)),
        solver.opts.max_inner_iter-solver.stats.iter,
    )
    nlp_opts_i[:max_iter] = max_iter

    return SolverCore.solve!(solver.solver, solver.nlp; nlp_opts_i...)
end

function solve_rnlp(
    solver::HomotopySolver{M,S,T,VT},
    n::Int,
) where {M,S<:MadNLPCore.AbstractMadNLPSolver,T,VT}
    # TODO(@anton) copying here seems expensiveish, perhaps store a "working settings" copy.
    nlp_opts_i = copy(solver.opts.nlp_solver_options)

    # Preprocess nlp_opts
    if n > 1
        nlp_opts_i[:bound_push] = something(
            solver.opts.warm_start_bound_push,
            get(nlp_opts_i, :bound_push, nothing),
            1e-2,
        )
    end
    max_iter = min(
        something(get(nlp_opts_i, :max_iter, nothing), typemax(Int)),
        solver.opts.max_inner_iter-solver.stats.iter,
    )
    nlp_opts_i[:max_iter] = max_iter
    return MadNLPCore.solve!(solver.solver; nlp_opts_i...)
end

function set_silent!(
    solver::HomotopySolver{M,S,T,VT},
) where {M,S<:SolverCore.AbstractOptimizationSolver,T,VT}
    return solver.opts.nlp_solver_options[:print_level] = 0
end

function set_silent!(
    solver::HomotopySolver{M,S,T,VT},
) where {M,S<:MadNLPCore.AbstractMadNLPSolver,T,VT}
    return solver.opts.nlp_solver_options[:print_level] = MadNLPCore.ERROR
end

function nlp_solve_acceptable(nlp_stats::AbstractExecutionStats)
    return nlp_stats.status ∈ [:first_order, :acceptable, :small_step]
end

function nlp_solve_acceptable(nlp_stats::MadNLPCore.MadNLPExecutionStats)
    return nlp_stats.status ∈ [
        MadNLPCore.SOLVE_SUCCEEDED,
        MadNLPCore.SOLVED_TO_ACCEPTABLE_LEVEL,
        MadNLPCore.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
    ]
end

# TODO(@anton) I think we _really_ should look into standardizing on Solver core
#              in MadNLP but this would require buyin from more of the MadNLP dev team
const SOLVER_CORE_TO_MPCC_SOLVER_STATUS = Dict(
    :exception => MadNLPCore.INTERNAL_ERROR,
    :first_order => MadNLPCore.SOLVE_SUCCEEDED,
    :acceptable => MadNLPCore.SOLVED_TO_ACCEPTABLE_LEVEL,
    :infeasible => MadNLPCore.INFEASIBLE_PROBLEM_DETECTED,
    :max_eval => MadNLPCore.MAXIMUM_ITERATIONS_EXCEEDED,
    :max_iter => MadNLPCore.MAXIMUM_ITERATIONS_EXCEEDED,
    :max_time => MadNLPCore.MAXIMUM_WALLTIME_EXCEEDED,
    :neg_pred => MadNLPCore.INTERNAL_ERROR,
    :not_desc => MadNLPCore.INTERNAL_ERROR,
    :small_residual => MadNLPCore.INTERNAL_ERROR,
    :small_step => MadNLPCore.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
    :stalled => MadNLPCore.INTERNAL_ERROR,
    :unbounded => MadNLPCore.INTERNAL_ERROR,
    :unknown => MadNLPCore.INTERNAL_ERROR,
    :user => MadNLPCore.USER_REQUESTED_STOP,
)

function convert_nlp_solve(nlp_stats::AbstractExecutionStats)
    return get(SOLVER_CORE_TO_MPCC_SOLVER_STATUS, nlp_stats.status, MadNLPCore.INTERNAL_ERROR)
end

function convert_nlp_solve(nlp_stats::MadNLPCore.MadNLPExecutionStats)
    return nlp_stats.status
end

function reset_nlp_solver!(
    solver::HomotopySolver{M,S,T,VT},
) where {M,S<:SolverCore.AbstractOptimizationSolver,T,VT}
    # Do nothing
end

function reset_nlp_solver!(
    solver::HomotopySolver{M,S,T,VT},
) where {M,S<:MadNLPCore.AbstractMadNLPSolver,T,VT}
    solver.solver.cnt.k = 0
    solver.solver.cnt.l = 0
    solver.solver.cnt.t = 0
    solver.solver.cnt.acceptable_cnt = 0
    solver.solver.cnt.unsuccessful_iterate = 0
    return solver.solver.cnt.restoration_fail_count = 0
end

function update_times!(
    solver::HomotopySolver{M,S,T,VT},
    nlp_stats::AbstractExecutionStats,
) where {M,S<:SolverCore.AbstractOptimizationSolver,T,VT}
    # TODO(@anton) ipopt interface doesn't give detailed times
    return solver.stats.iter += nlp_stats.iter
end

function update_times!(
    solver::HomotopySolver{M,S,T,VT},
    nlp_stats::MadNLPCore.MadNLPExecutionStats,
) where {M,S<:MadNLPCore.AbstractMadNLPSolver,T,VT}
    solver.stats.eval_function_time = nlp_stats.counters.eval_function_time
    solver.stats.linear_solver_time = nlp_stats.counters.linear_solver_time
    return solver.stats.iter += nlp_stats.iter
end

function print_headers(solver::HomotopySolver{M,S,T,VT}) where {M,S,T,VT}
    # TODO(@anton) also log nlpsolver output
    header = SolverCore.log_header(
        [:iter, :solver_status, :objective, :inf_cc, :nlp_iters],
        [Int, Symbol, T, T, Int],
    )
    MadNLPCore.@info(solver.logger, "CCOpt Homotopy Solver")
    MadNLPCore.@info(solver.logger, header)
end

function print_outer_iter(solver::HomotopySolver)
    row = SolverCore.log_row(
        Any[
            solver.k,
            Symbol(solver.stats.nlp_stats[end].status),
            solver.f_k,
            solver.inf_cc,
            solver.stats.nlp_stats[end].iter,
        ],
    )
    MadNLPCore.@info(solver.logger, row)
end

######################### Main loop #########################
function solve!(
    solver::HomotopySolver{M,T,VT};
    x = nothing,
    y = nothing,
    kwargs...,
) where {M,T,VT}
    return solve!(solver.mpcc, solver, solver.stats; x = x, y = y, kwargs...)
end

function solve!(
    mpcc::AbstractMPCCModel,
    solver::HomotopySolver,
    stats::HomotopySolverStats;
    x = nothing,
    y = nothing,
    kwargs...,
)
    solver.start_time = time()

    if x ≠ nothing
        solver.x_k .= x
    end
    if y ≠ nothing
        solver.y_k .= y
    end

    opts = solver.opts

    # TODO(@anton) Long term we may want a pre-process step for the options
    if opts.print_level == MadNLPCore.ERROR
        set_silent!(solver) # TODO(@anton) This is one of the options that madNLP doesn't respect on resolve :(
    end

    print_headers(solver)
    converged = false
    timeout = false
    max_inner_iter_reached = false
    set_relaxation(solver.nlp, solver.σ)
    ii = 1
    while ii ≤ opts.N_homotopy
        solver.k = ii
        nlp_stats = solve_rnlp(solver, ii)
        push!(stats.nlp_stats, nlp_stats)
        solver.inf_cc = comp_residual_product(mpcc, nlp_stats.solution)
        solver.x_k = nlp_stats.solution
        solver.y_k = nlp_stats.multipliers
        solver.f_k = nlp_stats.objective
        print_outer_iter(solver)

        copyto!(NLPModels.get_x0(solver.nlp), solver.x_k)
        copyto!(NLPModels.get_y0(solver.nlp), solver.y_k)
        update_times!(solver, nlp_stats)

        if nlp_solve_acceptable(nlp_stats) && solver.inf_cc ≤ solver.opts.comp_tol
            converged = true
            break
        end

        if stats.iter == solver.opts.max_inner_iter
            max_inner_iter_reached = true
            break
        end

        # 𝛽 > 1 decreases σ superlinearly when close to convergence
        solver.σ = min(opts.𝛼*solver.σ, solver.σ^opts.𝛽)
        set_relaxation(solver.nlp, solver.σ)
        ii += 1
        reset_nlp_solver!(solver)
    end
    if converged
        stats.status = convert_nlp_solve(stats.nlp_stats[end])
    elseif ii > opts.N_homotopy
        stats.status = MadNLPCore.MAXIMUM_ITERATIONS_EXCEEDED
    end
    stats.solution = solver.x_k
    stats.multipliers = solver.y_k[1:get_ncon(mpcc)] # Unreliable
    stats.inf_cc = solver.inf_cc
    stats.objective = solver.f_k
    stats.wall_time = time() - solver.start_time;
    return stats
end
