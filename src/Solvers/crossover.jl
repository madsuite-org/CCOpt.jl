const CROSSOVER_STATUSES = Dict(
    :b_stat => "B-Stationary point",
    :proj_failed => "Projection failed",
    :tr_min => "reached minimum trust region",
    :tiny_steps => "BNLP taking tiny steps",
    :max_iter => "maximum iteration",
)

@kwdef struct CrossoverOptions{T,LPCC_OPTS<:MadNLPCore.AbstractOptions} <:
              MadNLPCore.AbstractOptions
    lpcc_solver_opts::LPCC_OPTS = CCOpt.RelaxationOptions()
    lpcc_solver_kwargs::NamedTuple = NamedTuple()
    bnlp_solver_kwargs::NamedTuple = NamedTuple()
    proj_tr::T = 1.0
    tr0::T = 1e-4
    tr_min::T = 1e-6
    stationarity_tol::T = 1e-8
    tiny_step_tol::T = 1e-8
    s_stat_tol::T = 1e-7
    max_iter::Int = 10
    use_scaled_objective_stationarity::Bool = true
end

@kwdef mutable struct CrossoverCounters
    start_time::Float64 = 0.0
    lpcc_solves::Int = 0.0
    bnlp_solves::Int = 0.0
    lpcc_iter::Int = 0.0
    bnlp_iter::Int = 0.0
    bnlp_time::Float64 = 0.0
    lpcc_time::Float64 = 0.0
    proj_time::Float64 = 0.0
    construction_time::Float64 = 0.0
    total_time::Float64 = 0.0
end

mutable struct CrossoverExecutionStats{T,VT,LPCC_OPTS} <: AbstractExecutionStats
    lpcc_solver_opts::CrossoverOptions{T,LPCC_OPTS}
    status::Symbol
    objective::T
    solution::VT
    active_set::Vector{Bool}
    counters::CrossoverCounters
end

function SolverCore.getStatus(stats::CrossoverExecutionStats)
    return CROSSOVER_STATUSES[stats.status]
end

# options for LPCC solving with Gurobi
@kwdef struct GurobiLPCCSolverOptions{T} <: MadNLPCore.AbstractOptions
    M::T = 1000.0
end

# options for LPCC solving with HiGHS
@kwdef struct HiGHSLPCCSolverOptions{T} <: MadNLPCore.AbstractOptions
    M::T = 1000.0
end

function solve_lpcc(
    lpcc::LPCC,
    solver_opts::RelaxationOptions;
    bound_relax_factor = 0.0,
    kwargs...,
) where {LPCC<:LPCCModel}
    solver = CCOpt.RelaxationSolver(
        lpcc;
        solver_opts = solver_opts,
        bound_relax_factor = 0.0,
        kwargs...,
    )
    stats = CCOpt.solve_homotopy!(solver)

    y = Vector{Bool}(undef, get_ncc(lpcc))
    y .= comp_res_left(lpcc, stats.solution) .> comp_res_right(lpcc, stats.solution)
    return stats.status == MadNLPCore.SOLVE_SUCCEEDED, stats, y
end

macro add_to_timer(cnt_expr, action_expr)
    return quote
        local start_time = time()
        local out = $(esc(action_expr))
        $(esc(cnt_expr)) += time() - start_time
        out
    end
end

function crossover(
    mpcc::AbstractMPCCModel{T,VT},
    x0::VT,
    opts::CrossoverOptions{T},
) where {T,VT}
    cnt = CrossoverCounters()
    cnt.start_time = time()
    # Do projection
    proj_tr = opts.proj_tr
    proj_lpcc = @add_to_timer cnt.construction_time LPCCModel(mpcc, x0; tr = proj_tr)
    success, proj_sol, y0 = @add_to_timer cnt.lpcc_time solve_lpcc(
        proj_lpcc,
        opts.lpcc_solver_opts;
        opts.lpcc_solver_kwargs...,
    )
    cnt.lpcc_iter += 1

    if !success
        cnt.total_time = time() - cnt.start_time
        cnt.proj_time = cnt.total_time
        return CrossoverExecutionStats(opts, :proj_failed, obj(mpcc, x0), x0, y0, cnt)
    end

    bnlp = @add_to_timer cnt.construction_time BranchNLP(mpcc, y0)
    get_x0(bnlp) .= x0

    bnlp_sol = @add_to_timer cnt.bnlp_time MadNLPCore.madnlp(bnlp; opts.bnlp_solver_kwargs...)
    cnt.bnlp_iter += 1

    if bnlp_sol.status != MadNLPCore.SOLVE_SUCCEEDED
        cnt.total_time = time() - cnt.start_time
        cnt.proj_time = cnt.total_time
        return CrossoverExecutionStats(opts, :proj_failed, obj(mpcc, x0), x0, y0, cnt)
    end
    cnt.proj_time = time() - cnt.start_time
    x_curr = bnlp_sol.solution
    y_curr = y0
    y_next = similar(y0)
    f_curr = bnlp_sol.objective
    tr = opts.tr0
    ii = 1
    while ii <= opts.max_iter
        # Check for S-Stationarity
        if minimum(max.(comp_res_left(mpcc, x_curr), comp_res_right(mpcc, x_curr))) >=
           opts.s_stat_tol
            # bnlp solution is S-stationary up to tolerance and therefore B-Stationary
            cnt.total_time = time() - cnt.start_time
            return CrossoverExecutionStats(opts, :b_stat, f_curr, x_curr, y_curr, cnt)
        end
        I_00 = sum(
            max.(comp_res_left(mpcc, x_curr), comp_res_right(mpcc, x_curr)) .<=
            opts.s_stat_tol,
        )
        println("|I_00| = $(I_00)")
        while true
            lpcc = @add_to_timer cnt.construction_time LPCCModel(mpcc, x_curr; tr = tr)
            success, lpcc_sol, y_next = @add_to_timer cnt.lpcc_time solve_lpcc(
                lpcc,
                opts.lpcc_solver_opts;
                opts.lpcc_solver_kwargs...,
            )
            cnt.lpcc_iter += 1
            df =
                get_minimize(mpcc) ? f_curr - lpcc_sol.objective :
                lpcc_sol.objective - f_curr
            dy = sum(xor.(y_curr, y_next))
            println(
                "tr = $(tr), ||dy|| = $(dy) ||d|| = $(norm(lpcc_sol.solution)), df = $(df), df/||g|| = $(df/norm(grad(lpcc,x_curr)))",
            )
            if norm(lpcc_sol.solution) <= opts.stationarity_tol ||
               (
                opts.use_scaled_objective_stationarity ? df/norm(grad(lpcc, x_curr)) : df
            ) <= opts.stationarity_tol
                cnt.total_time = time() - cnt.start_time
                return CrossoverExecutionStats(opts, :b_stat, f_curr, x_curr, y_curr, cnt)
            end
            if any(y_next .!= y_curr)
                break
            else
                tr = 0.1*tr
                if tr <= opts.tr_min
                    cnt.total_time = time() - cnt.start_time
                    return CrossoverExecutionStats(
                        opts,
                        :tr_min,
                        f_curr,
                        x_curr,
                        y_curr,
                        cnt,
                    )
                end
            end
        end

        bnlp = @add_to_timer cnt.construction_time BranchNLP(mpcc, y_next)
        get_x0(bnlp) .= x_curr
        cnt.lpcc_iter += 1

        bnlp_sol = @add_to_timer cnt.bnlp_time MadNLPCore.madnlp(
            bnlp;
            bound_push = 1e-6,
            opts.bnlp_solver_kwargs...,
        )
        cnt.bnlp_iter += 1
        # if sol fails do something else)
        norm_delta_x = norm(x_curr .- bnlp_sol.solution)
        improvement =
            get_minimize(mpcc) ? f_curr - bnlp_sol.objective : bnlp_sol.objective - f_curr
        if improvement <= 0
            tr = 0.1*tr
            println("Δf = $(improvement), bnlp didn't improve, shrinking tr = $(tr)")
            if tr <= opts.tr_min
                cnt.total_time = time() - cnt.start_time
                return CrossoverExecutionStats(opts, :tr_min, f_curr, x_curr, y_curr, cnt)
            end
        else
            println("||Δx|| = $(norm_delta_x) Δf = $(improvement)")
            x_curr = bnlp_sol.solution
            f_curr = bnlp_sol.objective
            y_curr = y_next
            tr = opts.tr0
            ii += 1
        end
        if norm_delta_x <= opts.tiny_step_tol
            cnt.total_time = time() - cnt.start_time
            return CrossoverExecutionStats(opts, :tiny_step, f_curr, x_curr, y_curr, cnt)
        end
    end
    cnt.total_time = time() - cnt.start_time
    return CrossoverExecutionStats(opts, :max_iter, f_curr, x_curr, y_curr, cnt)
end
