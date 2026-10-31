@kwdef mutable struct CCOptCounters
    counters::MadNLPCore.MadNLPCounters
    start_time::Float64 = 0.0
    init_time::Float64 = 0.0
    solve_time::Float64 = 0.0
    solver_time::Float64 = 0.0
    total_time::Float64 = 0.0
end

mutable struct CCOptExecutionStats{T,VT} <: AbstractExecutionStats
    options::MadNLPCore.AbstractOptions
    mpcc_options::MadNLPCore.AbstractOptions
    status::MadNLPCore.Status
    objective::T
    solution::VT
    constraints::VT
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    multipliers_x1::VT
    multipliers_x2::VT
    dual_feas::T
    primal_feas::T
    inf_pr_cc::T
    iter::Int
    counters::CCOptCounters
end
