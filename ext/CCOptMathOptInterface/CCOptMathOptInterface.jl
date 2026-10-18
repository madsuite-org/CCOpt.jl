################################################################################
# This code is adapted from NLPModelsJuMP.
#
# Original source-code:
# https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/blob/main/src/MOI_wrapper.jl
#
# Copyright (c) 2018-2019: Abel Soares Siqueira and Dominique Orban
# NLPModelsJuMP.jl is licensed under the [MPL version 2.0](https://www.mozilla.org/MPL/2.0/).
################################################################################
module CCOptMathOptInterface

using MathOptComplements
using CCOpt
using MadNLP
using NLPModels
using MPCCModels
using NLPModelsJuMP

const MOI = MathOptComplements.MOI

function __init__()
    setglobal!(CCOpt, :Optimizer, Optimizer)
    return
end

# N.B.: required to have ModelFilter working with MOI.Utilities.get_bounds
function MOI.is_valid(model::MOI.Utilities.ModelFilter, index::MOI.Index)
    return MOI.is_valid(model.inner, index)::Bool
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _CC_SETS = MathOptComplements.ComplementsWithSetType{MOI.Nonnegatives}

mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Dict{String, Any}
    solver
    nlp::Union{Nothing, NLPModels.AbstractNLPModel}
    mpcc::Union{Nothing, MPCCModel}
    stats::Union{
        Nothing,
        CCOpt.CCOptExecutionStats{Float64, Vector{Float64}},
        MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}},
    }
    silent::Bool
    function Optimizer()
        return new(Dict{String, Any}(), nothing, nothing, nothing, nothing, false)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "CCOpt"

function MOI.is_empty(optimizer::Optimizer)
    return isnothing(optimizer.solver) && isnothing(optimizer.mpcc)
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.mpcc = nothing
    optimizer.nlp = nothing
    optimizer.stats = nothing
    return
end

###
### MOI.RawOptimizerAttribute
###

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return optimizer.options[param.name] = value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[param.name]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{
            <:Union{
                MOI.VariableIndex,
                MOI.ScalarAffineFunction{Float64},
                MOI.ScalarQuadraticFunction{Float64},
                MOI.ScalarNonlinearFunction,
            },
        },
        MOI.NLPBlock,
        MOI.UserDefinedFunction,
    },
)
    return true
end

###
### MOI.AbstractVariableAttribute
###

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

###
### `supports_constraint`
###

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
            MOI.ScalarNonlinearFunction,
        },
    },
    ::Type{<:_SETS},
)
    return true
end

# Support for complementarity constraint
# The problem should be reformulated before using the MathOptComplements's bridges
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{_CC_SETS},
)
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    options =
        Dict{Symbol, Any}(Symbol(key) => dest.options[key] for key in keys(dest.options))
    # Parse options for homotopy
    relax_options = Dict{Symbol, Any}()
    for key in keys(options)
        if hasfield(CCOpt.RelaxationOptions, key)
            relax_options[key] = options[key]
            delete!(options, key)
        end
    end
    # Check if silent is activated
    if dest.silent
        options[:print_level] = MadNLP.ERROR
        relax_options[:print_level] = MadNLP.ERROR
    end

    filtered_src = MOI.Utilities.ModelFilter(src) do item
        return item != (MOI.VectorOfVariables, _CC_SETS)
    end
    # Use filtered_src to build NLPModels
    nlp, index_map = NLPModelsJuMP.nlp_model(filtered_src)
    dest.nlp = nlp

    cc_cons = MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables, _CC_SETS}())
    # If no complementarity constraint is detected, we fallback to MadNLP
    if isempty(cc_cons)
        @warn(
            "The model does not have any complementarity constraints. Switching to MadNLP."
        )
        dest.mpcc = MPCCModel(nlp, Int[], Int[])
        dest.solver = MadNLP.MadNLPSolver(nlp; options...)
        return index_map
    end

    ind_cc1, ind_cc2 = MOI.VariableIndex[], MOI.VariableIndex[]
    for cidx in cc_cons
        fun = MOI.get(src, MOI.ConstraintFunction(), cidx)
        set = MOI.get(src, MOI.ConstraintSet(), cidx)
        n_comp = div(set.dimension, 2)
        for cc in 1:n_comp
            push!(ind_cc1, fun.variables[cc])
            push!(ind_cc2, fun.variables[cc+n_comp])
        end
    end

    ind_x1 = getfield.(ind_cc1, :value)
    ind_x2 = getfield.(ind_cc2, :value)

    dest.mpcc = MPCCModel(nlp, ind_x1, ind_x2)
    dest.solver = CCOpt.RelaxationSolver(
        dest.mpcc;
        solver_opts=CCOpt.RelaxationOptions(; relax_options...),
        options...,
    )

    return index_map
end

function MOI.optimize!(model::Optimizer)
    model.stats = if isa(model.solver, MadNLP.MadNLPSolver)
        MadNLP.solve!(model.solver)
    else
        CCOpt.solve_homotopy!(model.solver)
    end
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    cnt = optimizer.stats.counters
    return cnt.total_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    if optimizer.stats === nothing
        return "Optimize not called"
    end
    # Both CCOptExecutionStats and MadNLPExecutionStats carry MadNLP's options.
    return MadNLP.get_status_output(optimizer.stats.status, optimizer.stats.options)
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.stats, attr.name)
end

const TERMINATION_STATUS = Dict{MadNLP.Status, MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    MadNLP.INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN => MOI.INVALID_MODEL,
)

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.stats)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return TERMINATION_STATUS[optimizer.stats.status]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.objective
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.ALMOST_LOCALLY_SOLVED
        return MOI.NEARLY_FEASIBLE_POINT
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(optimizer, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = MOI.get(optimizer, MOI.TerminationStatus())
    if status == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    elseif status == MOI.ALMOST_LOCALLY_SOLVED
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.solution[vi.value]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1

### MOI.ConstraintPrimal
#
function row(
    optimizer::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}},
)
    return ci.value + 1
end

function row(
    optimizer::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{Float64}},
)
    nlp = optimizer.nlp
    offset = nlp.meta.nlin
    return ci.value + 1
end

function row(optimizer::Optimizer, ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction})
    nlp = optimizer.nlp
    n_linquad = nlp.quadcon.nquad
    offset = nlp.meta.nlin + nlp.quadcon.nquad
    return ci.value + offset + 1
end

### MOI.ConstraintDual

function _dual_multiplier(optimizer::Optimizer)
    nlp = optimizer.nlp
    return NLPModels.get_minimize(nlp) ? 1.0 : -1.0
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F, <:_SETS},
) where {
    F <: Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    },
}
    MOI.check_result_index_bounds(optimizer, attr)
    s = -_dual_multiplier(optimizer)
    return s * optimizer.stats.multipliers[row(optimizer, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex, S},
) where {
    S <: Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    },
}
    MOI.check_result_index_bounds(model, attr)
    if S <: MOI.GreaterThan{Float64}
        return model.stats.multipliers_L[ci.value]
    elseif S <: MOI.LessThan{Float64}
        return -model.stats.multipliers_U[ci.value]
    else
        return model.stats.multipliers_L[ci.value] - model.stats.multipliers_U[ci.value]
    end
end

end
