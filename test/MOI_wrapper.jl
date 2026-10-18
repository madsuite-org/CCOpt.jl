module TestMOI

using Test
import MathOptInterface as MOI
using NLPModelsJuMP
using MathOptComplements
using CCOpt

# Add bridges defined in MathOptComplements
const _ALL_BRIDGE_TYPES = Any[
    MathOptComplements.Bridges.SpecifySetTypeBridge,
    MathOptComplements.Bridges.ComplementsVectorizeBridge,
    MathOptComplements.Bridges.SplitIntervalBridge,
    MathOptComplements.Bridges.FlipSignBridge,
    MathOptComplements.Bridges.ToZerosBridge,
    MathOptComplements.Bridges.VerticalBridge,
]

function _add_all_bridges(model::MOI.ModelLike, ::Type{T} = Float64) where {T}
    for bridge_type in _ALL_BRIDGE_TYPES
        MOI.Bridges.add_bridge(model, bridge_type{T})
    end
    return
end

function test_runtests()
    model = MOI.instantiate(CCOpt.Optimizer, with_bridge_type = Float64)
    _add_all_bridges(model)
    MOI.set(model, MOI.Silent(), true) # comment this to enable output
    config = MOI.Test.Config(
        atol = 1e-4,
        optimal_status = MOI.LOCALLY_SOLVED,
        infeasible_status = MOI.LOCALLY_INFEASIBLE,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
            # TODO dual not done yet
            MOI.DualObjectiveValue,
        ],
    )
    MOI.Test.runtests(
        model,
        config,
        exclude = [
            "test_basic_VectorAffineFunction_Complements", # CC RHS is mispecified
            "test_basic_VectorQuadraticFunction_Complements", # CC RHS is mispecified
            r"test_linear_DUAL_INFEASIBLE*", # diverging iterate
            "test_linear_integration_delete_variables",
            "test_linear_VectorAffineFunction_empty_row",
            "test_model_copy_to_UnsupportedAttribute",
            "test_nonlinear_expression_hs109",
            "test_nonlinear_duals", #NLPModelsJuMP issue
            "test_nonlinear_hs071_no_hessian", # require support for LBFGS in MOI interface
            "test_nonlinear_hs071_hessian_vector_product", # not supported
            "test_nonlinear_mixed_complementarity", # issue with MathOptComplements
            "test_nonlinear_qp_complementarity_constraint", # issue with MathOptComplements
            "test_nonlinear_expression_multivariate_function", # legacy issue
            "test_nonlinear_invalid", # require support for ExprGraph
            "test_solve_TerminationStatus_DUAL_INFEASIBLE", # diverging iterate
            r"test_solve_VariableIndex_ConstraintDual*", # symbolic exception
            r"test_conic*",
        ],
    )
    return
end

function test_raw_status_string()
    # 0 <= x1 _|_ x2 >= 0, minimising the distance to (1, 1): optima at (1, 0) and (0, 1).
    inner = MOI.instantiate(CCOpt.Optimizer, with_bridge_type = Float64)
    _add_all_bridges(inner)
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        inner,
    )
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    for xi in x
        MOI.add_constraint(model, xi, MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Complements(2))
    f = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, xi, xi) for xi in x],
        [MOI.ScalarAffineTerm(-2.0, xi) for xi in x],
        2.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) ∈
          [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
    status = MOI.get(model, MOI.RawStatusString())
    @test status isa String
    @test startswith(status, "Optimal Solution Found") ||
          startswith(status, "Solved To Acceptable Level")
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

end  # module

TestMOI.runtests()
