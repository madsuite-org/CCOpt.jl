# Cases that are known not to work yet. Each check is recorded as `@test_broken`, so fixing
# the underlying issue shows up as an unexpected pass.
@testset "Known limitations" begin
    # MadNLP removes fixed variables, and `_adjust_cc_inds!` does not handle the case where
    # a complementarity variable itself is fixed. The relaxation solver reaches the solution
    # but never satisfies its termination test; the penalty solver reports infeasibility
    # with NaN multipliers.
    @testset "Fixed complementarity variable" begin
        @testset "$solver_name" for (solver_name, make_solver) in RESOLVE_SOLVERS
            prob = fixed_complementarity_variable(Float64)
            stats = CCOpt.solve_homotopy!(make_solver(prob.mpcc))
            @test_broken stats.status ∈
                         (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
            @test any(x -> isapprox(stats.solution, x; atol = 1e-5), prob.x_star)
        end
    end
end
