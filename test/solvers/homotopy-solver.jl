function make_homotopy_solver(mpcc; kwargs...)
    opts = CCOpt.HomotopySolverOptions(; print_level = MadNLP.ERROR, kwargs...)
    opts.nlp_solver_options =
        Dict(:bound_relax_factor => 0.0, :print_level => MadNLP.ERROR, :max_iter => 500)
    return CCOpt.HomotopySolver(mpcc, MadNLP.MadNLPSolver, opts)
end

@testset "Homotopy Solver Tests" begin
    # `HomotopySolverStats` does not return bound multipliers.
    @testset "MadNLP: $(build(Float64).name)" for build in TEST_PROBLEMS
        prob = build(Float64)
        stats = CCOpt.solve!(make_homotopy_solver(prob.mpcc; comp_tol = 1e-7))
        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.inf_cc ≤ 1e-7
        @test any(x -> isapprox(stats.solution, x; atol = 1e-5), prob.x_star)
        @test stats.objective ≈ prob.f_star atol = 1e-5
    end

    @testset "Exits" begin
        @testset "Maximum number of homotopy steps" begin
            # The two-branch problem needs ~8 outer iterations to reach comp_tol.
            prob = two_branches(Float64)
            stats = CCOpt.solve!(make_homotopy_solver(prob.mpcc; N_homotopy = 2))
            @test stats.status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
            @test length(stats.nlp_stats) == 2
        end
        @testset "Maximum number of inner iterations" begin
            prob = two_branches(Float64)
            stats = CCOpt.solve!(make_homotopy_solver(prob.mpcc; max_inner_iter = 5))
            @test stats.status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
            @test stats.iter == 5
        end
    end
end
