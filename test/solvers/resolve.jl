# Solving twice with the same solver object goes through `MadNLP.reinitialize!` instead of
# `initialize!`; warm starts additionally overwrite the primal and dual starting point.
const RESOLVE_SOLVERS = [
    "RelaxationSolver" =>
        mpcc -> CCOpt.RelaxationSolver(
            mpcc;
            solver_opts = CCOpt.RelaxationOptions(; print_level = MadNLP.ERROR),
            print_level = MadNLP.ERROR,
        ),
    "PenaltySolver" =>
        mpcc -> CCOpt.PenaltySolver(
            mpcc;
            solver_opts = CCOpt.PenaltyOptions{Float64}(; print_level = MadNLP.ERROR),
            print_level = MadNLP.ERROR,
        ),
]

@testset "Re-solve and warm start" begin
    @testset "$solver_name" for (solver_name, make_solver) in RESOLVE_SOLVERS
        @testset "$(build(Float64).name)" for build in
                                              [two_branches, multi_pair_constrained]
            prob = build(Float64)
            solver = make_solver(prob.mpcc)
            stats = CCOpt.solve_homotopy!(solver)
            test_solution(prob, stats)
            x, y = copy(stats.solution), copy(stats.multipliers)

            @testset "Re-solve" begin
                test_solution(prob, CCOpt.solve_homotopy!(solver))
            end

            @testset "Warm start from the solution" begin
                stats2 = CCOpt.solve_homotopy!(make_solver(prob.mpcc); x = x)
                test_solution(prob, stats2)
            end
        end
    end
end
