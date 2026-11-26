# Each configuration is run on every problem of `TEST_PROBLEMS` (see `test/problems.jl`).
# `solver_opts` are passed to `PenaltyOptions`, `ipm_opts` to MadNLP.
const PENALTY_CONFIGS = [
    "With dynamic update" =>
        (solver_opts = (dynamic_rho_update = true,), ipm_opts = (;)),
    "Without dynamic update" =>
        (solver_opts = (dynamic_rho_update = false,), ipm_opts = (;)),
    "No Q regularization" =>
        (solver_opts = (q_regularization = :none,), ipm_opts = (;)),
    "Q regularization eigenvalue_decomposition" =>
        (solver_opts = (q_regularization = :eigenvalue_decomposition,), ipm_opts = (;)),
    "Large initial penalty" => (solver_opts = (rho_0 = 1e3,), ipm_opts = (;)),
    "Quality Function" =>
        (solver_opts = (;), ipm_opts = (barrier = MadNLP.QualityFunctionUpdate(),)),
]

# Known failures, as (configuration, problem) => checks expected to fail.
# Remove an entry once the underlying bug is fixed: the test then reports an unexpected pass.
const PENALTY_BROKEN = Dict{Tuple{String,String},Vector{Symbol}}(
    # ρ is increased whenever μ decreases while complementarity is not yet satisfied. The
    # free-mode quality-function barrier changes μ at almost every iteration, so ρ grows to
    # 1e9 and ν₁ = z_L - ρ x₂ loses accuracy to cancellation.
    ("Quality Function", "two branches") => [:stationarity],
)

@testset "Penalty Test" begin
    @testset "$config_name" for (config_name, config) in PENALTY_CONFIGS
        @testset "$(build(Float64).name)" for build in TEST_PROBLEMS
            prob = build(Float64)
            make_solver(mpcc) = CCOpt.PenaltySolver(
                mpcc;
                solver_opts = CCOpt.PenaltyOptions{Float64}(;
                    print_level = MadNLP.INFO,
                    config.solver_opts...,
                ),
                print_level = MadNLP.INFO,
                config.ipm_opts...,
            )
            test_solver(
                prob,
                make_solver;
                broken = get(PENALTY_BROKEN, (config_name, prob.name), Symbol[]),
            )
        end
    end
end
