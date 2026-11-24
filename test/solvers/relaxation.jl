# Each configuration is run on every problem of `TEST_PROBLEMS` (see `test/problems.jl`).
# `solver_opts` are passed to `RelaxationOptions`, `ipm_opts` to MadNLP.
const RELAXATION_CONFIGS = [
    "Default Options" => (solver_opts = (;), ipm_opts = (;)),
    "Quality Function" => (
        solver_opts = (;),
        ipm_opts = (barrier = MadNLP.QualityFunctionUpdate(),),
    ),
    "Quality Function specialization" => (
        solver_opts = (use_specialized_barrier_update = true,),
        ipm_opts = (barrier = MadNLP.QualityFunctionUpdate(),),
    ),
    "Rolloff relaxation" => (
        solver_opts = (relaxation_update = CCOpt.RolloffRelaxationUpdate(),),
        ipm_opts = (;),
    ),
    "Endgame relaxation algorithm" => (
        solver_opts = (endgame_strategy = CCOpt.RelaxLBEndgameStrategy(),),
        ipm_opts = (;),
    ),
    "Q regularization critical_rho" => (
        solver_opts = (q_regularization = :critical_rho,),
        ipm_opts = (;),
    ),
    "Q regularization eigenvalue_decomposition" => (
        solver_opts = (q_regularization = :eigenvalue_decomposition,),
        ipm_opts = (;),
    ),
    # Relaxation updates
    "Proportional monotone relaxation" => (
        solver_opts = (relaxation_update = CCOpt.ProportionalRelaxationUpdate(monotone = true),),
        ipm_opts = (;),
    ),
    "Rolloff monotone relaxation" => (
        solver_opts = (relaxation_update = CCOpt.RolloffRelaxationUpdate(monotone = true),),
        ipm_opts = (;),
    ),
    "LOQO relaxation" => (
        solver_opts = (relaxation_update = CCOpt.LOQORelaxationUpdate(),),
        ipm_opts = (;),
    ),
    "Two-sided Scholtes relaxation" => (
        solver_opts = (relaxation_update = CCOpt.TwoSidedScholtesUpdate(),),
        ipm_opts = (;),
    ),
    "Scheduled relaxation" => (
        solver_opts = (relaxation_update = CCOpt.ScheduledRelaxationUpdate(),),
        ipm_opts = (;),
    ),
    # Barrier updates
    "Monotone barrier" => (solver_opts = (;), ipm_opts = (barrier = MadNLP.MonotoneUpdate(),)),
    "LOQO barrier" => (solver_opts = (;), ipm_opts = (barrier = MadNLP.LOQOUpdate(),)),
    "LOQO barrier specialization" => (
        solver_opts = (use_specialized_barrier_update = true,),
        ipm_opts = (barrier = MadNLP.LOQOUpdate(),),
    ),
    # KKT regularization
    "Vicente-Wright sum regularization" => (
        solver_opts = (kkt_regularization = :vicente_wright_sum,),
        ipm_opts = (;),
    ),
    "No KKT regularization" => (solver_opts = (kkt_regularization = :none,), ipm_opts = (;)),
    "Scaled KKT system" => (
        solver_opts = (;),
        ipm_opts = (kkt_system = MadNLP.ScaledSparseKKTSystem,),
    ),
    # Initialization
    "Centered complementarities" => (
        solver_opts = (center_complementarities = true,),
        ipm_opts = (;),
    ),
    "Respect complementarity bounds" => (
        solver_opts = (respect_comp_bounds = true,),
        ipm_opts = (;),
    ),
    "Reset slacks on update" => (
        solver_opts = (reset_slacks_on_update = true,),
        ipm_opts = (;),
    ),
    "Initial lower bound relaxation" => (solver_opts = (delta_init = 1e-4,), ipm_opts = (;)),
]

# Known failures, as (configuration, problem) => checks expected to fail.
# Remove an entry once the underlying bug is fixed: the test then reports an unexpected pass.
const RELAXATION_BROKEN = Dict{Tuple{String,String},Vector{Symbol}}()

@testset "RelaxationSolver Test" begin
    @testset "$config_name" for (config_name, config) in RELAXATION_CONFIGS
        @testset "$(build(Float64).name)" for build in TEST_PROBLEMS
            prob = build(Float64)
            make_solver(mpcc) = CCOpt.RelaxationSolver(
                mpcc;
                solver_opts = CCOpt.RelaxationOptions(;
                    print_level = MadNLP.ERROR,
                    config.solver_opts...,
                ),
                print_level = MadNLP.ERROR,
                config.ipm_opts...,
            )
            test_solver(
                prob,
                make_solver;
                broken = get(RELAXATION_BROKEN, (config_name, prob.name), Symbol[]),
            )
        end
    end
end
