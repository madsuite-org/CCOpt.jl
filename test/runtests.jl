using LinearAlgebra, SparseArrays, Test
using NLPModels, MPCCModels, ADNLPModels, CCOpt
using MadNLP

include("models/simple-model.jl")
include("models/simple-mpcc-model.jl")
include("problems.jl")
include("stationarity.jl")

include("test-stationarity.jl")
include("models/scholtes-relaxation.jl")
include("solvers/homotopy-solver.jl")
include("solvers/relaxation.jl")
include("solvers/penalty.jl")
include("solvers/resolve.jl")
include("solvers/known-limitations.jl")

if "MOI" in ARGS
    # only run MOI if requested!
    @testset "MOI interface" begin
        include("MOI_wrapper.jl")
    end
end
