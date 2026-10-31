module CCOpt
using NLPModels, MPCCModels, SolverCore, LinearAlgebra, SparseArrays
using Serialization
using Printf
using DataStructures: CircularBuffer, isfull
import MadNLPCore

include("types.jl")
include("LinearModel.jl")
include("LPCCModel.jl")
include("MPCCRelaxationModel.jl")
include("MPCCPenaltyModel.jl")

include("Models/scholtes_relaxation.jl")
include("Models/branch_nlp.jl")
include("Models/l1_exact_penalty.jl")
include("Models/big_m_model.jl")

include("utils.jl")
include("Solvers/common.jl")
include("Solvers/AbstractMPCCSolver.jl")
include("Solvers/homotopy_solver.jl")
include("Solvers/relaxation/Relaxation.jl")
include("Solvers/penalty/Penalty.jl")
include("Solvers/crossover.jl")

# This type is needed by the MOI wrapper.
global Optimizer

end # module CCOpt
