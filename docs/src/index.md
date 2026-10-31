
# CCOpt

CCOpt is an open-source nonlinear programming solver
for mathematical programs with complementarity constraints (MPCCs).
It is built on top of [MadNLP](https://github.com/MadNLP/MadNLP.jl),

## Design

### CCOpt's problem structure
CCOpt solves MPCCs formulated as
```math
  \begin{aligned}
    \min_{x} \; & f(x) \\
    \text{subject to} \quad & g_\ell \leq g(x) \leq g_u \\
                            & x_0^\flat \leq x_0 \leq x_0^\sharp \\
                            & x_1^\flat \leq x_1 \perp x_2 \geq x_2^\flat
  \end{aligned}
```
where $$x = (x_0, x_1, x_2) \in \mathbb{R}^n$$ is the decision variable,
$$f: \mathbb{R}^n \to \mathbb{R}$$ the objective function,
and $$g: \mathbb{R}^n \to \mathbb{R}^m$$ the nonlinear constraints.

!!! info
    CCOpt assumes that the complementarity constraints port only on a subset
    of the optimization variables $$(x_1, x_2)$$.

!!! warning
    For the problem to be well-specified, the variables $$x_1$$ and $$x_2$$
    should be lower-bounded, with no upper-bound. The variable $$x_0$$ can be anything.

### Algorithms

CCOpt implements several solution algorithms for MPCCs, all based on
the interior-point method. The two main algorithms are:

- `CCOpt.RelaxationSolver`: implements the relaxation method of [Raghunathan & Biegler](https://epubs.siam.org/doi/abs/10.1137/S1052623403429081) with proper safeguards.
- `CCOpt.PenaltySolver`: implements the $$\ell_1$$-penalty method of [Leyffer & al](https://epubs.siam.org/doi/abs/10.1137/040621065), with refinements.

In addition, CCOpt implements the crossover described in [this article](https://arxiv.org/abs/2501.13835) to identify a B-stationarity point with guarantees.


### Input format

CCOpt takes as input a MPCC formulated using the package [MPCCModels](https://github.com/MadNLP/MPCCModels.jl/) which is built on top of [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/). The user has to pass explicitly the indices for the left-hand complementarity variable $$x_1$$ and the right-hand complementarity variable $$x_2$$. The problem is specified as
```julia
using MPCCModels
mpcc = MPCCModel(nlp, ind_x1, ind_x2)

```

You can build `nlp` using your favorite modeler, including:
- JuMP (with [NLPModelsJuMP](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/)),
- CasADi (with [CasADiNLPModels](https://github.com/apozharski/CasADiNLPModels.jl/)),
- AMPL (with [AmplNLReader](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl/))
- [ExaModels](https://github.com/exanauts/ExaModels.jl).

In addition, CCOpt supports complementarity problems formulated using
[MathOptComplements](https://github.com/blegat/ComplementOpt.jl), a JuMP extension built for MPCCs.


### Linear solvers

CCOpt supports all the sparse linear solvers [wrapped inside MadNLP](https://madsuite.org/MadNLP.jl/dev/#Linear-solvers). CCOpt depends on MadNLPCore, which has no default sparse linear solver: load MadNLP to use MUMPS by default, or set the option `linear_solver`. We recommend using the HSL solvers for fastest performance.
