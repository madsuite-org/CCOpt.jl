# CCOpt

A Julia package for solving Mathematical Programs with Complementarity Constraints (MPCCs).
For details check out the [implementation paper](https://arxiv.org/abs/2604.18726).

| **License** | **Documentation** | **Build Status** |
|:-----------:|:-----------------:|:----------------:|
| [![License: MIT][license-img]][license-url] | [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] |

[license-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/madsuite-org/CCOpt.jl/blob/master/LICENSE
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://madsuite.org/CCOpt.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://madsuite.org/CCOpt.jl/dev
[build-gh-img]: https://github.com/madsuite-org/CCOpt.jl/actions/workflows/action.yml/badge.svg
[build-gh-url]: https://github.com/madsuite-org/CCOpt.jl/actions/workflows/action.yml


## Installation

To install CCOpt, simply proceed to
```julia
pkg> add CCOpt
```

## Usage

CCOpt takes as input a nonlinear program formulated with [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/). Taking a `nlp` as input, a MPCC is defined using the package [MPCCModels](https://github.com/MadNLP/MPCCModels.jl/)
```julia
using CCOpt
using MadNLP  # provides the default sparse linear solver (MUMPS)
using MPCCModels
mpcc = MPCCModel(nlp, ind_x1, ind_x2)
```
with `ind_x1` (resp. `ind_x2`) the indices of the variables appearing in the left-hand complementarity
(resp. right-hand complementarity).

CCOpt depends on MadNLPCore, which has no default sparse linear solver. Load MadNLP, as above, or pass the option `linear_solver` and load the package implementing it.

### Formulating a MPCC with JuMP

CCOpt supports the modeler JuMP with the extension [MathOptComplements](https://github.com/blegat/MathOptComplements.jl). The following example shows how to formulate a MPCC with JuMP and solve it with CCOpt:

```julia
using JuMP
using MathOptComplements
using NLPModelsJuMP
using CCOpt
using MadNLP

model = Model()
@variable(model, z[1:2] >= 0)
@objective(model, Min, z[1] + z[2])
@constraint(model, c1, z[2]^2 >= 1)
@constraint(model, comp, [z[1], z[2]] ∈ MOI.Complements(2))

MathOptComplements.Bridges.add_all_bridges(model)
set_optimizer(model, CCOpt.Optimizer)
JuMP.optimize!(model)

```

## Solution methods

### Relaxation method

Once specified, you can solve the MPCC problem implemented in `mpcc` using the relaxation method as
```julia
solver = CCOpt.RelaxationSolver(mpcc)
stats = CCOpt.solve_homotopy!(solver)
```
All the results (primal and dual solutions, objective, etc.) are stored in `stats`.

### Penalty method

Alternatively, you can solve `mpcc` using the penalty method as
```julia

solver = CCOpt.PenaltySolver(mpcc)
stats = CCOpt.solve_homotopy!(solver)
```

## Citation

If you use `CCOpt.jl` in your work, please cite:

```bibtex
@article{Pozharskiy2026,
  title={CCOpt: an Open-Source Solver for Large-Scale Mathematical Programs with Complementarity Constraints},
  author={Pozharskiy, Anton and Pacaud, Fran{\c{c}}ois and Diehl, Moritz and Nurkanovi{\'c}, Armin},
  journal={arXiv preprint arXiv:2604.18726},
  year={2026}
}
