# CCOpt Options
```@meta
CurrentModule = CCOpt
```

CCOpt, like every nonlinear programming solver, has many options.
As CCopt is built on `MadNLP` we point the user to the [`MadNLP` options reference](https://madsuite.org/MadNLP.jl/stable/options/) for more details.

!!! info
    One important `MadNLP` option to note is `bound_relax_factor`. 
    When solving MPCCs it is important to set this to `0.0` to ensure the nonnegativity 
    of the complementarity is respected.
    
## Relaxation Solver
### Relaxation parameter updates
In the relaxation based solver we provide several options of how to drive the relaxation parameter to zero.

```@docs
AbstractRelaxationUpdate
AbstractFixedRelaxationUpdate
AbstractAdaptiveRelaxationUpdate
ProportionalRelaxationUpdate
RolloffRelaxationUpdate
LOQORelaxationUpdate
ScheduledRelaxationUpdate
```

### Endgame strategy
When approaching a solution of an MPCC using a progressively tighter relaxation, 
the problem of solving the linearized KKT system can become quite difficult.
We implement a safeguarding heuristic for this which selectively relaxes the 
lower bounds of the complementarity variables, based on multiplier estimates.
```@docs
AbstractEndgameStrategy
NoEndgameStrategy
RelaxLBEndgameStrategy
```

### Q-Regularization
In order to regularize the non-convex contribution of the relaxed complementarity 
constraints to the Hessian of the Lagrangian function, we provide two choices
of specialized algorithm:

- `critical_rho`: use a maximal factor (in `[0,1]`) which makes each two by two complementarity hessian positive definite.
- `eigenvalue_decomposition`: projects the matrix into the space of positive matrices by clipping the minimum eigenvalue.

## List of Options
### Relaxation Solver
```@docs
RelaxationOptions
```
### Penalty Solver
```@docs
PenaltyOptions
```