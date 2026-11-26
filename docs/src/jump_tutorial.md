# JuMP Tutorial
On this page we provide a brief documentation for how to use `CCOpt` with the `JuMP` modelling tool.

## Prerequisites
In order to use `CCOpt` to solve complementarity problems modelled in [`JuMP`](https://jump.dev/JuMP.jl/stable/) you need to first install the required packages:
```@example JuMP_tutorial
using JuMP
using MathOptComplements
using NLPModelsJuMP
using CCOpt
```

The package `MathOptComplements` in particular provides the required reformulations which take `MOI.Complments` constraints applies the necessary reformulation to the structure of problem required by `CCOpt`.

## Modelling
```@example JuMP_tutorial
model = Model()
@variable(model, z[1:2] >= 0)
@objective(model, Min, z[1] + z[2])
@constraint(model, c1, z[2]^2 >= 1)
@constraint(model, comp, [z[1], z[2]] ∈ MOI.Complements(2))
println(model)
```

Modelling an MPCC in `JuMP` is done through the use of [`∈ MOI.Complements(2)`](@extref JuMP :jl:type:`MathOptInterface.Complements`).
Note that mixed complementarity constraints are reformulated to standard complementarity constraints, as these are not supported directly by `CCOpt`.

## Solving
```@example JuMP_tutorial
MathOptComplements.Bridges.add_all_bridges(model)
set_optimizer(model, CCOpt.Optimizer)
JuMP.optimize!(model)
```

In order to solve the problem using CCOpt, add the required `MathOptComplements` bridges, set the optimizer and call `optimize!` as you are used to.

To extract the solution you can use:

```@example JuMP_tutorial
println(solution_summary(model))
println("z: $(value(z))")
```

