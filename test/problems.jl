#=
    Collection of small MPCCs with analytically known solutions.

    Every problem is in vertical form, `0 ≤ x[ind_cc1] - l₁ ⟂ x[ind_cc2] - l₂ ≥ 0`, and is
    returned as a `TestProblem` holding the model together with its known solution.
    Problems are designed so that the reported solution is the *unique* S-stationary
    point (the other branches end at biactive points with negative MPCC multipliers),
    which makes the tests independent of which local solution an algorithm drifts to.

    To add a problem: write a constructor returning a `TestProblem` and append it to
    `TEST_PROBLEMS` at the bottom of the file.
=#

struct TestProblem{T,M<:AbstractMPCCModel{T}}
    name::String
    mpcc::M
    x_star::Vector{Vector{T}}  # known solution(s); a result must match one of them
    f_star::T            # known objective value (in the problem's own sense)
    biactive::Bool       # whether the solution has a biactive complementarity pair
end

"""
    single_pair(T)

The legacy `SimpleMPCCModel2`: one pair, no constraints, solution in the interior of a branch.

    min (x₁ - 0.1)² + (x₂ - 1)²   s.t.  0 ≤ x₁ ⟂ x₂ ≥ 0

Solution `x* = (0, 1)`, `f* = 0.01`. (The other branch `(0.1, 0)` costs 1.)
"""
function single_pair(T::Type)
    return TestProblem("single pair", SimpleMPCCModel2(T), [T[0, 1]], T(0.01), false)
end

"""
    two_branches(T)

The legacy `SimpleMPCCModel` started from `x₀ = (2, 1)`.

    min (x₁ - 1)² + (x₂ - 1)²   s.t.  0 ≤ x₁ ⟂ x₂ ≥ 0

Both `(1, 0)` and `(0, 1)` are global solutions with `f* = 1` (the starting point favours
`(1, 0)`, but either is accepted). The origin is a spurious C-stationary point
(`ν = (-2, -2)`).
"""
function two_branches(T::Type)
    mpcc = SimpleMPCCModel(T)
    copyto!(get_x0(mpcc), T[2, 1])
    return TestProblem("two branches", mpcc, [T[1, 0], T[0, 1]], T(1), false)
end

"""
    multi_pair_constrained(T)

Three complementarity pairs, one free variable, one linear equality and one nonlinear
inequality. The pairs are not adjacent in `x` so that index bookkeeping errors show up.

    x = (w, p₁, p₂, p₃, q₁, q₂, q₃),   pairs (pᵢ, qᵢ)

    min  (w - 1)² + (p₁ - 2)² + q₁ + p₂ + (q₂ - 1)² + (p₃ - 1)² + 2q₃
    s.t. w + p₁ = 2
         q₂² ≤ 1/2
         0 ≤ pᵢ ⟂ qᵢ ≥ 0,  i = 1, 2, 3

Solution `w = 1/2, p = (3/2, 0, 1), q = (0, 1/√2, 0)`, `f* = 1/2 + (1 - 1/√2)²`.
Both constraints are active.
"""
function multi_pair_constrained(T::Type)
    f(x) =
        (x[1] - 1)^2 + (x[2] - 2)^2 + x[5] + x[3] + (x[6] - 1)^2 + (x[4] - 1)^2 + 2 * x[7]
    c(x) = [x[1] + x[2], x[6]^2]
    x0 = T[1, 1, 1, 1, 1, 1, 1]
    lvar = T[-Inf, 0, 0, 0, 0, 0, 0]
    uvar = fill(T(Inf), 7)
    lcon = T[2, -Inf]
    ucon = T[2, 0.5]
    nlp = ADNLPModels.ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)
    mpcc = MPCCModel(nlp, [2, 3, 4], [5, 6, 7])
    q2 = sqrt(T(0.5))
    x_star = T[0.5, 1.5, 0, 1, 0, q2, 0]
    return TestProblem(
        "multiple pairs, constrained",
        mpcc,
        [x_star],
        T(0.5) + (1 - q2)^2,
        false,
    )
end

"""
    shifted_bounds_max(T)

Maximization problem whose complementarity variables have nonzero lower bounds, plus an
inactive upper bound.

    max  -(x₁ - 3)² - x₂
    s.t. 0 ≤ x₁ - 1 ⟂ x₂ + 1 ≥ 0,   x₁ ≤ 10

Solution `x* = (3, -1)`, `f* = 1`. The other branch ends at the biactive point `(1, -1)`,
which is not S-stationary.
"""
function shifted_bounds_max(T::Type)
    f(x) = -(x[1] - 3)^2 - x[2]
    nlp = ADNLPModels.ADNLPModel(f, T[2, 2], T[1, -1], T[10, Inf]; minimize = false)
    mpcc = MPCCModel(nlp, [1], [2])
    return TestProblem("shifted bounds, maximization", mpcc, [T[3, -1]], T(1), false)
end

"""
    fixed_variable(T)

A fixed variable placed *before* the complementarity variables, so that MadNLP's
fixed-variable removal shifts their internal indices (see `CCOpt._adjust_cc_inds!`).

    x = (y, x₁, x₂, z)

    min  (x₁ - 1 - y)² + x₂ + (z - y)²
    s.t. y = 1/2 (as the bounds 1/2 ≤ y ≤ 1/2)
         0 ≤ x₁ ⟂ x₂ ≥ 0

Solution `x* = (1/2, 3/2, 0, 1/2)`, `f* = 0`.
"""
function fixed_variable(T::Type)
    f(x) = (x[2] - 1 - x[1])^2 + x[3] + (x[4] - x[1])^2
    nlp = ADNLPModels.ADNLPModel(f, T[0.5, 1, 1, 1], T[0.5, 0, 0, -Inf], T[0.5, Inf, Inf, Inf])
    mpcc = MPCCModel(nlp, [2], [3])
    return TestProblem("fixed variable", mpcc, [T[0.5, 1.5, 0, 0.5]], T(0), false)
end

"""
    biactive_s_stationary(T)

Solution at a biactive pair with strictly positive MPCC multipliers (S-stationary, and the
global minimum).

    min  x₁ + x₂ + (z - 1)²   s.t.  0 ≤ x₁ ⟂ x₂ ≥ 0

Solution `x* = (0, 0, 1)`, `f* = 0`, with `ν = (1, 1)`.
"""
function biactive_s_stationary(T::Type)
    f(x) = x[1] + x[2] + (x[3] - 1)^2
    nlp = ADNLPModels.ADNLPModel(f, T[1, 1, 0], T[0, 0, -Inf], fill(T(Inf), 3))
    mpcc = MPCCModel(nlp, [1], [2])
    return TestProblem("biactive, S-stationary", mpcc, [T[0, 0, 1]], T(0), true)
end

"""
    fixed_complementarity_variable(T)

One side of the pair is a fixed variable. Not part of `TEST_PROBLEMS`: this case is not
supported yet (see the FIXME on `CCOpt._adjust_cc_inds!`).

    x = (z, x₁, x₂)

    min  (z - 1)² + x₁ + (x₂ - 2)²
    s.t. x₁ = 0 (as the bounds 0 ≤ x₁ ≤ 0),  0 ≤ x₁ ⟂ x₂ ≥ 0

Solution `x* = (1, 0, 2)`, `f* = 0`.
"""
function fixed_complementarity_variable(T::Type)
    f(x) = (x[1] - 1)^2 + x[2] + (x[3] - 2)^2
    nlp = ADNLPModels.ADNLPModel(f, T[0, 0, 1], T[-Inf, 0, 0], T[Inf, 0, Inf])
    mpcc = MPCCModel(nlp, [2], [3])
    return TestProblem("fixed complementarity variable", mpcc, [T[1, 0, 2]], T(0), false)
end

const TEST_PROBLEMS = [
    two_branches,
    single_pair,
    multi_pair_constrained,
    shifted_bounds_max,
    fixed_variable,
    biactive_s_stationary,
]
