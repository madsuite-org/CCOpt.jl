#=
    Solver-independent check of MPCC first-order conditions.

    For a vertical-form MPCC

        min f(x)  s.t.  lc ≤ c(x) ≤ uc,  lx ≤ x ≤ ux,  0 ≤ x₁ - l₁ ⟂ x₂ - l₂ ≥ 0

    a point is S-stationary if there are multipliers (y, z_L, z_U, ν₁, ν₂) with

        ∇f(x) + ∇c(x)ᵀy - z_L + z_U = 0,     where z_L[x₁] = ν₁ and z_L[x₂] = ν₂,
        z_L, z_U ≥ 0 and complementary to the bounds of the non-complementarity variables,
        ν₁ = 0 where x₁ > l₁,  ν₂ = 0 where x₂ > l₂,
        ν₁ ≥ 0 and ν₂ ≥ 0 on biactive pairs.

    Multipliers follow MadNLP's (Ipopt's) sign convention. For maximization problems the
    conditions are those of `min -f`: MadNLP reports `y` with flipped sign but `z_L`, `z_U`
    unflipped, so we use `-∇f` and `-y`.
=#

"""
    mpcc_residuals(mpcc, x, y, zl, zu, nu1, nu2; act_tol = 1e-4)

Return a `NamedTuple` of residuals of the S-stationarity conditions, each of which is zero
at an exact S-stationary point:

- `primal`: violation of the constraint and variable bounds,
- `complementarity`: `max_i min(x₁ᵢ - l₁ᵢ, x₂ᵢ - l₂ᵢ)`,
- `stationarity`: ∞-norm of the gradient of the MPCC Lagrangian,
- `bound_duals`: negativity of `z_L`, `z_U` and their complementarity with the bounds,
  for variables outside the complementarity pairs,
- `nu_inactive`: size of `ν` on the side of a pair that is not at its bound,
- `sign`: negativity of `ν` on biactive pairs (zero iff S-stationary rather than only
  weakly stationary).

A side of a pair is considered at its bound when `xᵢ - lᵢ ≤ act_tol`.
"""
function mpcc_residuals(mpcc, x, y, zl, zu, nu1, nu2; act_tol = 1e-4)
    T = eltype(x)
    s = get_minimize(mpcc) ? one(T) : -one(T)
    lvar, uvar = get_lvar(mpcc), get_uvar(mpcc)
    ind1, ind2 = get_ind_cc1(mpcc), get_ind_cc2(mpcc)
    is_cc = falses(length(x))
    is_cc[ind1] .= true
    is_cc[ind2] .= true

    # Primal feasibility
    primal = max(zero(T), maximum(lvar .- x), maximum(x .- uvar))
    if get_ncon(mpcc) > 0
        cx = cons(mpcc, x)
        primal = max(primal, maximum(get_lcon(mpcc) .- cx), maximum(cx .- get_ucon(mpcc)))
    end

    # Complementarity
    r1 = x[ind1] .- lvar[ind1]
    r2 = x[ind2] .- lvar[ind2]
    complementarity = maximum(abs.(min.(r1, r2)); init = zero(T))

    # Stationarity of the MPCC Lagrangian
    zl_mpcc = copy(zl)
    zl_mpcc[ind1] .= nu1
    zl_mpcc[ind2] .= nu2
    g = s .* grad(mpcc, x) .- zl_mpcc .+ zu
    if get_ncon(mpcc) > 0
        g .+= jac(mpcc, x)' * (s .* y)
    end
    stationarity = norm(g, Inf)

    # Bound multipliers of the variables outside the complementarity pairs
    bound_duals = zero(T)
    for i in eachindex(x)
        is_cc[i] && continue
        bound_duals = max(bound_duals, -zl[i], -zu[i])
        isfinite(lvar[i]) && (bound_duals = max(bound_duals, abs(zl[i] * (x[i] - lvar[i]))))
        isfinite(uvar[i]) && (bound_duals = max(bound_duals, abs(zu[i] * (uvar[i] - x[i]))))
    end

    # MPCC multipliers
    active1 = r1 .<= act_tol
    active2 = r2 .<= act_tol
    nu_inactive = max(
        maximum(abs.(nu1[.!active1]); init = zero(T)),
        maximum(abs.(nu2[.!active2]); init = zero(T)),
    )
    biactive = active1 .& active2
    sign = max(
        maximum(.-nu1[biactive]; init = zero(T)),
        maximum(.-nu2[biactive]; init = zero(T)),
        zero(T),
    )

    return (; primal, complementarity, stationarity, bound_duals, nu_inactive, sign)
end

"""
    mpcc_residuals(mpcc, stats::CCOpt.CCOptExecutionStats; kwargs...)

Residuals at the solution and multipliers reported by a CCOpt solver.
"""
function mpcc_residuals(mpcc, stats::CCOpt.CCOptExecutionStats; kwargs...)
    return mpcc_residuals(
        mpcc,
        stats.solution,
        stats.multipliers,
        stats.multipliers_L,
        stats.multipliers_U,
        stats.multipliers_x1,
        stats.multipliers_x2;
        kwargs...,
    )
end

"""
    test_solver(prob::TestProblem, make_solver; atol = 1e-5, broken = Symbol[])

Build a solver with `make_solver(prob.mpcc)`, solve, and run `test_solution`. If `broken`
contains `:crash`, the solve is expected to throw and is recorded as `@test_broken`.
"""
function test_solver(prob, make_solver; atol = 1e-5, broken = Symbol[])
    if :crash in broken
        @test_broken (CCOpt.solve_homotopy!(make_solver(prob.mpcc)); true)
        return nothing
    end
    stats = CCOpt.solve_homotopy!(make_solver(prob.mpcc))
    return test_solution(prob, stats; atol = atol, broken = broken)
end

"""
    test_solution(prob::TestProblem, stats; atol = 1e-5, broken = Symbol[])

Run the standard battery of tests on a solver result: success status, distance to the
known solution and objective, and every S-stationarity residual. Checks whose names are in
`broken` are recorded as `@test_broken` (known failures).
"""
function test_solution(prob, stats; atol = 1e-5, broken = Symbol[])
    T = typeof(prob.f_star)
    res = mpcc_residuals(prob.mpcc, stats; act_tol = sqrt(T(atol)))
    checks = (
        status = stats.status ∈ (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL),
        solution = any(x -> isapprox(stats.solution, x; atol = atol), prob.x_star),
        objective = isapprox(stats.objective, prob.f_star; atol = atol),
        (k => v ≤ atol for (k, v) in pairs(res))...,
    )
    for (name, ok) in pairs(checks)
        @testset "$name" begin
            if name in broken
                @test_broken ok
            else
                @test ok
            end
        end
    end
    return res
end
