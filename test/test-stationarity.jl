# Sanity checks of the S-stationarity checker itself, using analytic multipliers.
@testset "Stationarity checker" begin
    @testset "S-stationary biactive point" begin
        prob = biactive_s_stationary(Float64)
        res = mpcc_residuals(prob.mpcc, only(prob.x_star), Float64[], zeros(3), zeros(3), [1.0], [1.0])
        @test all(iszero, values(res))
    end

    @testset "C-stationary point is rejected" begin
        # At the origin of the two-branch problem ∇f = (-2, -2): the Lagrangian is stationary
        # with ν = (-2, -2), but the negative multipliers mean the point is not S-stationary.
        prob = two_branches(Float64)
        res = mpcc_residuals(prob.mpcc, [0.0, 0.0], Float64[], zeros(2), zeros(2), [-2.0], [-2.0])
        @test res.stationarity == 0
        @test res.sign == 2
    end

    @testset "Wrong multipliers are rejected" begin
        prob = two_branches(Float64)
        # At (1, 0): ν₁ must vanish (x₁ inactive), ν₂ = ∂f/∂x₂ = -2.
        res = mpcc_residuals(prob.mpcc, [1.0, 0.0], Float64[], zeros(2), zeros(2), [0.0], [-2.0])
        @test all(iszero, values(res))
        res = mpcc_residuals(prob.mpcc, [1.0, 0.0], Float64[], zeros(2), zeros(2), [0.5], [-2.0])
        @test res.nu_inactive == 0.5
        @test res.stationarity == 0.5
    end

    @testset "Constraints, maximization and shifted bounds" begin
        prob = multi_pair_constrained(Float64)
        q2 = sqrt(0.5)
        # Stationarity in w: 2(w - 1) + y₁ = 0 ⇒ y₁ = 1. In q₂: 2(q₂ - 1) + 2q₂y₂ = 0.
        y = [1.0, (1 - q2) / q2]
        nu1 = [0.0, 1.0, 0.0]          # ν for p: p₂ = 0 with ∂f/∂p₂ = 1
        nu2 = [1.0, 0.0, 2.0]          # ν for q: q₁ = q₃ = 0
        res = mpcc_residuals(prob.mpcc, only(prob.x_star), y, zeros(7), zeros(7), nu1, nu2)
        @test maximum(values(res)) ≤ 1e-12
        @test mpcc_residuals(prob.mpcc, only(prob.x_star), -y, zeros(7), zeros(7), nu1, nu2).stationarity > 0.5

        prob = shifted_bounds_max(Float64)
        # Conditions of min (x₁ - 3)² + x₂: ν₂ = 1.
        res = mpcc_residuals(prob.mpcc, only(prob.x_star), Float64[], zeros(2), zeros(2), [0.0], [1.0])
        @test all(iszero, values(res))
        res = mpcc_residuals(prob.mpcc, [3.0, -1.5], Float64[], zeros(2), zeros(2), [0.0], [1.0])
        @test res.primal == 0.5
    end
end
