@testset "3. Dynamic Spatial Panels" begin
    _progress("3. Dynamic Spatial Panels")
    rng = MersenneTwister(303); L = 10; N = L * L; T_tot = 8; NT = N * T_tot; W = make_rook_weights(L)
    X = randn(rng, NT, 1); β_true = [0.8]; γ_true = 0.3; η_true = 0.15; ρ_true = 0.35; σ_true = 0.4
    μ_i = randn(rng, N)
    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    y = zeros(NT)
    y[1:N] = S_lu \ (μ_i .+ X[1:N, :] * β_true .+ σ_true .* randn(rng, N))
    for t in 2:T_tot
        idx = ((t - 1) * N + 1):(t * N)
        idx_prev = ((t - 2) * N + 1):((t - 1) * N)
        y[idx] = S_lu \ (μ_i .+ γ_true .* y[idx_prev] .+ η_true .* (W * y[idx_prev]) .+ X[idx, :] * β_true .+ σ_true .* randn(rng, N))
    end

    m_dyn = fit_dynamic_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_tot)
    @test length(coef(m_dyn)) == 4  # y_lag, Wy_lag, x1, ρ
    @test m_dyn.T_post == T_tot - 1
    @test isapprox(m_dyn.rho, ρ_true; atol=0.15)
end
