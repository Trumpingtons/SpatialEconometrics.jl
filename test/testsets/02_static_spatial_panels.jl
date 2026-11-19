@testset "2. Static Spatial Panels" begin
    _progress("2. Static Spatial Panels")
    rng = MersenneTwister(202); L = 12; N = L * L; T_per = 6; NT = N * T_per; W = make_rook_weights(L)
    X = randn(rng, NT, 2); β_true = [1.5, -0.8]; ρ_true = 0.45; σ_true = 0.5
    μ_i = randn(rng, N); ξ_t = randn(rng, T_per)
    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    y = zeros(NT)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ σ_true .* randn(rng, N))
    end

    m_panel = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per; approach=LeeYuTransformation())
    @test isapprox(m_panel.rho, ρ_true; atol=0.12)
    @test length(coef(m_panel)) == 3  # x1, x2, ρ
    @test nobs(m_panel) == NT
    @test length(residuals(m_panel)) == NT

    m_dqml = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_per; approach=DirectQML(bias_correct=true))
    @test length(coef(m_dqml)) == 3
end
