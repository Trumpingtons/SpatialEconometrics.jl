@testset "16. Spatial Regime Models" begin
    _progress("16. Spatial Regime Models")
    rng = MersenneTwister(1600); W = make_rook_weights(15); N = size(W, 1)
    X = randn(rng, N, 2); ρ_true = 0.4
    IN = sparse(1.0I, N, N)

    # Create two regimes: left half vs right half of grid
    labels = [((i - 1) % 15) < 8 ? 1 : 2 for i in 1:N]
    regimes = ExogenousRegimes(labels)

    # DGP with different coefficients per regime
    β_r1 = [2.0, 1.0, -0.5]; β_r2 = [0.5, -0.8, 1.2]
    y = zeros(N)
    for i in 1:N
        β = labels[i] == 1 ? β_r1 : β_r2
        y[i] = β[1] + X[i, :]' * β[2:3]
    end
    y .+= 0.5 .* randn(rng, N)
    y = (IN - ρ_true * W) \ y

    # Fit regime model with spatial lag
    m_reg = fit_spatial_regime(W, X, y, regimes; spatial_lag=true, varnames=["x1", "x2"])
    @test m_reg.n_regimes == 2
    @test m_reg.has_spatial_lag
    @test length(coef(m_reg)) == 7  # 2 regimes × 3 params + ρ
    @test isfinite(loglikelihood(m_reg))

    # Fit without spatial lag
    m_ols = fit_spatial_regime(W, X, y, regimes; spatial_lag=false, varnames=["x1", "x2"])
    @test !m_ols.has_spatial_lag
    @test length(coef(m_ols)) == 6  # 2 regimes × 3 params

    # Regime Chow test
    m_pooled = fit_spatial(SAR(), W, X, y)
    ct = regime_chow_test(m_pooled, m_reg)
    @test ct.stat >= 0.0
    @test 0.0 <= ct.p_value <= 1.0
    @test ct.df == length(coef(m_reg)) - length(coef(m_pooled))

    # SKATER regimes
    coords = hcat(repeat(1.0:15, 15), repeat(1.0:15, inner=15))
    sr = skater_regimes(W, coords, 3)
    @test maximum(sr.labels) == 3
    @test length(sr.labels) == N
    @test minimum(sr.labels) == 1
end
