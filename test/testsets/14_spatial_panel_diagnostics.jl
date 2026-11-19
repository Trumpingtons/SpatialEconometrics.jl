@testset "14. Spatial Panel Diagnostics" begin
    _progress("14. Spatial Panel Diagnostics")
    rng = MersenneTwister(1400); L = 10; N = L * L; T_per = 8; NT = N * T_per
    W = make_rook_weights(L)
    X = randn(rng, NT, 2); β_true = [1.5, -0.8]; ρ_true = 0.45
    μ_i = randn(rng, N); ξ_t = randn(rng, T_per)
    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    y = zeros(NT)
    for t in 1:T_per
        idx = ((t-1)*N+1):(t*N)
        y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N))
    end

    # Spatial Chow test
    ct = spatial_chow_test(y, X, W, N, T_per, 4)
    @test ct.stat >= 0.0
    @test 0.0 <= ct.p_value <= 1.0
    @test ct.break_period == 4

    # Joint LM tests
    X_with_const = hcat(ones(NT), X)
    plm = panel_lm_tests(y, X_with_const, W, N, T_per)
    @test plm.joint_stat >= 0.0
    @test plm.lag_stat >= 0.0
    @test plm.error_stat >= 0.0
    @test plm.re_stat >= 0.0
    @test 0.0 <= plm.joint_p <= 1.0
    # Should detect spatial dependence
    @test plm.lag_p < 0.10 || plm.error_p < 0.10

    # Panel Moran's I
    m_pan = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
    pm = panel_moran(residuals(m_pan), W, N, T_per)
    @test pm.T_periods == T_per
    @test length(pm.period_results) == T_per
    @test isfinite(pm.pooled.I_stat)

    # Hausman test (compare two panel models with different FE)
    m_fe = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_per)
    m_twfe = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
    ht = spatial_hausman_test(m_fe, m_twfe)
    # IndivFE against TwoWayFE is a near-cancellation, so the variance
    # difference often has no positive direction at all and the test is
    # not computable. df reports how many directions were used.
    @test 0 <= ht.df <= 3
    if ht.df == 0
        @test isnan(ht.stat) && isnan(ht.p_value)
    else
        @test ht.stat >= 0.0
        @test 0.0 <= ht.p_value <= 1.0
    end
end
