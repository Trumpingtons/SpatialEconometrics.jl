@testset "13. Model Selection & Comparison" begin
    _progress("13. Model Selection & Comparison")
    rng = MersenneTwister(1300); W = make_rook_weights(15); N = size(W, 1)
    X = randn(rng, N, 2)
    y = (I - 0.5 * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * [0.6, -0.3] .+ 0.5 .* randn(rng, N))

    m_sar = fit_spatial(SAR(), W, X, y)
    m_sdm = fit_spatial(SDM(), W, X, y)

    # AIC/BIC
    aic_sar = spatial_aic(m_sar); aic_sdm = spatial_aic(m_sdm)
    bic_sar = spatial_bic(m_sar); bic_sdm = spatial_bic(m_sdm)
    @test isfinite(aic_sar) && isfinite(aic_sdm)
    @test isfinite(bic_sar) && isfinite(bic_sdm)

    # LR test (SAR nested in SDM)
    lr = lr_test(m_sar, m_sdm)
    @test lr.stat >= 0.0
    @test 0.0 <= lr.p_value <= 1.0
    @test lr.df == 2  # SDM adds 2 WX terms

    # Vuong test
    m_sem = fit_spatial_error(W, X, y)
    vt = vuong_test(m_sar, m_sem)
    @test isfinite(vt.stat)
    @test 0.0 <= vt.p_value <= 1.0
    @test vt.preference in (:model1, :model2, :indistinguishable)

    # Wald test (H₀: β₂ = 0)
    R = zeros(1, length(coef(m_sar)))
    R[1, 3] = 1.0  # third coef = β₂
    r = [0.0]
    wt = wald_test(m_sar, R, r)
    @test wt.stat >= 0.0
    @test 0.0 <= wt.p_value <= 1.0

    # Common factor test
    β_sdm = coef(m_sdm)
    K = 2  # number of X variables
    beta_slope = β_sdm[2:K+1]
    theta = β_sdm[K+2:2K+1]
    rho = m_sdm.rho
    cft = common_factor_test(beta_slope, theta, rho, vcov(m_sdm))
    @test cft.stat >= 0.0
    @test 0.0 <= cft.p_value <= 1.0

    # Comparison table
    tbl = model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR", "SDM", "SEM"])
    @test occursin("SAR", tbl)
    @test occursin("SDM", tbl)
    @test occursin("SEM", tbl)
    @test occursin("AIC", tbl)
end
