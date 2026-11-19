@testset "17. Spatial Simulation & Monte Carlo" begin
    _progress("17. Spatial Simulation & Monte Carlo")
    rng = MersenneTwister(1700); L = 10; N = L * L
    W = make_rook_weights(L)

    # SAR DGP
    dgp_sar = SAR_DGP(0.5, [1.5, -0.8], 0.5)
    X = randn(rng, N, 2)
    y, _ = simulate_spatial(dgp_sar, W, X; rng=rng)
    @test length(y) == N
    @test all(isfinite, y)

    # SEM DGP
    dgp_sem = SEM_DGP(0.4, [1.0, -0.5], 0.6)
    y_sem, _ = simulate_spatial(dgp_sem, W, X; rng=rng)
    @test length(y_sem) == N

    # SDM DGP
    dgp_sdm = SDM_DGP(0.3, [1.0, -0.5], [0.4, -0.2], 0.5)
    y_sdm, _ = simulate_spatial(dgp_sdm, W, X; rng=rng)
    @test length(y_sdm) == N

    # SLX DGP
    dgp_slx = SLX_DGP([1.0, -0.5], [0.3, -0.1], 0.4)
    y_slx, _ = simulate_spatial(dgp_slx, W, X; rng=rng)
    @test length(y_slx) == N

    # SAC DGP
    dgp_sac = SAC_DGP(0.3, 0.2, [1.0, -0.5], 0.5)
    y_sac, _ = simulate_spatial(dgp_sac, W, X; rng=rng)
    @test length(y_sac) == N

    # SDEM DGP
    dgp_sdem = SDEM_DGP(0.4, [1.0, -0.5], [0.3, -0.2], 0.5)
    y_sdem, _ = simulate_spatial(dgp_sdem, W, X; rng=rng)
    @test length(y_sdem) == N

    # Monte Carlo with few reps (just checking it runs)
    mc = monte_carlo_spatial(
        SAR_DGP(0.5, [1.5, -0.8], 0.5), W, N, 2,
        (W, X, y) -> fit_spatial(SAR(), W, X, y);
        config=MonteCarloConfig(n_replications=10, seed=42)
    )
    @test length(mc.param_means) == 3  # β1, β2, ρ
    @test length(mc.param_bias) == 3
    @test all(isfinite, mc.param_rmse)
    @test haskey(mc.rejection_rates, "moran_05")
end
