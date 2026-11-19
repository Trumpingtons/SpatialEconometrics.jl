@testset "17b. Simulation DGPs and Monte Carlo bookkeeping (finding 22)" begin
    _progress("17b. Simulation DGPs and Monte Carlo bookkeeping (finding 22)")
    L = 5; W = make_rook_weights(L); N = size(W, 1); Wd = Matrix(W)
    X = randn(MersenneTwister(77), N, 2)
    b = [1.0, -0.5]; th = [0.4, 0.2]; s = 0.8; ρ = 0.5; λ = 0.3
    Ai = inv(I - ρ * Wd); Bi = inv(I - λ * Wd)

    # ── Each DGP against its closed-form mean and covariance ──
    # With X held fixed, y is multivariate normal with a known mean and
    # covariance for every one of the six processes. A sign error or a
    # swapped A/B filter shows up immediately here.
    cases = [
        ("SAR",  SAR_DGP(ρ, b, s),      Ai * (X * b),            s^2 * Ai * Ai'),
        ("SEM",  SEM_DGP(λ, b, s),      X * b,                   s^2 * Bi * Bi'),
        ("SDM",  SDM_DGP(ρ, b, th, s),  Ai * (X * b + Wd * X * th), s^2 * Ai * Ai'),
        ("SLX",  SLX_DGP(b, th, s),     X * b + Wd * X * th,     s^2 * Matrix(I, N, N)),
        ("SAC",  SAC_DGP(ρ, λ, b, s),   Ai * (X * b),            s^2 * Ai * Bi * Bi' * Ai'),
        ("SDEM", SDEM_DGP(λ, b, th, s), X * b + Wd * X * th,     s^2 * Bi * Bi'),
    ]
    R = 8_000
    for (nm, dgp, m_want, V_want) in cases
        rg = MersenneTwister(4242)
        acc = zeros(N); M2 = zeros(N, N)
        # X is passed through untouched, so the caller can reuse it.
        @test simulate_spatial(dgp, W, X; rng=rg)[2] === X
        for _ in 1:R
            y, _ = simulate_spatial(dgp, W, X; rng=rg)
            acc .+= y; M2 .+= y * y'
        end
        m_hat = acc ./ R
        V_hat = M2 ./ R .- m_hat * m_hat'
        # Mean within a few Monte Carlo standard errors.
        mc_se = sqrt.(diag(V_want) ./ R)
        @test all(abs.(m_hat .- m_want) .<= 4.5 .* mc_se)
        # Covariance within 10% of the largest entry.
        @test maximum(abs.(V_hat .- V_want)) < 0.10 * maximum(abs.(V_want))
    end

    # σ scales the standard deviation, not the variance: doubling σ must
    # double the spread of y about its mean.
    rg_a = MersenneTwister(99); rg_b = MersenneTwister(99)
    ya = [simulate_spatial(SAR_DGP(ρ, b, 1.0), W, X; rng=rg_a)[1] for _ in 1:3000]
    yb = [simulate_spatial(SAR_DGP(ρ, b, 2.0), W, X; rng=rg_b)[1] for _ in 1:3000]
    sa = std(reduce(hcat, ya), dims=2); sb = std(reduce(hcat, yb), dims=2)
    @test isapprox(sb ./ sa, fill(2.0, N); rtol=0.08)

    # ── Monte Carlo bookkeeping ──
    Wm = make_rook_weights(8); Nm = size(Wm, 1)
    CSx = SpatialEconometrics.CrossSection
    dgp_m = SAR_DGP(0.4, [1.0, -0.5], 1.0)
    mc = monte_carlo_spatial(dgp_m, Wm, Nm, 2,
        (w, x, y) -> fit_spatial(CSx.SAR(), w, x, y);
        config=MonteCarloConfig(n_replications=60, seed=5))
    @test mc.n_successful == 60
    @test all(isfinite, mc.param_means)
    # bias is mean minus truth, exactly.
    @test isapprox(mc.param_bias, mc.param_means .- [1.0, -0.5, 0.4]; rtol=1e-12)
    # RMSE combines bias and spread; it can never be below |bias|.
    @test all(mc.param_rmse .>= abs.(mc.param_bias) .- 1e-12)
    @test isapprox(mc.param_rmse,
                   sqrt.(mc.param_bias .^ 2 .+ mc.param_stds .^ 2); rtol=1e-12)
    # A fixed seed must give a reproducible experiment.
    mc2 = monte_carlo_spatial(dgp_m, Wm, Nm, 2,
        (w, x, y) -> fit_spatial(CSx.SAR(), w, x, y);
        config=MonteCarloConfig(n_replications=60, seed=5))
    @test mc.param_means == mc2.param_means

    # ── Finding 22: a short coefficient vector must not read as zero ──
    # SimShortMock reports an intercept and two slopes and no spatial
    # parameter, so the DGP's ρ row has nothing behind it.
    mc_s = @test_logs (:warn, r"trailing 1 row") match_mode=:any monte_carlo_spatial(
        dgp_m, Wm, Nm, 2,
        (w, x, y) -> SimShortMock(hcat(ones(Nm), x) \ y);
        config=MonteCarloConfig(n_replications=30, seed=3))
    @test mc_s.n_successful == 30            # the fits did succeed
    @test all(isfinite, mc_s.param_means[1:2])   # β rows are real estimates
    @test isnan(mc_s.param_means[3])         # ρ was never estimated
    @test isnan(mc_s.param_stds[3])
    @test isnan(mc_s.param_bias[3])
    @test isnan(mc_s.param_rmse[3])
    # The β rows must be unaffected by the NaN beside them.
    @test isapprox(mc_s.param_means[1], 1.0; atol=0.2)
    @test isapprox(mc_s.param_means[2], -0.5; atol=0.2)

    # ── Every replication failing is an error, not an empty table ──
    @test_throws ErrorException monte_carlo_spatial(
        dgp_m, Wm, Nm, 2, (w, x, y) -> error("no fit");
        config=MonteCarloConfig(n_replications=5, seed=1))

    # ── Diagnostic rejection rates are sized under the null ──
    # The three OLS diagnostics are computed from the simulated data and do
    # not depend on the estimator, so a cheap mock keeps this fast.
    mc_null = monte_carlo_spatial(SAR_DGP(0.0, [1.0, -0.5], 1.0), Wm, Nm, 2,
        (w, x, y) -> SimFullMock(vcat(hcat(ones(Nm), x) \ y, 0.0));
        config=MonteCarloConfig(n_replications=300, seed=21))
    for k in ("moran_05", "lm_lag_05", "lm_error_05")
        @test 0.0 <= mc_null.rejection_rates[k] <= 0.12   # nominal 0.05
    end
    # …and have power against a genuine lag process.
    mc_alt = monte_carlo_spatial(SAR_DGP(0.5, [1.0, -0.5], 1.0), Wm, Nm, 2,
        (w, x, y) -> SimFullMock(vcat(hcat(ones(Nm), x) \ y, 0.0));
        config=MonteCarloConfig(n_replications=300, seed=22))
    @test mc_alt.rejection_rates["lm_lag_05"] > 0.5
    @test mc_alt.rejection_rates["lm_lag_05"] > mc_null.rejection_rates["lm_lag_05"]
end
