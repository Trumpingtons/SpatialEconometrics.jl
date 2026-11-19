@testset "22. Regressions found by the example scripts" begin
    _progress("22. Regressions found by the example scripts")
    W = make_rook_weights(12); N = size(W, 1)
    IN = sparse(1.0I, N, N)
    rng = MersenneTwister(4242)
    X = randn(rng, N, 2)
    y = (IN - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))

    # ── show() must not throw ──
    # Several modules called nobs/stderror inside Base.show without
    # importing them from StatsAPI. The suite never rendered a model, so
    # printing any of these threw UndefVarError for users.
    for m in (fit_slx(W, X, y), fit_sdem(W, X, y), fit_sac(W, W, X, y),
              fit_esf(X, y, W; max_eigenvectors=5), fit_re_esf(X, y, W),
              fit_spatial_regime(W, X, y,
                  ExogenousRegimes([i <= N ÷ 2 ? 1 : 2 for i in 1:N])))
        str = sprint(show, m)
        @test !isempty(str)
        @test occursin("N = ", str) || occursin("N  ", str)
    end

    # ── chi-squared survival function ──
    # The regime module used a truncated lower-incomplete-gamma series that
    # only converges for z < a+1. Past that it returned a tiny partial sum,
    # so the survival function came back near 1 and a decisive rejection was
    # reported as "fail to reject".
    RM = SpatialEconometrics.SpatialRegimeModels
    MS = SpatialEconometrics.SpatialModelSelection
    for (x, df) in [(0.5, 1), (3.841, 1), (6.635, 1), (7.815, 3),
                    (11.345, 3), (2.0, 3), (100.0, 3), (50.0, 10),
                    (1.0, 10), (499.5, 3), (1000.0, 5)]
        a = RM._chi2_sf(x, df)
        @test 0.0 <= a <= 1.0
        # Must agree with the independently written implementation.
        @test a ≈ MS._chi2_sf(x, df) rtol = 1e-8
    end
    # Reference values (R: pchisq(q, df, lower.tail=FALSE)).
    @test RM._chi2_sf(3.841, 1) ≈ 0.05 atol = 1e-4
    @test RM._chi2_sf(7.815, 3) ≈ 0.05 atol = 1e-4
    @test RM._chi2_sf(11.345, 3) ≈ 0.01 atol = 1e-4
    @test RM._chi2_sf(0.5, 1) ≈ 0.4795001 atol = 1e-6
    # The case that was silently wrong: a huge statistic must reject.
    @test RM._chi2_sf(499.5, 3) < 1e-100
    # log-gamma must stay finite where Gamma itself would overflow.
    @test isfinite(RM._chi2_sf(1000.0, 500))

    # A regime Chow test on a genuine structural break must reject.
    β_a = [1.0, 2.0, -0.5]; β_b = [1.0, 0.3, -1.6]
    labels = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]
    η = similar(X[:, 1])
    for i in 1:N
        b = labels[i] == 1 ? β_a : β_b
        η[i] = b[1] + X[i, 1] * b[2] + X[i, 2] * b[3]
    end
    y_br = (IN - 0.45 * W) \ (η .+ 0.4 .* randn(MersenneTwister(9), N))
    m_pool = fit_spatial(SpatialEconometrics.SAR(), W, X, y_br)
    m_reg = fit_spatial_regime(W, X, y_br, ExogenousRegimes(labels))
    chow = regime_chow_test(m_pool, m_reg)
    @test chow.stat > 0.0
    @test chow.df == length(coef(m_reg)) - length(coef(m_pool))
    @test chow.p_value < 1e-6          # was reported as 0.998

    # ── SKATER must not produce degenerate regions ──
    # Cutting the LONGEST MST edge peels off a single outlying leaf, which
    # left the downstream regime regression rank-deficient.
    signal = hcat(Float64.(labels) .+ 0.3 .* randn(MersenneTwister(11), N), X[:, 1])
    for k in (2, 3)
        reg = skater_regimes(W, signal, k)
        sizes = [count(==(r), reg.labels) for r in 1:k]
        @test length(reg.labels) == N
        @test sort(unique(reg.labels)) == collect(1:k)
        @test minimum(sizes) >= 2        # no singleton regions
        @test sum(sizes) == N
    end
    # min_size is honoured, and the regions stay usable for a regression.
    reg20 = skater_regimes(W, signal, 2; min_size = 20)
    @test minimum([count(==(r), reg20.labels) for r in 1:2]) >= 20
    @test length(coef(fit_spatial_regime(W, X, y_br, reg20))) == 7
    @test_throws AssertionError skater_regimes(W, signal, 2; min_size = N)

    # ── Monte Carlo parameter alignment ──
    # The DGPs omit an intercept but the estimators fit one, so taking the
    # leading n_params coefficients compared the intercept against β₁ and
    # reported the off-by-one as enormous bias with a tiny SD.
    Wmc = make_rook_weights(10); Nmc = size(Wmc, 1)
    CSx = SpatialEconometrics.CrossSection
    mc = monte_carlo_spatial(
        SAR_DGP(0.5, [1.5, -0.8], 0.5), Wmc, Nmc, 2,
        (Wm, Xm, ym) -> fit_spatial(CSx.SAR(), Wm, Xm, ym);
        config = MonteCarloConfig(n_replications = 20, seed = 7))
    @test mc.n_successful == 20          # was 0 — every replication threw
    @test mc.n_replications == 20
    @test mc.param_names == ["β1", "β2", "ρ"]
    @test mc.param_means[1] ≈ 1.5 atol = 0.15
    @test mc.param_means[2] ≈ -0.8 atol = 0.15
    @test mc.param_means[3] ≈ 0.5 atol = 0.15
    @test all(abs.(mc.param_bias) .< 0.15)
    @test all(mc.param_stds .>= 0.0)
    # Diagnostics ran rather than being skipped by the estimator's `continue`.
    @test mc.rejection_rates["moran_05"] > 0.5
    @test !isempty(sprint(show, mc))

    # ── spatial_hausman_test must match coefficients by NAME ──
    # It aligned the two coefficient vectors by position, so comparing an
    # FE fit (no intercept) with a pooled fit (intercept) contrasted
    # β_income with the intercept and returned a statistic of 7.8e6: a
    # decisive-looking rejection assembled from unrelated parameters.
    let
        rngh = MersenneTwister(1717)
        Lh = 8; Nh = Lh * Lh; Th = 5; NTh = Nh * Th
        Wh = make_rook_weights(Lh)
        Sh = lu(sparse(1.0I, Nh, Nh) - 0.4 * Wh)
        Xh = randn(rngh, NTh, 2)
        yh = zeros(NTh)
        μh = 2.0 .* randn(rngh, Nh)
        for t in 1:Th
            idx = ((t - 1) * Nh + 1):(t * Nh)
            yh[idx] = Sh \ (μh .+ Xh[idx, :] * [1.0, -0.5] .+ 0.5 .* randn(rngh, Nh))
        end
        fp(fe) = fit_spatial_panel(SAR(), fe, Wh, Xh, yh, Nh, Th;
                                   varnames = ["a", "b"])
        m_feh = fp(IndivFE())
        m_pooh = fp(NoFE())

        # The premise of the bug: the two layouts genuinely differ.
        @test coefnames(m_feh) == ["a", "b", "ρ"]
        @test coefnames(m_pooh) == ["(Intercept)", "a", "b", "ρ"]

        # The NoFE fit is the less precise one -- an unmodelled
        # individual effect inflates σ̂² and so every standard error --
        # so it goes first, and V_fe - V_re is positive definite.
        ht = spatial_hausman_test(m_pooh, m_feh)
        # The pooled intercept has no FE counterpart, so it drops out.
        @test ht.coefnames == ["a", "b", "ρ"]
        @test ht.df == 3
        @test ht.stat >= 0.0
        @test 0.0 <= ht.p_value <= 1.0

        # Swapping the arguments compares the same set of names, but
        # the variance difference is then negative definite in every
        # direction, so there is nothing to invert and the test says
        # so instead of returning a number.
        ht_bad = spatial_hausman_test(m_feh, m_pooh)
        @test ht_bad.coefnames == ["a", "b", "ρ"]
        @test ht_bad.df == 0
        @test isnan(ht_bad.stat)
        @test all(eigvals(Symmetric(vcov(m_feh)[1:3, 1:3] .-
                                    vcov(m_pooh)[2:4, 2:4])) .< 0)

        # Comparing a model with itself is the degenerate case: identical
        # coefficients, so the statistic must be zero and p must be 1.
        ht_self = spatial_hausman_test(m_feh, m_feh)
        @test ht_self.stat ≈ 0.0 atol = 1e-6
        @test ht_self.p_value ≈ 1.0 atol = 1e-8

        # `names` restricts the comparison to a chosen subset.
        ht_sl = spatial_hausman_test(m_pooh, m_feh; names = ["a", "b"])
        @test ht_sl.coefnames == ["a", "b"]
        @test ht_sl.df == 2

        # An unknown name is an error, not a silently smaller comparison.
        @test_throws ErrorException spatial_hausman_test(m_pooh, m_feh;
                                                        names = ["nope"])
        # The compared names appear in the printed output.
        @test occursin("a, b, ρ", sprint(show, ht))
    end
end
