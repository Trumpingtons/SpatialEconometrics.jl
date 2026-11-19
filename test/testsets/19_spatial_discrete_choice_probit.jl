@testset "19. Spatial Discrete Choice (Probit)" begin
    _progress("19. Spatial Discrete Choice (Probit)")
    rng = MersenneTwister(1900); L = 14; W = make_rook_weights(L); N = size(W, 1)
    X = randn(rng, N, 2)
    X_aug = hcat(ones(N), X)
    β_true = [0.3, 1.2, -0.9]; ρ_true = 0.4
    IN = sparse(1.0I, N, N)
    y_star = (IN - ρ_true * W) \ (X_aug * β_true .+ randn(rng, N))
    y = Float64.(y_star .> 0)
    # Guard against a degenerate draw making the tests vacuous.
    @test 0.2 < mean(y) < 0.8

    # ── Linearized GMM (Klier-McMillen) ──
    m_gmm = fit_spatial_probit(W, X, y; varnames=["x1", "x2"])
    @test length(coef(m_gmm)) == 4  # intercept + 2β + ρ
    @test coefnames(m_gmm) == ["(Intercept)", "x1", "x2", "ρ"]
    @test size(vcov(m_gmm)) == (4, 4)
    @test length(stderror(m_gmm)) == 4
    @test all(isfinite, coef(m_gmm))
    @test nobs(m_gmm) == N
    @test dof_residual(m_gmm) == N - 4
    @test abs(m_gmm.rho) < 1.0
    @test isfinite(loglikelihood(m_gmm))
    @test loglikelihood(m_gmm) < 0.0

    # Fitted probabilities are genuine probabilities and beat a coin flip.
    probs = predict(m_gmm)
    @test length(probs) == N
    @test all(p -> 0.0 <= p <= 1.0, probs)
    @test mean((probs .>= 0.5) .== (y .== 1.0)) > 0.65
    @test isapprox(residuals(m_gmm), y .- probs; atol=1e-10)

    # Slope signs must be recovered; ρ̂ is high-variance at this N
    # (SD ≈ 0.25 over seeds), so the band is deliberately wide.
    @test coef(m_gmm)[2] > 0.0
    @test coef(m_gmm)[3] < 0.0
    @test isapprox(m_gmm.rho, ρ_true; atol=0.35)

    # ── Marginal effects ──
    eff = marginal_effects(m_gmm)
    @test length(eff.direct) == 2
    @test eff.varnames == ["x1", "x2"]
    @test all(isfinite, eff.total)
    # total = direct + indirect, by construction
    @test isapprox(eff.direct .+ eff.indirect, eff.total; atol=1e-10)
    # Effects inherit the sign of their coefficient.
    @test eff.direct[1] > 0.0 && eff.total[1] > 0.0
    @test eff.direct[2] < 0.0 && eff.total[2] < 0.0
    # With ρ > 0 the spillover runs the same way as the direct effect.
    @test eff.indirect[1] > 0.0

    # ── Bayesian Gibbs (Albert-Chib) ──
    m_bayes = fit_bayesian_spatial_probit(W, X, y; n_samples=600, burnin=300,
        varnames=["x1", "x2"], rng=MersenneTwister(1901))
    @test length(coef(m_bayes)) == 4
    @test coefnames(m_bayes) == ["(Intercept)", "x1", "x2", "ρ"]
    @test size(m_bayes.beta_draws) == (600, 3)
    @test length(m_bayes.rho_draws) == 600
    @test all(r -> -1.0 < r < 1.0, m_bayes.rho_draws)
    @test all(p -> 0.0 <= p <= 1.0, predict(m_bayes))
    # The default proposal should land in the usable acceptance band.
    @test 0.15 < m_bayes.accept_rate < 0.6
    @test coef(m_bayes)[2] > 0.0
    @test coef(m_bayes)[3] < 0.0
    @test isapprox(m_bayes.rho, ρ_true; atol=0.3)
    # Posterior SD must be positive — a stuck chain would show zero spread.
    @test std(m_bayes.rho_draws) > 1e-4

    # ── Ordered probit (3 categories, no intercept) ──
    rng_o = MersenneTwister(1902)
    z_ord = (IN - 0.35 * W) \ (X * [1.0, -0.7] .+ randn(rng_o, N))
    y_ord = [zi <= 0.0 ? 1 : (zi <= 1.2 ? 2 : 3) for zi in z_ord]
    @test all(j -> count(==(j), y_ord) >= 10, 1:3)

    m_ord = fit_spatial_ordered_probit(W, X, y_ord; n_samples=600, burnin=300,
        varnames=["x1", "x2"], rng=MersenneTwister(1903))
    @test m_ord.n_categories == 3
    @test length(coef(m_ord)) == 3  # 2β + ρ (no intercept)
    @test coefnames(m_ord) == ["x1", "x2", "ρ"]
    @test length(m_ord.cutpoints) == 2
    @test m_ord.cutpoints[1] == 0.0          # τ₁ fixed for identification
    @test m_ord.cutpoints[2] > m_ord.cutpoints[1]  # cutpoints must be ordered
    @test size(m_ord.cut_draws) == (600, 2)
    @test coef(m_ord)[1] > 0.0
    @test coef(m_ord)[2] < 0.0
    @test isapprox(m_ord.rho, 0.35; atol=0.3)
    @test 0.15 < m_ord.accept_rate < 0.6

    # ── Input validation ──
    @test_throws AssertionError fit_spatial_probit(W, X, fill(2.0, N))
    @test_throws AssertionError fit_spatial_probit(W, X, y[1:end-1])
    @test_throws AssertionError fit_spatial_ordered_probit(W, X, y_ord; n_categories=1)
end
