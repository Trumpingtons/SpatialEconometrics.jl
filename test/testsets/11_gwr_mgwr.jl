@testset "11. GWR & MGWR" begin
    _progress("11. GWR & MGWR")
    rng = MersenneTwister(1111); N = 150
    coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
    X = randn(rng, N, 2)
    # Spatially varying: β1 increases with x-coordinate
    β1 = 1.0 .+ 0.3 .* coords[:, 1]
    β2 = fill(-0.5, N)  # constant
    y = 2.0 .+ β1 .* X[:, 1] .+ β2 .* X[:, 2] .+ 0.5 .* randn(rng, N)

    m_gwr = fit_gwr(X, y, coords; bandwidth=AdaptiveBandwidth(40))
    @test m_gwr.R2 > 0.5
    @test size(m_gwr.coef_local) == (N, 3)  # intercept + 2 vars
    @test length(residuals(m_gwr)) == N
    # Check that local β1 varies more than local β2
    @test std(m_gwr.coef_local[:, 2]) > std(m_gwr.coef_local[:, 3])

    m_mgwr = fit_mgwr(X, y, coords)
    @test m_mgwr.R2 > 0.5
    @test length(m_mgwr.bandwidths) == 3  # intercept + 2 vars
    @test size(m_mgwr.coef_local) == (N, 3)

    # MGWR degrees of freedom, accumulated per Yu et al. (2020). The fit
    # is additive in its components, so S = sum_k R_k and tr(S) must be
    # the sum of the per-covariate traces exactly, not just to tolerance.
    @test length(m_mgwr.enp) == 3
    @test all(isfinite, m_mgwr.enp)
    @test sum(m_mgwr.enp) == m_mgwr.hat_trace
    @test all(>=(1.0), m_mgwr.enp)      # each coefficient costs a parameter
    @test 3.0 <= m_mgwr.hat_trace <= N

    # sigma2 and AICc must now carry that penalty. Before the accumulator
    # existed, sigma2 was RSS/N and AICc was N*log(RSS/N) + N -- the
    # tr(S) = 0 case, which made MGWR beat GWR by exactly the penalty it
    # was missing. The last assertion is the one that catches a relapse.
    let RSS = sum(abs2, residuals(m_mgwr)), t = m_mgwr.hat_trace
        @test isapprox(m_mgwr.sigma2, RSS / (N - t); rtol = 1e-10)
        @test isapprox(m_mgwr.AICc,
                       N * log(RSS / N) + N * (N + t) / (N - 2 - t);
                       rtol = 1e-10)
        @test m_mgwr.AICc > N * log(RSS / N) + N + 1.0
    end

    # beta1 varies with the x-coordinate and beta2 is constant, so the
    # first should buy more effective parameters on a shorter bandwidth.
    # This is the substantive claim MGWR makes, and it reads straight off
    # the accumulator rather than off the coefficient spread.
    @test m_mgwr.enp[2] > m_mgwr.enp[3]
    @test m_mgwr.bandwidths[2] < m_mgwr.bandwidths[3]

    # The standard errors come from the coefficient operator B_k, so they
    # are finite and strictly positive everywhere once it is accumulated.
    @test all(isfinite, m_mgwr.se_local)
    @test all(>(0.0), m_mgwr.se_local)

    # Opting out must change only what is reported, never the fit: the
    # accumulator rides alongside the back-fitting loop and must not
    # perturb it. Everything it identifies is NaN rather than a number
    # that would invite comparison with a GWR AICc.
    m_noinf = fit_mgwr(X, y, coords; inference = false)
    @test m_noinf.coef_local == m_mgwr.coef_local
    @test isnan(m_noinf.hat_trace)
    @test isnan(m_noinf.AICc)
    @test all(isnan, m_noinf.enp)
    @test isapprox(m_noinf.sigma2, sum(abs2, residuals(m_noinf)) / N;
                   rtol = 1e-10)

    # Above max_inference_n the accumulator is skipped with a warning,
    # not an error: K+1 dense N x N operators is a memory ceiling, not a
    # reason to refuse to fit.
    m_gated = @test_logs (:warn, r"degrees-of-freedom accumulator") match_mode=:any fit_mgwr(X, y, coords; max_inference_n = 10)
    @test isnan(m_gated.hat_trace)
    @test m_gated.R2 > 0.5

    # tr(S) is an effective parameter count, so it lives in [K, N].
    # It was once formed with R^{-1} in place of R^{-T}, which pushed it
    # past N on a badly scaled X and then collapsed the N - tr(S) divisor
    # in sigma2 onto its floor of 1, leaving sigma2 equal to RSS.
    @test 3.0 <= m_gwr.hat_trace <= N
    @test isapprox(m_gwr.sigma2,
                   sum(abs2, residuals(m_gwr)) / (N - m_gwr.hat_trace);
                   rtol = 1e-10)

    # As the bandwidth grows every weight tends to one, so GWR collapses
    # onto OLS: tr(S) -> K and the local coefficients become the global
    # ones. This pins the trace to a value known in closed form.
    let Xc = hcat(ones(N), X), b_ols = Xc \ y
        m_wide = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(1.0e4))
        @test isapprox(m_wide.hat_trace, 3.0; atol = 1e-4)
        @test isapprox(vec(mean(m_wide.coef_local, dims = 1)), b_ols; rtol = 1e-6)
        @test all(isapprox.(m_wide.coef_local[1, :], b_ols; rtol = 1e-6))
    end
end
