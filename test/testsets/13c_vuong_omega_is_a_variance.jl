@testset "13c. Vuong: omega is a variance of pointwise ratios" begin
    _progress("13c. Vuong: omega is a variance of pointwise ratios")
    # vuong_test built omega out of the AGGREGATE log-likelihoods, as
    # sqrt(abs(2(ll1-ll2)/n - adj^2)). That is twice a mean difference, not
    # a variance: it came out 0.71x the right size here, giving p = 0.324
    # against a correct 0.482. And because the quantity under the root goes
    # negative when model 1 fits worse, the abs() left the statistic
    # non-antisymmetric in its arguments. Only 0 <= p <= 1 was asserted.
    MS = SpatialEconometrics.SpatialModelSelection
    rng = MersenneTwister(1_300_014)
    L = 12; N = L * L
    W1 = make_rook_weights(L); W2 = make_queen_weights(L)
    W1d = Matrix(W1); W2d = Matrix(W2)
    X = hcat(randn(rng, N), randn(rng, N))
    y = 1.0 .+ X * [2.0, -1.0] .+ 0.6 .* (W1d * X[:, 1]) .+ randn(rng, N)

    # Non-nested: same response, different W. Both are plain OLS, so the
    # per-observation log-density is unambiguous and can be written out.
    m1 = fit_slx(W1, X, y); m2 = fit_slx(W2, X, y)

    function ll_i(m, Wd)
        Xa = hcat(ones(N), X, Wd * X)
        e = y .- Xa * coef(m)[1:size(Xa, 2)]
        s2 = m.sigma2
        -0.5 .* log(2pi * s2) .- (e .^ 2) ./ (2s2)
    end
    l1 = ll_i(m1, W1d); l2 = ll_i(m2, W2d)
    @test isapprox(sum(l1), loglikelihood(m1); rtol = 1e-12)
    @test isapprox(sum(l2), loglikelihood(m2); rtol = 1e-12)
    @test isapprox(MS._pointwise_loglik(m1), l1; rtol = 1e-10)

    # The contributions have to sum to the stored log-likelihood for every
    # model type, Jacobian and all. That is what lets one routine serve
    # them without knowing which spatial structure produced the fit.
    y_lag = (I - 0.5 .* W1d) \ (hcat(ones(N), X) * [1.0, 2.0, -1.0] .+
                                randn(rng, N))
    for m in (fit_spatial(SAR(), W1, X, y_lag; method = ExactLU()),
              fit_spatial(SDM(), W1, X, y_lag; method = ExactLU()),
              fit_spatial_error(W1, X, y_lag; method = ExactLU()),
              fit_sac(W1, W1, X, y_lag; method = ExactLU()),
              fit_sdem(W1, X, y_lag; method = ExactLU()),
              m1)
        @test length(MS._pointwise_loglik(m)) == N
        @test isapprox(sum(MS._pointwise_loglik(m)), loglikelihood(m);
                       rtol = 1e-12)
    end

    k1 = N - dof_residual(m1); k2 = N - dof_residual(m2)
    mi = l1 .- l2
    omega_ref = sqrt(sum(abs2, mi .- mean(mi)) / N)
    z_ref = ((loglikelihood(m1) - loglikelihood(m2)) - (k1 - k2)) /
            (sqrt(N) * omega_ref)

    vt = vuong_test(m1, m2)
    @test isapprox(vt.stat, z_ref; rtol = 1e-10)
    # No literal for vt.stat here. randn(::MersenneTwister, n) is only
    # reproducible within a Julia minor version, so the same seed gives
    # different data on 1.10 and on 1.13 -- X sums to -32.82 against
    # -19.63 -- and this statistic came out 0.114 against 1.877. Both
    # agreed with z_ref to 1e-10, which is the assertion that carries the
    # finding: the statistic is whatever the data say, and what is being
    # pinned is that it equals an independently recomputed reference.
    @test 0 <= vt.p_value <= 1

    # Guard the specific defect rather than only the value, so this stays
    # a regression test under any draw: rebuild the old omega and check it
    # is both wrong and wrong by a margin that moves the p-value.
    adj_old = ((loglikelihood(m1) - loglikelihood(m2)) - (k1 - k2)) / N
    omega_old = sqrt(abs(2.0 * (loglikelihood(m1) - loglikelihood(m2)) / N -
                         adj_old^2) + eps())
    @test !isapprox(omega_old, omega_ref; rtol = 0.05)
    stat_old = sqrt(N) * adj_old / omega_old
    @test !isapprox(stat_old, vt.stat; rtol = 0.05)

    # A Vuong statistic flips sign exactly when its arguments swap. The old
    # one gave +0.9871 against -0.9804.
    vt_rev = vuong_test(m2, m1)
    @test isapprox(vt.stat, -vt_rev.stat; rtol = 1e-12)
    @test isapprox(vt.p_value, vt_rev.p_value; rtol = 1e-12)

    # omega is a standard deviation, so it must not collapse towards zero
    # just because the two fits are close.
    @test omega_ref > 0.1
    @test 0.0 <= vt.p_value <= 1.0
end
