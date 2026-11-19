@testset "13b. Log-likelihoods against an independent Gaussian density" begin
    _progress("13b. Log-likelihoods against an independent Gaussian density")
    # Every one of these was previously asserted only to be finite. The
    # concentrated likelihood each estimator maximises is rewritten here
    # straight from the density, with the filtered residual formed
    # explicitly, so a wrong constant, a dropped Jacobian or the wrong
    # filter would all show up.
    rng = MersenneTwister(1_300_013)
    L = 9; N = L * L
    W = make_rook_weights(L); Wd = Matrix(W)
    X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
    beta = [1.0, 2.0, -1.0]
    y_lag = (I - 0.5 .* Wd) \ (Xa * beta .+ randn(rng, N))
    y_err = Xa * beta .+ ((I - 0.4 .* Wd) \ randn(rng, N))

    ld(A) = first(logabsdet(A))
    gll(e, s2, jac) = -0.5N * log(2pi) - 0.5N * log(s2) -
                      dot(e, e) / (2s2) + jac
    Xsdm = hcat(ones(N), X, Wd * X)

    m_sar = fit_spatial(SAR(), W, X, y_lag; method = ExactLU())
    e_sar = (I - m_sar.rho .* Wd) * y_lag .- Xa * coef(m_sar)[1:3]
    @test isapprox(loglikelihood(m_sar),
                   gll(e_sar, m_sar.sigma2, ld(I - m_sar.rho .* Wd));
                   rtol = 1e-12)
    @test isapprox(m_sar.sigma2, dot(e_sar, e_sar) / N; rtol = 1e-12)

    m_sdm = fit_spatial(SDM(), W, X, y_lag; method = ExactLU())
    e_sdm = (I - m_sdm.rho .* Wd) * y_lag .- Xsdm * coef(m_sdm)[1:5]
    @test isapprox(loglikelihood(m_sdm),
                   gll(e_sdm, m_sdm.sigma2, ld(I - m_sdm.rho .* Wd));
                   rtol = 1e-12)

    m_sem = fit_spatial_error(W, X, y_err; method = ExactLU())
    e_sem = (I - m_sem.lambda .* Wd) * (y_err .- Xa * coef(m_sem)[1:3])
    @test isapprox(loglikelihood(m_sem),
                   gll(e_sem, m_sem.sigma2, ld(I - m_sem.lambda .* Wd));
                   rtol = 1e-12)

    m_slx2 = fit_slx(W, X, y_lag)
    e_slx2 = y_lag .- Xsdm * coef(m_slx2)[1:5]
    @test isapprox(loglikelihood(m_slx2), gll(e_slx2, m_slx2.sigma2, 0.0);
                   rtol = 1e-12)

    # SAC carries two Jacobians, and its filter is (I-lambda*W2)(I-rho*W1).
    m_sac2 = fit_sac(W, W, X, y_lag; method = ExactLU())
    A = I - m_sac2.rho .* Wd; B = I - m_sac2.lambda .* Wd
    e_sac2 = B * (A * y_lag .- Xa * coef(m_sac2)[1:3])
    @test isapprox(loglikelihood(m_sac2),
                   gll(e_sac2, m_sac2.sigma2, ld(A) + ld(B)); rtol = 1e-12)

    m_sdem2 = fit_sdem(W, X, y_err; method = ExactLU())
    e_sdem2 = (I - m_sdem2.lambda .* Wd) *
              (y_err .- Xsdm * coef(m_sdem2)[1:5])
    @test isapprox(loglikelihood(m_sdem2),
                   gll(e_sdem2, m_sdem2.sigma2,
                       ld(I - m_sdem2.lambda .* Wd)); rtol = 1e-12)

    # residuals(m) must BE the residual that enters the likelihood, and the
    # ML variance must be e-prime-e over n. _pointwise_loglik needs both.
    for (m, jac) in ((m_sar, ld(I - m_sar.rho .* Wd)),
                     (m_sdm, ld(I - m_sdm.rho .* Wd)),
                     (m_sem, ld(I - m_sem.lambda .* Wd)),
                     (m_slx2, 0.0),
                     (m_sac2, ld(A) + ld(B)),
                     (m_sdem2, ld(I - m_sdem2.lambda .* Wd)))
        e = residuals(m)
        @test isapprox(sum(abs2, e) / N, m.sigma2; rtol = 1e-10)
        @test isapprox(loglikelihood(m), gll(e, m.sigma2, jac); rtol = 1e-12)
    end
end
