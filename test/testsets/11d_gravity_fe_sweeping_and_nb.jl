@testset "11d. Gravity: FE sweeping and NB dispersion" begin
    _progress("11d. Gravity: FE sweeping and NB dispersion")
    rng = MersenneTwister(5150)
    No, Nd = 12, 10; N = No * Nd
    o = repeat(1:No, inner = Nd)
    d = repeat(1:Nd, outer = No)
    ao = randn(rng, No) .* 0.5
    ad = randn(rng, Nd) .* 0.4
    # log_distance deliberately picks up the origin effect, so the
    # regressors are NOT orthogonal to the fixed effects. That is what
    # exposes a sweep applied to y but not to X: with independent
    # regressors the omission is almost invisible.
    logd = rand(rng, N) .* 2 .+ 1 .+ 1.5 .* [ao[o[n]] for n in 1:N]
    lang = Float64.(rand(rng, N) .< 0.3)
    Xg = hcat(logd, lang)
    eta = Xg * [-0.8, 0.5] .+ 2.0
    for n in 1:N; eta[n] += ao[o[n]] + ad[d[n]]; end
    mu_true = exp.(eta)
    yg = Float64[max(0.0, round(mu_true[n] + sqrt(mu_true[n]) *
                                randn(rng))) for n in 1:N]

    # A two-way within PPML written out independently, as the reference.
    # The point of the rewrite is that the response and every regressor
    # pass through the SAME projection; the previous code swept y with a
    # single pass of independent origin and destination means (not the
    # two-way projection, and double-counting the overall level) while
    # sweeping X by alternating projection, and diverged to ~1e50.
    function dm2!(z, o, d, No, Nd, w)
        for _ in 1:500
            ch = 0.0
            for (ids, Ng) in ((o, No), (d, Nd))
                num = zeros(Ng); den = zeros(Ng)
                for n in eachindex(z)
                    num[ids[n]] += w[n] * z[n]; den[ids[n]] += w[n]
                end
                for g in 1:Ng; den[g] > 0 && (num[g] /= den[g]); end
                for n in eachindex(z)
                    z[n] -= num[ids[n]]
                    ch = max(ch, abs(num[ids[n]]))
                end
            end
            ch < 1e-13 && break
        end
        z
    end
    function ppml_ref(y, X, o, d, No, Nd)
        Nn, Kk = size(X)
        b = zeros(Kk); mu = fill(max(mean(y), 0.1), Nn)
        for _ in 1:300
            bo = copy(b)
            et = log.(max.(mu, 1e-12))
            z = et .+ (y .- mu) ./ max.(mu, 1e-12)
            w = max.(mu, 1e-12)
            zd = dm2!(copy(z), o, d, No, Nd, w)
            Xd = reduce(hcat, [dm2!(copy(X[:, k]), o, d, No, Nd, w)
                               for k in 1:Kk])
            Ws = Diagonal(sqrt.(w))
            b = (Ws * Xd) \ (Ws * zd)
            r = z .- X * b
            fo = zeros(No); fd = zeros(Nd)
            for _ in 1:500
                ch = 0.0
                for (ids, Ng, f) in ((o, No, fo), (d, Nd, fd))
                    num = zeros(Ng); den = zeros(Ng)
                    for n in 1:Nn
                        e = r[n] - fo[o[n]] - fd[d[n]]
                        num[ids[n]] += w[n] * e; den[ids[n]] += w[n]
                    end
                    for g in 1:Ng
                        den[g] > 0 && (f[g] += num[g] / den[g];
                                       ch = max(ch, abs(num[g] / den[g])))
                    end
                end
                ch < 1e-13 && break
            end
            en = X * b
            for n in 1:Nn; en[n] += fo[o[n]] + fd[d[n]]; end
            mu = exp.(clamp.(en, -30, 30))
            maximum(abs, b .- bo) < 1e-10 && break
        end
        b
    end

    m_bi = fit_gravity(PoissonGravity(), yg, Xg, o, d; fe = BilateralFE(),
                       maxiter = 300, tol = 1e-10)
    @test m_bi.converged
    @test m_bi.iterations < 50          # it used to exhaust 200 and diverge
    @test all(isfinite, m_bi.coef)
    @test isapprox(m_bi.coef, ppml_ref(yg, Xg, o, d, No, Nd); rtol = 1e-6)

    # All three sweeps must converge, and the one-way ones must now
    # demean the regressors too.
    for f in (OriginFE(), DestinationFE())
        m = fit_gravity(PoissonGravity(), yg, Xg, o, d; fe = f,
                        maxiter = 300, tol = 1e-10)
        @test m.converged
        @test all(isfinite, m.coef)
    end

    # The recovered effects must reproduce the fit exactly: that is the
    # check that _solve_fe! inverts the sweep. Independent one-pass means
    # per side do not, because both are measured from the same residual
    # and their sum counts the level twice.
    mr = multilateral_resistance(m_bi)
    eta_hat = Xg * m_bi.coef .+
              [mr.outward[o[n]] + mr.inward[d[n]] for n in 1:N]
    @test isapprox(exp.(eta_hat), m_bi.fitted; rtol = 1e-8)

    # A regressor with no within variation is unidentified, and saying so
    # beats returning whatever the least squares happens to pick.
    X_absorbed = hcat(Xg[:, 1], Float64[o[n] for n in 1:N])
    @test_throws ArgumentError fit_gravity(PoissonGravity(), yg, X_absorbed,
                                           o, d; fe = OriginFE())

    # The negative binomial has to differ from Poisson. It used to compute
    # the NB weights, discard them, refit the identical Poisson model each
    # pass and report Poisson's vcov -- so its coefficients were Poisson's
    # and its standard errors were too.
    m_p = fit_gravity(PoissonGravity(), yg, Xg, o, d; maxiter = 300, tol = 1e-10)
    m_nb = fit_gravity(NegBinGravity(), yg, Xg, o, d; maxiter = 300, tol = 1e-10)
    @test m_nb.converged
    @test m_nb.alpha > 0.0
    @test isfinite(m_nb.alpha)
    @test m_nb.vcov != m_p.vcov
    # Overdispersion is what Poisson understates, so NB errors are wider.
    @test all(diag(m_nb.vcov) .>= diag(m_p.vcov) .- 1e-12)
    # PPML stays consistent under overdispersion, so the coefficients
    # should agree even though the weights and errors do not.
    @test isapprox(m_nb.coef, m_p.coef; rtol = 0.25)

    # alpha = 0 through the shared core must be Poisson exactly.
    @test isapprox(m_p.coef,
                   fit_gravity(PoissonGravity(), yg, Xg, o, d;
                               fe = BilateralFE(), maxiter = 300,
                               tol = 1e-10).coef; rtol = 1e-12)

    # Deviance is now 2*sum[y log(y/mu) - (y - mu)], not 2*(ll - null_ll).
    dev_cf = 2 * sum(yg[n] > 0 ?
                     yg[n] * log(yg[n] / m_p.fitted[n]) -
                     (yg[n] - m_p.fitted[n]) : m_p.fitted[n]
                     for n in 1:N)
    @test isapprox(m_p.deviance, dev_cf; rtol = 1e-10)
end
