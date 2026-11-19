@testset "21c. ESF GLM Families against Independent Densities" begin
    _progress("21c. ESF GLM Families against Independent Densities")
    SF = SpatialEconometrics.SpatialFiltering
    rng = MersenneTwister(606)
    n = 300
    Xg = hcat(ones(n), randn(rng, n), randn(rng, n))
    bg = [0.4, 0.6, -0.3]
    ηg = Xg * bg
    μ_log = exp.(ηg)
    μ_logit = 1 ./ (1 .+ exp.(-ηg))
    α_t = 0.5; r_nb = 1 / α_t

    y_p = Float64[rand(rng, Poisson(m)) for m in μ_log]
    y_b = Float64[rand(rng, Bernoulli(m)) for m in μ_logit]
    y_n = Float64[rand(rng, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                  for m in μ_log]

    # ---- the log-likelihoods are the textbook densities -----------
    @test isapprox(SF._glm_loglik(:poisson, y_p, μ_log, 0.0, 0.0),
                   sum(logpdf(Poisson(μ_log[i]), Int(y_p[i])) for i in 1:n);
                   rtol = 1e-10)
    @test isapprox(SF._glm_loglik(:logit, y_b, μ_logit, 0.0, 0.0),
                   sum(logpdf(Bernoulli(μ_logit[i]), Bool(y_b[i])) for i in 1:n);
                   rtol = 1e-10)
    @test isapprox(SF._glm_loglik(:negbin, y_n, μ_log, α_t, 0.0),
                   sum(logpdf(NegativeBinomial(r_nb, r_nb / (r_nb + μ_log[i])),
                              Int(y_n[i])) for i in 1:n);
                   rtol = 1e-10)
    # the Gaussian branch is the concentrated form, -N/2(log2π + logσ² + 1)
    @test isapprox(SF._glm_loglik(:gaussian, y_p, μ_log, 0.0, 2.0),
                   -0.5 * n * (log(2π) + log(2.0) + 1); rtol = 1e-12)

    # ---- IRLS reaches the actual maximum --------------------------
    # Differentiate an independently written log-likelihood at the IRLS
    # answer: the score must vanish and a Newton step must not move.
    nll_p(v) = -sum(logpdf(Poisson(exp(dot(view(Xg, i, :), v))), Int(y_p[i]))
                    for i in 1:n)
    nll_b(v) = -sum(logpdf(Bernoulli(1 / (1 + exp(-dot(view(Xg, i, :), v)))),
                           Bool(y_b[i])) for i in 1:n)
    nll_n(v) = -sum(logpdf(NegativeBinomial(r_nb,
                        r_nb / (r_nb + exp(dot(view(Xg, i, :), v)))),
                        Int(y_n[i])) for i in 1:n)

    for (fam, yv, nll, al) in ((:poisson, y_p, nll_p, 0.0),
                               (:logit, y_b, nll_b, 0.0),
                               (:negbin, y_n, nll_n, α_t))
        bh, μh, wh, conv = SF._fit_glm(Xg, yv, fam; α = al)
        @test conv
        g = ForwardDiff.gradient(nll, bh)
        @test maximum(abs, g) < 1e-6
        @test norm(ForwardDiff.hessian(nll, bh) \ g) < 1e-8
        # fitted means are the inverse link of Xβ̂
        η̂ = Xg * bh
        exp_μ = fam === :logit ? 1 ./ (1 .+ exp.(-η̂)) : exp.(η̂)
        @test maximum(abs, μh .- exp_μ) < 1e-10
    end

    # The log and logit links are canonical for the Poisson and the
    # Bernoulli, so the IRLS weights reproduce the observed Hessian
    # exactly. The log link is NOT canonical for NB2, so there the
    # working weights give the expected information instead and the two
    # genuinely differ -- which is why vcov uses X′WX and not the Hessian.
    for (fam, yv, nll, al) in ((:poisson, y_p, nll_p, 0.0),
                               (:logit, y_b, nll_b, 0.0))
        bh, _, wh, _ = SF._fit_glm(Xg, yv, fam; α = al)
        H = ForwardDiff.hessian(nll, bh)
        @test maximum(abs, (H .- Xg' * (wh .* Xg)) ./ abs.(H)) < 1e-10
    end
    let
        bh, _, wh, _ = SF._fit_glm(Xg, y_n, :negbin; α = α_t)
        H = ForwardDiff.hessian(nll_n, bh)
        # The claim is the contrast with the 1e-10 above, so the threshold
        # is set orders of magnitude away from both sides rather than at
        # the value one draw produced. Measured across five seeds on two
        # Julia versions the discrepancy runs 0.026 to 0.616, while the
        # canonical links stay near 1e-15; the old 0.1 bound sat inside
        # that spread and failed at 0.076.
        @test maximum(abs, (H .- Xg' * (wh .* Xg)) ./ abs.(H)) > 1e-3
    end

    # ---- Pearson residuals use the right variance function --------
    μv = [0.5, 2.0, 4.0]; yv3 = [1.0, 1.0, 5.0]
    @test SF._pearson_resid(:gaussian, yv3, μv, 0.0) ≈ yv3 .- μv
    @test SF._pearson_resid(:poisson, yv3, μv, 0.0) ≈ (yv3 .- μv) ./ sqrt.(μv)
    @test SF._pearson_resid(:negbin, yv3, μv, 0.5) ≈
          (yv3 .- μv) ./ sqrt.(μv .+ 0.5 .* μv .^ 2)
    let p3 = [0.2, 0.5, 0.9], y3 = [0.0, 1.0, 1.0]
        @test SF._pearson_resid(:logit, y3, p3, 0.0) ≈
              (y3 .- p3) ./ sqrt.(p3 .* (1 .- p3))
    end

    # ---- the dispersion update ------------------------------------
    # For NB2, Var(y) = μ + αμ², so ((y-μ)² - μ)/μ² has mean α. The
    # average is over the residual degrees of freedom, not N: μ̂ is
    # fitted, so dividing by N biases α downwards.
    y_od = [3.0, 6.0, 1.0]          # overdispersed, so α̂ > 0
    raw_terms = ((y_od .- μv) .^ 2 .- μv) ./ μv .^ 2
    @test sum(raw_terms) > 0
    @test isapprox(SF._update_negbin_alpha(y_od, μv),
                   sum(raw_terms) / 3; rtol = 1e-12)
    @test isapprox(SF._update_negbin_alpha(y_od, μv; n_par = 2),
                   sum(raw_terms) / 1; rtol = 1e-12)
    # dividing by the residual df, not N, raises the estimate
    @test SF._update_negbin_alpha(y_od, μv; n_par = 2) >
          SF._update_negbin_alpha(y_od, μv)

    # Underdispersed data makes the moment estimate negative, which is
    # not a dispersion; the floor keeps it positive rather than
    # handing a negative variance to the working weights.
    @test sum(((yv3 .- μv) .^ 2 .- μv) ./ μv .^ 2) < 0
    @test SF._update_negbin_alpha(yv3, μv) > 0
    @test SF._update_negbin_alpha([1.0, 1.0], [1.0, 1.0]) > 0

    # ---- digamma and trigamma against closed forms ---------------
    # ψ(1) = -γ, ψ(1/2) = -γ - 2log2, ψ(n) = -γ + Σ_{k<n} 1/k;
    # ψ₁(1) = π²/6, ψ₁(1/2) = π²/2, ψ₁(n) = π²/6 - Σ_{k<n} 1/k².
    let γ = 0.5772156649015328606
        @test isapprox(SF._digamma(1.0), -γ; rtol = 1e-13)
        @test isapprox(SF._digamma(0.5), -γ - 2log(2); rtol = 1e-13)
        @test isapprox(SF._trigamma(1.0), π^2 / 6; rtol = 1e-12)
        @test isapprox(SF._trigamma(0.5), π^2 / 2; rtol = 1e-12)
        for m in 2:8
            @test isapprox(SF._digamma(Float64(m)),
                           -γ + sum(1 / k for k in 1:(m - 1)); rtol = 1e-13)
            @test isapprox(SF._trigamma(Float64(m)),
                           π^2 / 6 - sum(1 / k^2 for k in 1:(m - 1)); rtol = 1e-12)
        end
        # the recurrences, across the switch from recursion to the series
        for x in (0.01, 0.37, 1.0, 4.5, 9.3, 9.999, 10.0, 25.0, 1e3)
            @test isapprox(SF._digamma(x + 1) - SF._digamma(x), 1 / x; rtol = 1e-11)
            @test isapprox(SF._trigamma(x + 1) - SF._trigamma(x),
                           -1 / x^2; rtol = 1e-10)
        end
    end

    # ---- the dispersion by maximum likelihood --------------------
    # Score for α at fixed μ, written in θ = 1/α, with ∂ℓ/∂α = -θ²s(θ).
    # Differentiating the Distributions log-likelihood in α must agree.
    ll_a(a) = sum(logpdf(NegativeBinomial(1 / a, (1 / a) / ((1 / a) + μ_log[i])),
                         Int(y_n[i])) for i in 1:n)
    for a in (0.2, 0.5, 0.6, 1.3)
        sc, _ = SF._negbin_theta_score(y_n, μ_log, 1 / a)
        @test isapprox(-(1 / a)^2 * sc, ForwardDiff.derivative(ll_a, a);
                       rtol = 1e-10)
    end
    # the reported derivative in θ is the derivative of the score
    for a in (0.3, 0.9)
        θa = 1 / a
        sc, dsc = SF._negbin_theta_score(y_n, μ_log, θa)
        h = 1e-5 * θa
        num = (SF._negbin_theta_score(y_n, μ_log, θa + h)[1] -
               SF._negbin_theta_score(y_n, μ_log, θa - h)[1]) / (2h)
        @test isapprox(dsc, num; rtol = 1e-6)
        @test dsc < 0                      # the score falls in θ, rises in α
    end

    # The MLE is a root of that score, and it is not merely close to the
    # best point of a 2001-point grid over the same range -- it is better.
    α_ml = SF._negbin_alpha_mle(y_n, μ_log)
    @test abs(SF._negbin_theta_score(y_n, μ_log, 1 / α_ml)[1]) < 1e-8
    let grid = range(0.05, 2.0; length = 2001), vals = [ll_a(a) for a in grid]
        @test abs(α_ml - grid[argmax(vals)]) < 2e-3   # grid spacing is 1e-3
        @test ll_a(α_ml) >= maximum(vals) - 1e-9
    end
    # and it beats the moment estimate on its own criterion
    @test ll_a(α_ml) > ll_a(SF._update_negbin_alpha(y_n, μ_log; n_par = 3))
    # starting value must not change the answer
    for a0 in (1e-4, 0.05, 1.0, 20.0)
        @test isapprox(SF._negbin_alpha_mle(y_n, μ_log; α_start = a0), α_ml;
                       rtol = 1e-8)
    end

    # Underdispersed data maximises the likelihood at the Poisson limit,
    # so α̂ sits on the floor rather than going negative.
    @test SF._negbin_alpha_mle(yv3, μv) ≈ 1e-6
    @test SF._negbin_alpha_mle([1.0, 1.0], [1.0, 1.0]) ≈ 1e-6
    # the boundary test is the Poisson overdispersion score ½Σ[(y-μ)²-y]
    @test sum((yv3 .- μv) .^ 2 .- yv3) < 0
    @test sum((y_od .- μv) .^ 2 .- y_od) > 0
    @test SF._negbin_alpha_mle(y_od, μv) > 1e-6

    # α is consistent: the bias shrinks with N. 1500 observations with no
    # spatial structure should land within 10% of the truth.
    function alpha_at(nn, R, seed; mode = :moments)
        rg = MersenneTwister(seed); acc = 0.0
        for _ in 1:R
            Xn = hcat(ones(nn), randn(rg, nn))
            mu = exp.(Xn * [0.4, 0.6])
            yn = Float64[rand(rg, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                         for m in mu]
            a = 0.0
            for _ in 1:30
                _, mh, _, _ = SF._fit_glm(Xn, yn, :negbin; α = a)
                an = mode === :mle ?
                    SF._negbin_alpha_mle(yn, mh; α_start = a) :
                    SF._update_negbin_alpha(yn, mh; n_par = size(Xn, 2))
                abs(an - a) < 1e-10 && (a = an; break)
                a = an
            end
            acc += a
        end
        acc / R
    end
    @test isapprox(alpha_at(1500, 40, 2001), α_t; rtol = 0.10)

    # The MLE is consistent too, and at N = 1500 the two schemes agree:
    # the gap between them is a small-sample effect, not a different
    # estimand.
    @test isapprox(alpha_at(1500, 40, 2001; mode = :mle), α_t; rtol = 0.10)
    @test isapprox(alpha_at(1500, 40, 2001; mode = :mle),
                   alpha_at(1500, 40, 2001); rtol = 0.05)

    # The MLE is the more precise of the two, which is the actual reason
    # to prefer it: on the 60 samples below its root mean squared error
    # around the true α is 0.200 against the moment estimate's 0.249,
    # and over 3000 samples at this N it is 0.199 against 0.285, with a
    # median bias of -8.7% against -16.2%. On the fit_esf coverage
    # design below the moment estimate also lands on the positivity
    # floor for 5.6% of fits against the MLE's 0.7%, and those fits
    # cover 13-19 points worse than the rest.
    function alpha_rmse(nn, R, seed, mode)
        rg = MersenneTwister(seed); acc = 0.0
        for _ in 1:R
            Xn = hcat(ones(nn), randn(rg, nn))
            mu = exp.(Xn * [0.4, 0.6])
            yn = Float64[rand(rg, NegativeBinomial(r_nb, r_nb / (r_nb + m)))
                         for m in mu]
            a = 0.0
            for _ in 1:30
                _, mh, _, _ = SF._fit_glm(Xn, yn, :negbin; α = a)
                an = mode === :mle ?
                    SF._negbin_alpha_mle(yn, mh; α_start = a) :
                    SF._update_negbin_alpha(yn, mh; n_par = size(Xn, 2))
                abs(an - a) < 1e-10 && (a = an; break)
                a = an
            end
            acc += (a - α_t)^2
        end
        sqrt(acc / R)
    end
    @test alpha_rmse(81, 60, 3001, :mle) <
          0.9 * alpha_rmse(81, 60, 3001, :moments)

    # ---- the dispersion= switch inside fit_esf -------------------
    let Wd = make_rook_weights(8), Nd = size(Wd, 1)
        rgd = MersenneTwister(515)
        Xd = randn(rgd, Nd, 2)
        ηd = 0.3 .+ Xd * [0.5, -0.4]
        yd = Float64[rand(rgd, NegativeBinomial(r_nb, r_nb / (r_nb + exp(e))))
                     for e in ηd]
        m_ml = fit_esf(Xd, yd, Wd; family = :negbin, max_eigenvectors = 4)
        m_mm = fit_esf(Xd, yd, Wd; family = :negbin, max_eigenvectors = 4,
                       dispersion = :moments)
        # :mle is the default
        @test m_ml.alpha == fit_esf(Xd, yd, Wd; family = :negbin,
                                    max_eigenvectors = 4,
                                    dispersion = :mle).alpha
        # the two schemes differ, and :mle has the higher likelihood at
        # the dispersion it reports
        @test m_ml.alpha != m_mm.alpha
        s_ml = abs(SF._negbin_theta_score(yd, m_ml.fitted, 1 / m_ml.alpha)[1])
        s_mm = abs(SF._negbin_theta_score(yd, m_mm.fitted, 1 / m_mm.alpha)[1])
        @test s_ml < 1e-4
        @test s_mm > 100 * s_ml
        @test_throws AssertionError fit_esf(Xd, yd, Wd; family = :negbin,
                                            dispersion = :mom)
        # dispersion= is inert for the other three families
        for fam in (:gaussian, :poisson)
            yf = fam === :gaussian ? Xd * [1.0, -0.5] .+ randn(rgd, Nd) :
                 Float64[rand(rgd, Poisson(exp(e))) for e in ηd]
            a = fit_esf(Xd, yf, Wd; family = fam, max_eigenvectors = 3)
            b = fit_esf(Xd, yf, Wd; family = fam, max_eigenvectors = 3,
                        dispersion = :moments)
            @test coef(a) == coef(b)
            @test a.alpha == b.alpha == 0.0
        end
    end

    # ---- coverage for the canonical families ----------------------
    # Poisson and logit report the inverse expected information, which is
    # the right variance for a canonical link; NB2 runs mildly low.
    function esf_coverage(family, R, seed; dispersion = :mle)
        Wl = make_rook_weights(9); Nl = size(Wl, 1)
        eb = spatial_eigenvectors(Wl; min_moran_ratio = 0.25)
        ps = findall(>(1e-10), eb.values)
        Esp = eb.vectors[:, ps[1:min(6, length(ps))]]
        rg = MersenneTwister(seed); hit = 0
        for _ in 1:R
            Xr = randn(rg, Nl, 2)
            et = 0.4 .+ Xr * [0.6, -0.3] .+ 0.8 .* (Esp * randn(rg, size(Esp, 2)))
            yr = if family === :poisson
                Float64[rand(rg, Poisson(exp(clamp(e, -20, 5)))) for e in et]
            elseif family === :negbin
                Float64[rand(rg, NegativeBinomial(r_nb,
                    r_nb / (r_nb + exp(clamp(e, -20, 5))))) for e in et]
            else
                Float64[rand(rg, Bernoulli(1 / (1 + exp(-e)))) for e in et]
            end
            mm = fit_esf(Xr, yr, Wl; family = family, eigen_basis = eb,
                         max_eigenvectors = 10, dispersion = dispersion)
            abs(coef(mm)[2] - 0.6) <= 1.96 * stderror(mm)[2] && (hit += 1)
        end
        hit / R
    end
    @test esf_coverage(:poisson, 120, 1234) > 0.87
    @test esf_coverage(:logit, 120, 1234) > 0.87
    # The negative binomial reaches the canonical families' coverage once
    # the dispersion is its own MLE. Over 6000 replications it runs at
    # 93.3% against the Poisson's 93.1% on this design; with the moment
    # estimate it was 92.2%. Neither reaches 95%: what is left is the
    # post-selection effect that the fit_esf docstring documents, not the
    # dispersion.
    @test esf_coverage(:negbin, 120, 1234) > 0.87
end
