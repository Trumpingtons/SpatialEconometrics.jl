@testset "41. Semiparametric spatial" begin
    _progress("41. Semiparametric spatial")
    # Shared draw: knn W on random points, the Su & Jin (2010, §6)
    # regressor x = 0.2z² − z + v and m(z) = 2 logistic(z).
    function sp_draw(rng, n, ρ)
        coords = rand(rng, n, 2)
        W = sparse(knn_weights(coords, 6))
        z = randn(rng, n)
        x = 0.2 .* z .^ 2 .- z .+ randn(rng, n)
        y = (I - ρ * W) \ (x .+ 2 .* exp.(z) ./ (1 .+ exp.(z)) .+ randn(rng, n))
        return y, reshape(x, :, 1), z, W
    end

    @testset "41a. the smoother reproduces polynomials of its order" begin
        rng = MersenneTwister(41_001)
        z = randn(rng, 300)
        S1 = local_polynomial_smoother(z, 0.4; order = 1)
        @test maximum(abs.(S1 * ones(300) .- 1)) < 1e-10
        @test maximum(abs.(S1 * z .- z)) < 1e-9
        # Order 1 does not reproduce a quadratic; order 2 does.
        @test maximum(abs.(S1 * z .^ 2 .- z .^ 2)) > 1e-3
        S2 = local_polynomial_smoother(z, 0.6; order = 2)
        @test maximum(abs.(S2 * z .^ 2 .- z .^ 2)) < 1e-8
        Z = randn(rng, 300, 2)
        S3 = local_polynomial_smoother(Z, 0.8)
        @test maximum(abs.(S3 * Z[:, 2] .- Z[:, 2])) < 1e-9
        @test_throws ArgumentError local_polynomial_smoother(z, 0.4; order = 0)
        # A bandwidth far too small for the data widens every window
        # rather than failing, and the result still reproduces lines.
        Sw, nw = SpatialEconometrics.SemiparametricSpatial._smoother(
            reshape(z ./ std(z), :, 1), 1e-4, 1)
        @test nw == 300
        @test maximum(abs.(Sw * z .- z)) < 1e-8
        _, nw0 = SpatialEconometrics.SemiparametricSpatial._smoother(
            reshape(z ./ std(z), :, 1), 2.0, 1)
        @test nw0 == 0
        @test_throws ArgumentError local_polynomial_smoother(randn(rng, 50, 5), 1.0)
    end

    @testset "41b. a flat smoother is fit_spatial with z as a regressor" begin
        # As h → ∞ every kernel weight tends to 1 and the local-linear
        # smoother becomes the projection on [1, z]. The profile QML is
        # then exactly the SAR QML with design [1, x, z].
        rng = MersenneTwister(41_002)
        y, X, z, W = sp_draw(rng, 250, 0.5)
        # The exact claim, with no optimiser in it: the profile likelihood
        # depends on the data only through the first-stage residuals of y
        # and Wy, and with a flat smoother those are (Frisch–Waugh–Lovell)
        # the OLS residuals on [1, x, z] that fit_spatial concentrates on.
        SPm = SpatialEconometrics.SemiparametricSpatial
        Zs, _ = SPm._standardise(reshape(z, :, 1))
        fs = SPm._first_stage(SPm._smoother(Zs, 1e5, 1)[1], Matrix(X), y, W * y)
        D = hcat(ones(length(y)), X, z)
        ols_res(v) = v .- D * (D \ v)
        @test maximum(abs.(fs.e0 .- ols_res(y))) < 1e-8 * norm(y)
        @test maximum(abs.(fs.e1 .- ols_res(W * y))) < 1e-8 * norm(W * y)
        # The two fits then maximise the same function. Each locates ρ̂ by
        # Brent's method only to its tolerance, 1e-7, so the estimates
        # agree to that, not to rounding: 4e-8 apart on Julia 1.10 under
        # Linux's BLAS, where an atol of 1e-8 failed CI on 2026-11-13.
        m = fit_partially_linear_sar(y, X, z, W; bandwidth = 1e5, method = ExactLU())
        f = fit_spatial(SAR(), W, hcat(X, z), y; method = ExactLU())
        @test m.rho ≈ f.rho atol = 1e-6
        @test m.beta[1] ≈ coef(f)[2] atol = 1e-6
        @test stderror(m)[2] ≈ f.rho_se rtol = 1e-6
        @test stderror(m)[1] ≈ stderror(f)[2] rtol = 1e-6
        @test loglikelihood(m) ≈ loglikelihood(f) atol = 1e-6
        @test m.bandwidth_rule === :user
    end

    @testset "41c. the closed-form root, by hand and against Smirnov's linear form" begin
        rng = MersenneTwister(41_003)
        y, X, z, W = sp_draw(rng, 250, 0.5)
        n = length(y)
        Xa = hcat(ones(n), X, z)
        c = closed_form_rho(y, Xa, W)
        H = Xa * (Xa \ Matrix(1.0I, n, n))
        e0 = y .- H * y
        e1 = W * y; e1 = e1 .- H * e1
        e2 = W * (W * y); e2 = e2 .- H * e2
        a = dot(e1, e2); b = dot(e0, e2) + dot(e1, e1); cc = dot(e0, e1)
        @test c.rho ≈ (b - sqrt(b^2 - 4a * cc)) / (2a) rtol = 1e-10
        @test abs(c.a * c.rho^2 - c.b * c.rho + c.c) < 1e-8 * abs(c.b)
        @test c.se_classical ≈ sqrt(c.sigma2 / (b - 2c.rho * a)) rtol = 1e-10
        @test !c.smoothed
        # The flat-smoother model's closed form is the linear one.
        m = fit_partially_linear_sar(y, X, z, W; bandwidth = 1e5, method = ExactLU())
        cm = closed_form_rho(m)
        @test cm.rho ≈ c.rho atol = 1e-8
        @test cm.smoothed
        # tr W ≠ 0 breaks the quasi-score, so it is refused.
        Wd = W + 0.1 * sparse(1.0I, n, n)
        @test_throws ArgumentError closed_form_rho(y, Xa, Wd)
    end

    @testset "41d. spectral overlap: a projection on the modes reads 1, its complement 0" begin
        rng = MersenneTwister(41_004)
        coords = rand(rng, 120, 2)
        W = sparse(knn_weights(coords, 5))
        E = spatial_eigenvectors(W; n_candidates = 25).vectors
        P = E[:, 1:10] * E[:, 1:10]'
        r = spectral_overlap(P, W; k = 25)
        @test r.k == 25
        @test all(abs.(r.pi[1:10] .- 1) .< 1e-10)
        @test all(r.pi[11:25] .< 1e-10)
        @test r.pibar ≈ 10 / 25 atol = 1e-10
        @test spectral_overlap(I - P, W; k = 10).pibar < 1e-10
        @test occursin("no null distribution", sprint(show, r))
    end

    @testset "41e. an intercept in X is refused, not silently absorbed" begin
        rng = MersenneTwister(41_005)
        y, X, z, W = sp_draw(rng, 150, 0.4)
        @test_throws ArgumentError fit_partially_linear_sar(y, hcat(ones(150), X), z, W)
        @test_throws ArgumentError fit_partially_linear_sar(y, X, z, W; bandwidth = :silverman)
    end

    @testset "41f. a flat time smoother is the DirectQML panel with τ interactions" begin
        # As h → ∞ the local-linear fit in τ is the global fit on
        # [X, τX]. With a constant in X and fixed effects, that is the
        # within model on [x, τ, τx]. Measured: ρ to 7e-12, the
        # log-likelihood to 5e-8. The SEs differ (1.1% here, at T = 20)
        # by one convention, checked exactly below.
        rng = MersenneTwister(41_006)
        N, Tp = 40, 20
        W = spzeros(N, N)
        for i in 1:N, k in (-2, -1, 1, 2)
            W[i, mod1(i + k, N)] = 0.25
        end
        τ = (1:Tp) ./ Tp
        x = randn(rng, N, Tp); α = randn(rng, N)
        Y = zeros(N, Tp)
        for t in 1:Tp
            Y[:, t] = (I - 0.4W) \ ((1 + τ[t]) .+ (1 + 2τ[t]^2) .* x[:, t] .+ α .+ randn(rng, N))
        end
        y = vec(Y); X = hcat(ones(N * Tp), vec(x))
        m = fit_tv_coef_sar(y, X, W, N, Tp; bandwidth = 1e4, method = ExactLU())
        tt = vec(repeat(collect(τ)', N))
        f = fit_spatial_panel(SAR(), IndivFE(), W, hcat(vec(x), tt, tt .* vec(x)), y, N, Tp;
                              approach = DirectQML(bias_correct = false), method = ExactLU())
        # Two Brent searches, each to its tolerance 1e-7 (see 41b).
        @test m.rho ≈ f.rho atol = 1e-6
        @test loglikelihood(m) ≈ loglikelihood(f) atol = 1e-5
        # The SEs differ by one convention only. Rebuild LGG's Σ₀ from
        # the fit's own pieces: with the trace terms scaled by T it is
        # rho_se; scaled by T − 1, as the package's panel code does
        # under individual fixed effects, it is that fit's SE.
        SS = SpatialEconometrics.SemiparametricSpatial
        Xs = [X[(t-1)*N+1:t*N, :] for t in 1:Tp]
        Wy = vec(W * Y)
        g = SS._tv_fit(Xs, y, Wy, N, Tp, 1e4, W, ExactLU(), (-0.995, 0.995))
        v = y .- g.ρ̂ .* Wy .- g.D * g.α
        μ = SS._tv_apply(g.Sm, v) .+ g.D * g.α
        u = g.e0 .- g.ρ̂ .* g.e1; σ2 = dot(u, u) / (N * Tp)
        R = vec(W * ((I - g.ρ̂ * W) \ reshape(μ, N, Tp)))
        Rt = R .- SS._tv_apply(g.Sm, R); Rt .-= g.Dt * (g.F \ Rt)
        trG, trG2, trGtG = SpatialEconometrics.SpatialUtils.g_traces(W, g.ρ̂)
        se_at(Te) = sqrt(inv([dot(Rt, Rt) / σ2 + Te * (trG2 + trGtG)  Te * trG / σ2;
                              Te * trG / σ2  N * Te / (2σ2^2)])[1, 1])
        @test m.rho_se ≈ se_at(Tp) rtol = 1e-8
        @test stderror(f)[end] ≈ se_at(Tp - 1) rtol = 1e-8
        @test abs(m.rho_se / stderror(f)[end] - 1) > 1e-3
        # β̂_x(τ) is the global line coef_x + coef_τx τ.
        cf = coef(f)
        @test maximum(abs.(m.beta[:, 2] .- (cf[1] .+ cf[3] .* τ))) < 1e-6
        @test sum(m.alpha) ≈ 0 atol = 1e-8
        @test size(m.beta) == (Tp, 2) && size(m.beta_se) == (Tp, 2)
        @test all(>(0), m.beta_se)
    end

    @testset "41g. accessors, show, and the cross-validated paths" begin
        rng = MersenneTwister(41_007)
        y, X, z, W = sp_draw(rng, 200, 0.5)
        m = fit_partially_linear_sar(y, X, z, W; bandwidth = :lscv, method = ExactLU())
        @test m.bandwidth_rule === :lscv
        @test 0 <= m.n_widened < 200
        @test occursin("widened", sprint(show, m))
        @test nobs(m) == 200
        @test length(coef(m)) == length(coefnames(m)) == 2
        @test coefnames(m) == ["x1", "rho"]
        @test issymmetric(vcov(m))
        @test fitted(m) .+ residuals(m) ≈ y
        # The fitted parts add up: y = ρ̂Wy + Xβ̂ + m̂ + residual.
        @test m.rho .* (W * y) .+ X * m.beta .+ m.m_hat .+ residuals(m) ≈ y
        s = sprint(show, m)
        @test occursin("not a test", s)
        @test occursin("not thereby a causal spillover", s)
        c = closed_form_rho(m)
        @test occursin("no consistency proof", sprint(show, c))
        @test c.se_hc0 > 0 && c.se_shac > 0
    end
end
