@testset "38. Weak vs strong cross-sectional dependence" begin
    _progress("38. Weak vs strong cross-sectional dependence")
    # Pesaran (2015) and Bailey, Kapetanios & Pesaran (2016) state their
    # results as equations and as thresholds, so most gates here are
    # machine-precision identities or exact integers rather than bands.
    #
    # Every assertion is unconditional. Where a claim is conditional it
    # is asserted as an implication, `@test !guard || claim`, never as a
    # `@test` behind an `if` -- a guarded test that fails its guard does
    # not fail, it silently disappears and the only trace is the
    # assertion count. See the `37e` entry in docs/AUDIT.md.
    CF = SpatialEconometrics.SpatialCommonFactors

    # A panel whose error dependence has exponent alpha: a factor loads
    # on M = N^alpha of the N units, which is both papers' construction.
    function cf_panel(rng, N, T, alpha; loading_mean = 1.0)
        E = randn(rng, N, T)
        if alpha > 0
            M = max(1, round(Int, N^alpha))
            f = randn(rng, T)
            lam = loading_mean .+ 0.2 .* randn(rng, M)
            @views E[1:M, :] .+= lam .* transpose(f)
        end
        return E
    end

    @testset "38a. CD and CD_lm against the printed equations" begin
        # Pesaran (2015) eqs (2), (5), (9), (10), written out literally.
        function by_definition(E)
            N, T = size(E)
            rho = zeros(N, N)
            for i in 1:N, j in 1:N
                num = 0.0
                for t in 1:T; num += E[i, t] * E[j, t]; end
                d1 = 0.0; for t in 1:T; d1 += E[i, t]^2; end
                d2 = 0.0; for t in 1:T; d2 += E[j, t]^2; end
                rho[i, j] = num / (sqrt(d1) * sqrt(d2))
            end
            acc = 0.0; acc_lm = 0.0; acc_abs = 0.0
            for i in 1:(N - 1), j in (i + 1):N
                acc += rho[i, j]
                acc_abs += abs(rho[i, j])
                acc_lm += T * rho[i, j]^2 - 1.0
            end
            np = N * (N - 1) ÷ 2
            rb = acc / np
            return (cd = sqrt(T * N * (N - 1) / 2) * rb,
                    lm = sqrt(1.0 / (N * (N - 1))) * acc_lm,
                    rho_bar = rb, abs_rho_bar = acc_abs / np)
        end

        rng = MersenneTwister(3_801)
        for (N, T) in ((5, 10), (12, 8), (25, 20))
            for alpha in (0.0, 0.9)
                E = cf_panel(rng, N, T, alpha)
                ref = by_definition(E)
                rc = pesaran_cd(vec(E), N, T; variant = :cd)
                rl = pesaran_cd(vec(E), N, T; variant = :lm)
                @test isapprox(rc.stat, ref.cd; rtol = 1e-10, atol = 1e-12)
                @test isapprox(rl.stat, ref.lm; rtol = 1e-10, atol = 1e-12)
                @test isapprox(rc.rho_bar, ref.rho_bar; rtol = 1e-10, atol = 1e-12)
                @test isapprox(rc.abs_rho_bar, ref.abs_rho_bar;
                               rtol = 1e-10, atol = 1e-12)
            end
        end

        # The matrix method is the vector method.
        let E = randn(MersenneTwister(1), 20, 15)
            @test pesaran_cd(vec(E), 20, 15).stat == pesaran_cd(E).stat
        end

        # p-value is two-sided normal, and symmetric in the statistic.
        let E = cf_panel(MersenneTwister(2), 30, 40, 0.9)
            r = pesaran_cd(vec(E), 30, 40)
            @test isapprox(r.p_value,
                           2 * (1 - SpatialEconometrics.SpatialUtils._normcdf(abs(r.stat)));
                           rtol = 1e-12)
            @test 0.0 <= r.p_value <= 1.0
        end
    end

    @testset "38b. the independence benchmark for mean |corr|" begin
        # Under independence |rho_hat| is half-normal with sd 1/sqrt(T),
        # so E|rho_hat| = sqrt(2/(pi T)). The field `abs_rho_bar_null`
        # is that value and the measured average must land on it.
        rng = MersenneTwister(3_802)
        for T in (30, 60, 120)
            N = 40
            acc = 0.0; reps = 60
            for _ in 1:reps
                E = randn(rng, N, T)
                acc += pesaran_cd(vec(E), N, T).abs_rho_bar
            end
            m = acc / reps
            null = sqrt(2 / (pi * T))
            @test isapprox(pesaran_cd(vec(randn(rng, N, T)), N, T).abs_rho_bar_null,
                           null; rtol = 1e-12)
            # 60 reps of an average over 780 pairs: tight.
            @test isapprox(m, null; rtol = 0.05)
        end
    end

    @testset "38c. eq (12): the implicit null is weak dependence" begin
        # The reported threshold is (2 - eps)/4 with eps = ln(T)/ln(N).
        rng = MersenneTwister(3_803)
        for (N, T) in ((100, 50), (50, 50), (200, 20))
            r = pesaran_cd(vec(randn(rng, N, T)), N, T)
            eps_rate = log(T) / log(N)
            @test isapprox(r.implicit_null_alpha, (2 - eps_rate) / 4;
                           rtol = 1e-12)
            # It is a number in (0, 1/2] for any sane panel shape.
            @test 0.0 < r.implicit_null_alpha <= 0.5
        end
    end

    @testset "38d. alpha: the published formulas" begin
        # eqs (13), (14), (15), recomputed directly.
        rng = MersenneTwister(3_804)
        N, T = 80, 60
        X = cf_panel(rng, N, T, 0.95)
        # Replicate the standardisation and eq (13)/(14) by hand.
        M = copy(X)
        for i in 1:N
            mu = mean(@view M[i, :]); s = std(@view M[i, :])
            M[i, :] = (M[i, :] .- mu) ./ s
        end
        xbar_t = vec(mean(M, dims = 1))
        s2 = sum((xbar_t .- mean(xbar_t)) .^ 2) / T
        a_ref = 1 + 0.5 * log(s2) / log(N)

        r = alpha_exponent(vec(X), N, T; bias_correct = false)
        @test isapprox(r.sigma2_xbar, s2; rtol = 1e-10)
        @test isapprox(r.alpha_raw, a_ref; rtol = 1e-10)
        @test isapprox(r.alpha, a_ref; rtol = 1e-10)
        @test r.kappa2 == 1.0

        # kappa2 shifts alpha by exactly (1/2) ln(kappa2)/ln(N).
        for k2 in (0.25, 0.5, 2.0)
            rk = alpha_exponent(vec(X), N, T; bias_correct = false, kappa2 = k2)
            @test isapprox(rk.alpha, a_ref - 0.5 * log(k2) / log(N);
                           rtol = 1e-10)
        end

        # The bias correction only ever subtracts.
        rb = alpha_exponent(vec(X), N, T; bias_correct = true)
        @test rb.alpha <= rb.alpha_raw
        @test isapprox(rb.alpha_raw, a_ref; rtol = 1e-10)
    end

    @testset "38e. alpha: scale invariance, exactly" begin
        rng = MersenneTwister(3_805)
        N, T = 60, 50
        X = cf_panel(rng, N, T, 0.95)
        a1 = alpha_exponent(vec(X), N, T).alpha
        X2 = copy(X)
        for i in 1:N; X2[i, :] .*= (0.1 + 10 * rand(rng)); end
        a2 = alpha_exponent(vec(X2), N, T).alpha
        # Standardisation makes this exact, not approximate.
        @test isapprox(a1, a2; rtol = 1e-12, atol = 1e-12)
        # Shifting a series is absorbed too.
        X3 = copy(X)
        for i in 1:N; X3[i, :] .+= 5 * randn(rng); end
        @test isapprox(alpha_exponent(vec(X3), N, T).alpha, a1;
                       rtol = 1e-12, atol = 1e-12)
        # Without standardisation the scale matters, which is the point
        # of having the option at all.
        b1 = alpha_exponent(vec(X), N, T; standardize = false).alpha
        b2 = alpha_exponent(vec(X2), N, T; standardize = false).alpha
        @test !isapprox(b1, b2; atol = 1e-6)
    end

    @testset "38f. alpha: the two thresholds are enforced" begin
        rng = MersenneTwister(3_806)
        # alpha <= 1/2 is NOT IDENTIFIED: must throw, not return.
        @test_throws ArgumentError alpha_exponent(vec(randn(rng, 80, 100)), 80, 100)
        let e = try
                    alpha_exponent(vec(randn(rng, 80, 100)), 80, 100); nothing
                catch err; err end
            @test e isa ArgumentError
            @test occursin("NOT IDENTIFIED", sprint(showerror, e))
        end
        # A strong factor is identified, and inference is valid there.
        let X = cf_panel(rng, 100, 80, 1.0)
            r = alpha_exponent(vec(X), 100, 80)
            @test r.alpha > 0.5
            # implication, not a guarded test: inference_valid is
            # equivalent to alpha > 4/7 by construction.
            @test r.inference_valid == (r.alpha > 4 / 7)
            @test !r.inference_valid || isfinite(r.se)
            @test !r.inference_valid || (r.ci[1] < r.alpha < r.ci[2])
            @test r.inference_valid || isnan(r.se)
        end
    end

    @testset "38g. alpha: named failures" begin
        rng = MersenneTwister(3_807)
        X = cf_panel(rng, 40, 30, 0.95)
        @test_throws ArgumentError alpha_exponent(vec(X), 2, 30)     # N < 3
        @test_throws ArgumentError alpha_exponent(vec(X[:, 1:1]), 40, 1)  # T < 2
        @test_throws ArgumentError alpha_exponent(vec(X), 40, 30; kappa2 = 0.0)
        @test_throws ArgumentError alpha_exponent(vec(X), 40, 30; kappa2 = -1.0)
        # A constant unit cannot be standardised.
        let Xc = copy(X); Xc[3, :] .= 2.0
            @test_throws ArgumentError alpha_exponent(vec(Xc), 40, 30)
        end
        # non-finite input
        let Xn = copy(X); Xn[1, 1] = NaN
            @test_throws ArgumentError alpha_exponent(vec(Xn), 40, 30)
        end
    end

    @testset "38h. CD: named failures" begin
        rng = MersenneTwister(3_808)
        E = randn(rng, 20, 15)
        @test_throws ArgumentError pesaran_cd(vec(E), 20, 15; variant = :nope)
        @test_throws ArgumentError pesaran_cd(vec(E[1:2, :]), 2, 15)
        @test_throws DimensionMismatch pesaran_cd(vec(E), 20, 14)
        let En = copy(E); En[1, 1] = Inf
            @test_throws ArgumentError pesaran_cd(vec(En), 20, 15)
        end
        # A perfectly fitted unit has no residual variation, so its
        # correlations are undefined and that is said rather than
        # producing a NaN statistic.
        let Ez = copy(E); Ez[4, :] .= 0.0
            @test_throws ArgumentError pesaran_cd(vec(Ez), 20, 15)
        end
    end

    @testset "38i. the verdict ties the two together" begin
        rng = MersenneTwister(3_809)
        # Independent: weak, and alpha unidentified.
        let v = dependence_verdict(vec(randn(rng, 60, 80)), 60, 80)
            @test v.kind === :unidentified
            @test v.alpha === nothing
            @test !isempty(v.recommendation)
        end
        # Strong one-signed factor: strong, alpha near 1.
        #
        # `loading_mean = 2.0` is deliberate and the margin is asserted
        # rather than just the verdict. At `loading_mean = 1.0` the
        # estimate is alpha-hat = 0.9093 (sd 0.0109 over 400 draws),
        # which is 0.85 sd above the 0.9 threshold the `:strong` branch
        # uses -- so 17% of draws come out `:semi_weak` and the test
        # passed or failed by RNG stream. It failed on Julia 1.10.13
        # while passing on 1.13.3, which is how it was found. The cause
        # is not a defect: `kappa2 = 1` biases alpha-hat down by
        # (1/2)ln(kappa2)/ln(N) under standardisation, documented on
        # `alpha_exponent`. At `loading_mean = 2.0` the minimum over
        # 400 draws is 0.9516, so the claim is a property of the design
        # and not of the draw.
        let v = dependence_verdict(
                    vec(cf_panel(rng, 60, 80, 1.0; loading_mean = 2.0)),
                    60, 80)
            @test v.kind === :strong
            @test v.alpha !== nothing
            @test v.alpha.alpha > 0.93
            @test v.cd.p_value < 0.05
        end
        # Cancelling loadings: the configuration where BOTH published
        # diagnostics fail. The verdict must still call it strong, on
        # the absolute average.
        let N = 40, T = 60
            f = randn(rng, T)
            lam = [i <= N ÷ 2 ? 1.0 : -1.0 for i in 1:N]
            E = lam .* transpose(f) .+ 0.5 .* randn(rng, N, T)
            v = dependence_verdict(vec(E), N, T)
            @test v.kind === :strong
            # alpha is refused here -- it is the measured outcome, and
            # it is why the verdict cannot rely on alpha alone.
            @test v.alpha === nothing
            @test v.cd.abs_rho_bar > 2 * v.cd.abs_rho_bar_null
            @test abs(v.cd.rho_bar) < 0.05
            @test occursin("cancel", v.recommendation)
        end
    end

    @testset "38j. spatial CCE" begin
        rng = MersenneTwister(3_810)
        W = cf_grid_W(6, 5); N = size(W, 1); T = 25
        Im = Matrix(1.0I, N, N)
        rho, beta = 0.4, [1.0, -0.5]
        Ainv = inv(Im - rho * Matrix(W))
        X = randn(rng, N * T, 2)
        f = randn(rng, T); lam = 1.0 .+ randn(rng, N)
        y = Vector{Float64}(undef, N * T)
        for t in 1:T
            r = ((t - 1) * N + 1):(t * N)
            y[r] = Ainv * (X[r, :] * beta .+ lam .* f[t] .+ randn(rng, N))
        end

        m = fit_sp_cce(y, X, W, N, T)
        @test m isa SpCCEModel
        @test length(coef(m)) == 3
        @test coefnames(m)[1] == "rho"
        @test length(stderror(m)) == 3
        @test all(isfinite, coef(m))
        @test all(>(0), stderror(m))
        @test nobs(m) == N * T
        @test m.n_augment == 3           # ybar + 2 Xbar columns
        @test m.augment == [:y, :X]
        @test isfinite(m.sigma2) && m.sigma2 > 0
        @test length(residuals(m)) == N * T
        @test isfinite(m.first_stage_F) && m.first_stage_F > 0
        # The estimator exists to recover rho under a factor; on one
        # draw ask only that it is in the right region and far better
        # than the unadjusted fit's known +0.28 bias.
        @test abs(m.rho - rho) < 0.2

        # Augmentation set changes the projected column count.
        let m2 = fit_sp_cce(y, X, W, N, T; augment = [:y])
            @test m2.n_augment == 1
        end
        let m3 = fit_sp_cce(y, X, W, N, T; augment = [:y, :X, :Wy])
            @test m3.n_augment == 4
        end

        # Mean group returns per-unit slopes and a dispersion vcov.
        let mg = fit_sp_cce(y, X, W, N, T; mean_group = true)
            @test mg.coef_mg !== nothing
            @test size(mg.coef_mg, 2) == 2
            @test length(mg.coef_mg_mean) == 2
            @test size(mg.vcov_mg) == (2, 2)
            @test all(>=(0), diag(mg.vcov_mg))
            @test isposdef(mg.vcov_mg + 1e-12I)
        end

        # Named failures.
        @test_throws ArgumentError fit_sp_cce(y, X, W, N, T; augment = Symbol[])
        @test_throws ArgumentError fit_sp_cce(y, X, W, N, T; augment = [:bogus])
        @test_throws DimensionMismatch fit_sp_cce(y, X, W, N, T + 1)
        # Augmentation wider than T is refused rather than fitted on
        # negative degrees of freedom.
        @test_throws ArgumentError fit_sp_cce(y[1:(N * 3)], X[1:(N * 3), :],
                                              W, N, 3;
                                              augment = [:y, :X, :Wy])
        # TimeFE is redundant with the augmentation and says so.
        @test_logs (:warn,) match_mode = :any fit_sp_cce(y, X, W, N, T;
                                                         fe = TimeFE())

        @test occursin("Common Correlated Effects", sprint(show, m))
        @test occursin("First-stage F", sprint(show, m))
    end

    @testset "38k. a time effect is a factor with equal loadings" begin
        # Exact, and the sharpest statement of what this module adds:
        # TimeFE removes a homogeneous-loading factor to machine
        # precision and leaves a heterogeneous one essentially intact.
        SU = SpatialEconometrics.SpatialUtils
        rng = MersenneTwister(3_811)
        N, T = 30, 25
        f = randn(rng, T)
        hom = vec(ones(N) * transpose(f))
        @test maximum(abs, SU.within_transform!(copy(hom), N, T, TimeFE())) < 1e-12

        lam = randn(rng, N)
        het = vec(lam * transpose(f))
        after = SU.within_transform!(copy(het), N, T, TimeFE())
        @test maximum(abs, after) > 0.1 * maximum(abs, het)
        # Almost all of the variance survives.
        @test var(after) / var(het) > 0.9
    end

    @testset "38l. show methods run" begin
        rng = MersenneTwister(3_812)
        E = cf_panel(rng, 50, 60, 1.0)
        @test occursin("Pesaran CD", sprint(show, pesaran_cd(vec(E), 50, 60)))
        @test occursin("Breusch", sprint(show, pesaran_cd(vec(E), 50, 60;
                                                          variant = :lm)))
        @test occursin("Exponent", sprint(show, alpha_exponent(vec(E), 50, 60)))
        @test occursin("Verdict", sprint(show, dependence_verdict(vec(E), 50, 60)))
    end
end
