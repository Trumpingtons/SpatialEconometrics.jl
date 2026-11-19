@testset "27. Durbin Score Tests (Koley & Bera)" begin
    _progress("27. Durbin Score Tests (Koley & Bera)")
    SD = SpatialEconometrics.SpatialDiagnostics
    chi2sf = SpatialEconometrics.SpatialUtils._chi2_sf

    # ── 27a. shapes, degrees of freedom, and the guards ───────────────
    @testset "27a. structure" begin
        rng = MersenneTwister(2701)
        W = make_rook_weights(7); N = size(W, 1)
        X = hcat(ones(N), randn(rng, N, 2))
        y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

        t = sdm_tests(y, X, W)
        @test t.joint.df == 3          # rho plus two Durbin terms
        @test t.robust_lag.df == 1
        @test t.durbin.df == 2
        @test t.robust_durbin.df == 2
        @test length(t.terms) == 2
        @test all(s -> s.df == 1, t.terms)
        for s in (t.joint, t.robust_lag, t.durbin, t.robust_durbin, t.terms...)
            @test s.stat >= 0
            @test 0 <= s.p_value <= 1
            @test s isa ScoreTestResult
        end
        # the names carry the column the term belongs to
        @test t.terms[1].test_name == "RS theta[x2] | rest"
        @test t.terms[2].test_name == "RS theta[x3] | rest"
        @test occursin("df", sprint(show, t.joint))

        # X with no non-constant column has no W*X_d block to test:
        # W * 1 = 1 for a row-standardised W, so the constant drops out.
        @test_throws ArgumentError lm_test_sdm(y, ones(N, 1), W)
        @test_throws ArgumentError lm_test_durbin(y, ones(N, 1), W)
        @test_throws DimensionMismatch lm_test_sdm(y[1:end-1], X, W)

        # a duplicated covariate makes the Durbin block singular, and the
        # per-term statistics must say so rather than return a ridged
        # number (finding 16).
        Xdup = hcat(ones(N), X[:, 2], X[:, 2])
        @test_throws ArgumentError lm_test_durbin_terms(y, Xdup, W)
    end

    # ── 27b. RS_rho IS the existing LM-lag statistic ──────────────────
    # The two are the same algebra reached by different code paths, so
    # they must agree to machine precision. Anything else means one of
    # them is wrong.
    @testset "27b. reduction to lm_test_lag" begin
        rng = MersenneTwister(2702)
        W = make_rook_weights(7); N = size(W, 1)
        X = hcat(ones(N), randn(rng, N, 2))
        y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

        c = SD._sdm_score_components(y, X, W)
        @test c.d_rho^2 / c.J_rr ≈ lm_test_lag(y, X, W).stat rtol = 1e-14

        # The block inverse identity: the per-term formula applied at the
        # rho position equals the Bera-Yoon robust lag statistic.
        Jinv = inv(Symmetric(c.J)); u = Jinv * c.d
        @test u[1]^2 / Jinv[1, 1] ≈
              lm_test_robust_lag_durbin(y, X, W).stat rtol = 1e-12

        # The theta block carries no determinant term; only rho's does.
        @test c.J_rr > c.Tr
        @test c.J[1, 1] == c.J_rr
        @test c.J[2:end, 2:end] ≈ Matrix(c.J_tt)
    end

    # ── 27c. the analytic scores against ForwardDiff ──────────────────
    # The SDM log-likelihood written here from the model, not from the
    # package source:
    #   y = rho*W*y + X*beta + W*X_d*theta + eps,  eps ~ N(0, sigma2 I)
    @testset "27c. scores against automatic differentiation" begin
        rng = MersenneTwister(2703)
        W = make_rook_weights(7); N = size(W, 1)
        Wden = Matrix(W)
        X = hcat(ones(N), randn(rng, N, 2))
        y = X * [1.0, 0.5, -0.3] .+ randn(rng, N)

        function sdm_loglik(p)
            K = size(X, 2)
            rho = p[1]; theta = p[2:3]; beta = p[4:3+K]; s2 = p[end]
            r = (I - rho * Wden) * y .- X * beta .- (Wden * X[:, 2:3]) * theta
            return -0.5 * N * (log(2pi) + log(s2)) +
                   logabsdet(I - rho * Wden)[1] - dot(r, r) / (2 * s2)
        end

        c = SD._sdm_score_components(y, X, W)
        bhat = X \ y
        e = y .- X * bhat
        p0 = vcat(0.0, 0.0, 0.0, bhat, dot(e, e) / N)
        g = ForwardDiff.gradient(sdm_loglik, p0)

        @test c.d_rho ≈ g[1] rtol = 1e-10
        @test c.d_theta[1] ≈ g[2] rtol = 1e-10
        @test c.d_theta[2] ≈ g[3] rtol = 1e-10
        # the nuisance scores vanish at the OLS null, which is what makes
        # these the right scores to test with
        @test maximum(abs, g[4:6]) < 1e-10
        @test abs(g[end]) < 1e-10
    end

    # ── 27d. the null distribution ────────────────────────────────────
    # The audit's standing gate: a score statistic is believed only once
    # its null distribution has been measured. Each statistic must match
    # its chi-squared reference in mean (df), variance (2df) and size at
    # three levels. Measured over 20000 replications this session:
    #
    #   statistic        df   mean    var    p<.10  p<.05  p<.01
    #   RS joint          3   3.054   6.050  0.103  0.051  0.010
    #   RS* rho | theta   1   0.978   1.953  0.097  0.047  0.010
    #   RS theta          2   2.076   4.075  0.107  0.054  0.009
    #   RS* theta | rho   2   2.065   4.077  0.107  0.052  0.010
    #   RS theta[1]       1   1.022   1.956  0.104  0.052  0.009
    #
    # The suite runs 3000, so the bounds below are about three standard
    # errors wide. They are still far tighter than the 1.3x-17x
    # mis-scalings of findings 13-16.
    @testset "27d. null distribution, 3000 replications" begin
        R = 3000
        rng = MersenneTwister(2704)
        W = make_rook_weights(9); N = size(W, 1)
        X = hcat(ones(N), randn(rng, N, 2))
        beta = [1.0, 0.5, -0.3]

        S = [Float64[] for _ in 1:5]
        for _ in 1:R
            y = X * beta .+ randn(rng, N)
            t = sdm_tests(y, X, W)
            for (i, s) in enumerate((t.joint, t.robust_lag, t.durbin,
                                     t.robust_durbin, t.terms[1]))
                push!(S[i], s.stat)
            end
        end

        for (i, df) in enumerate((3, 1, 2, 2, 1))
            v = S[i]
            @test mean(v) ≈ df atol = 0.25               # E[chi2] = df
            @test var(v) ≈ 2df rtol = 0.20               # Var[chi2] = 2df
            @test 0.075 <= mean(x -> chi2sf(x, df) < 0.10, v) <= 0.130
            @test 0.035 <= mean(x -> chi2sf(x, df) < 0.05, v) <= 0.068
            @test 0.004 <= mean(x -> chi2sf(x, df) < 0.01, v) <= 0.019
        end
    end

    # ── 27e. the robust forms do what they claim ──────────────────────
    # This is the whole point of the Bera-Yoon construction and it is
    # sharp: under a lag with no Durbin terms, the plain RS_theta picks up
    # the lag and over-rejects, while RS*_theta holds its nominal size --
    # and symmetrically for LM-lag against RS*_rho under SLX data.
    # Measured over 2000 replications this session:
    #
    #   data        RS theta  RS* theta | LM-lag  RS* rho
    #   SAR(0.5)      0.161     0.051   |  0.926   0.887
    #   SLX           0.716     0.703   |  0.070   0.052
    #   SDM(0.4)      0.772     0.448   |  0.870   0.651
    @testset "27e. robustness to the other parameter" begin
        R = 1000
        rng = MersenneTwister(2705)
        W = make_rook_weights(9); N = size(W, 1)
        X = hcat(ones(N), randn(rng, N, 2))

        function rejection(dgp)
            out = zeros(Int, 4)
            for _ in 1:R
                y, _ = simulate_spatial(dgp, W, X[:, 2:end]; rng = rng)
                t = sdm_tests(y, X, W)
                t.durbin.p_value        < 0.05 && (out[1] += 1)
                t.robust_durbin.p_value < 0.05 && (out[2] += 1)
                t.lag.p_value           < 0.05 && (out[3] += 1)
                t.robust_lag.p_value    < 0.05 && (out[4] += 1)
            end
            out ./ R
        end

        # a lag, no Durbin terms
        r = rejection(SAR_DGP(0.5, [0.5, -0.3], 1.0))
        @test r[1] > 0.10                  # RS theta is fooled by the lag
        @test 0.03 <= r[2] <= 0.08         # RS* theta is not
        @test r[3] > 0.80 && r[4] > 0.80   # both lag tests have power

        # Durbin terms, no lag
        r = rejection(SLX_DGP([0.5, -0.3], [0.6, -0.5], 1.0))
        @test r[1] > 0.50 && r[2] > 0.50   # both Durbin tests have power
        @test r[3] > r[4]                  # LM-lag is fooled by W*X
        @test 0.03 <= r[4] <= 0.08         # RS* rho is not

        # both present: everything should reject
        r = rejection(SDM_DGP(0.4, [0.5, -0.3], [0.5, -0.4], 1.0))
        @test all(x -> x > 0.30, r)
    end
end
