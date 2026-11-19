@testset "36. Spatial shift-share diagnostics" begin
    _progress("36. Spatial shift-share diagnostics")
    # Three of the four quantities in this module are exact algebraic
    # identities published as such -- the Borusyak-Hull-Jaravel
    # shock-level equivalence, the Goldsmith-Pinkham-Sorkin-Swift
    # decomposition, and the reduction of the Adao-Kolesar-Morales
    # variance to a cluster-robust one under full specialisation. Those
    # are checked against independent computation at machine precision,
    # because an identity that holds to 1e-13 and one that holds to
    # 1e-3 are different claims and only the first is the published
    # one. The two test statistics owe a null-distribution Monte Carlo
    # under the standing rule, and their measured tables live in
    # docs/AUDIT.md; what is asserted here is the loose bound that
    # would catch a regression.
    SS = SpatialEconometrics.SpatialShiftShare

    # Shares with a tunable concentration across shocks. Rows sum to
    # one, which is the complete-shares case.
    function ss_shares(rng, N, K; concentration = 0.0)
        a = exp.(concentration .* randn(rng, K))
        W = zeros(N, K)
        for i in 1:N
            row = a .* (0.2 .+ rand(rng, K))
            W[i, :] = row ./ sum(row)
        end
        return W
    end

    # Residuals carrying a sector-mix component: regions with similar
    # industry mixes are correlated regardless of geography, which is
    # the dependence the AKM standard error exists for. `mix` is the
    # ratio of sectoral to idiosyncratic standard deviation; the
    # sqrt(K) rescaling keeps that ratio fixed as K grows, since W*eta
    # has standard deviation about sqrt(1/K) when rows sum to one.
    function ss_resid(rng, W; mix = 3.0, nu = 1.0)
        N, K = size(W)
        return W * (randn(rng, K) .* (mix * sqrt(K))) .+ randn(rng, N) .* nu
    end

    @testset "36a. design validation" begin
        rng = MersenneTwister(4601)
        W = ss_shares(rng, 40, 6)
        g = randn(rng, 6)
        d = shift_share_design(W, g)

        @test size(d.W) == (40, 6)
        @test d.X ≈ W * g
        @test d.complete_shares
        @test d.s_bar ≈ vec(sum(W; dims = 1)) ./ 40
        # An intercept is always present, and nothing else by default.
        @test size(d.Z) == (40, 1)
        @test all(==(1.0), d.Z)
        # Controls are appended after the intercept.
        C = randn(rng, 40, 2)
        dc = shift_share_design(W, g; controls = C)
        @test size(dc.Z) == (40, 3)
        @test dc.Z[:, 2:3] ≈ C

        # Each refusal is by name rather than a downstream failure.
        @test_throws DimensionMismatch shift_share_design(W, randn(rng, 5))
        @test_throws ArgumentError shift_share_design(W[:, 1:1], g[1:1])
        @test_throws ArgumentError shift_share_design(fill(NaN, 40, 6), g)
        @test_throws ArgumentError shift_share_design(W, fill(Inf, 6))
        # An all-zero share column would divide by zero on aggregation.
        Wz = copy(W); Wz[:, 3] .= 0.0
        @test_throws ArgumentError shift_share_design(Wz, g)
        # Collinear controls are caught by the shared rank check.
        @test_throws ArgumentError shift_share_design(W, g;
            controls = hcat(C[:, 1], C[:, 1]))

        # Incomplete shares warn and report, rather than silently
        # adding the exposure sum as a control.
        Wi = W .* 0.5; Wi[1, :] .*= 0.5
        di = @test_logs (:warn, r"do not sum to a constant") match_mode = :any begin
            shift_share_design(Wi, g)
        end
        @test !di.complete_shares
        @test di.exposure_sum[1] < di.exposure_sum[2]
    end

    @testset "36b. shock-level equivalence, exactly" begin
        # Borusyak, Hull & Jaravel (2022) eq. 4: the regional estimate
        # equals an exposure-weighted estimate formed from K
        # shock-level observations. This is an identity, so the gate is
        # machine precision, and it is checked with controls present
        # since the residualisation is where it could go wrong.
        worst_direct = 0.0; worst_iv = 0.0; worst_fn6 = 0.0
        for rep in 1:60
            rng = MersenneTwister(4700 + rep)
            N = rand(rng, 40:90); K = rand(rng, 5:15)
            W = ss_shares(rng, N, K)
            g = randn(rng, K)
            d = shift_share_design(W, g; controls = randn(rng, N, 2))
            y = randn(rng, N); t = d.X .* 0.7 .+ randn(rng, N)

            f = fit_shift_share(y, d)
            a = shock_aggregate(d, y)
            worst_direct = max(worst_direct,
                abs(a.beta_equiv - f.beta) / max(1e-12, abs(f.beta)))
            @test !a.is_iv

            fiv = fit_shift_share(y, d; treatment = t)
            aiv = shock_aggregate(d, y; treatment = t)
            worst_iv = max(worst_iv,
                abs(aiv.beta_equiv - fiv.beta) / max(1e-12, abs(fiv.beta)))
            @test aiv.is_iv
            # Footnote 6: the exposure-weighted aggregates of a
            # residual vector have weighted mean exactly zero.
            worst_fn6 = max(worst_fn6, abs(dot(a.s_bar, a.y_bar)))
        end
        @test worst_direct < 1e-10
        @test worst_iv < 1e-10
        @test worst_fn6 < 1e-12
        # Measured: 2.29e-13, 2.26e-13 and 2.04e-16.
        @test worst_direct < 1e-11
        @test worst_fn6 < 1e-14
    end

    @testset "36c. Rotemberg decomposition, exactly" begin
        # Goldsmith-Pinkham, Sorkin & Swift (2020) Proposition 3. Both
        # the unit sum of the weights and the decomposition itself are
        # identities.
        worst_sum = 0.0; worst_dec = 0.0; saw_negative = false
        for rep in 1:60
            rng = MersenneTwister(4800 + rep)
            N = rand(rng, 40:90); K = rand(rng, 5:15)
            W = ss_shares(rng, N, K)
            d = shift_share_design(W, randn(rng, K);
                                   controls = randn(rng, N, 2))
            y = randn(rng, N); t = d.X .* 0.7 .+ randn(rng, N)

            r = rotemberg_weights(y, t, d)
            f = fit_shift_share(y, d; treatment = t)
            worst_sum = max(worst_sum, abs(r.alpha_sum - 1))
            worst_dec = max(worst_dec,
                abs(r.beta_overall - f.beta) / max(1e-12, abs(f.beta)))
            r.n_negative > 0 && (saw_negative = true)
            @test length(r.alpha) == K
            @test r.order[1] == argmax(abs.(r.alpha))
        end
        @test worst_sum < 1e-10
        @test worst_dec < 1e-10
        # Measured: 1.42e-14 and 2.64e-13.
        @test worst_sum < 1e-12
        # Negative weights are expected, not a defect: the paper says
        # so. If they never appeared the test would be checking a
        # design too tame to be representative.
        @test saw_negative
    end

    @testset "36d. AKM reduces to clustering, exactly" begin
        # Under full specialisation -- every region in exactly one
        # sector -- Adao, Kolesar & Morales note that their variance
        # becomes the cluster-robust variance clustering on that
        # sector. Checked against the package's own vcov_cluster with
        # the degrees-of-freedom correction off, so the reference is
        # independent code rather than a rearrangement of the same
        # expression.
        worst = 0.0
        for rep in 1:40
            rng = MersenneTwister(4900 + rep)
            N = rand(rng, 40:80); K = rand(rng, 4:8)
            assign = rand(rng, 1:K, N)
            for k in 1:K; assign[k] = k; end   # every sector used
            W = zeros(N, K)
            for i in 1:N; W[i, assign[i]] = 1.0; end
            d = shift_share_design(W, randn(rng, K))
            y = d.X .* 1.5 .+ randn(rng, N)

            f = fit_shift_share(y, d; vcov = :akm)
            spec = SpatialEconometrics.ClusterRobustVCOV(assign;
                                                         dof_correction = false)
            Vc = SpatialEconometrics.vcov_cluster(
                reshape(f.X_tilde, :, 1), residuals(f), spec)
            worst = max(worst, abs(f.se^2 - Vc[1, 1]) / max(1e-300, Vc[1, 1]))
        end
        @test worst < 1e-10
        # Measured: 2.97e-15.
        @test worst < 1e-12
    end

    @testset "36e. the shock-level projection" begin
        # Step 2 of their Remark 5 computes X_ring by regressing the
        # residualised regressor on the share matrix. It is NOT the
        # shock vector g: the intercept in Z demeans the regressor
        # first. Pinning the relationship is what proves the step is
        # implemented as published rather than shortcut to g.
        worst_ne = 0.0; worst_sq = 0.0; worst_c = 0.0
        for rep in 1:40
            rng = MersenneTwister(5000 + rep)
            K = rand(rng, 3:8)
            g = randn(rng, K)

            # Square and invertible: the projection is exact.
            Wsq = rand(rng, K, K) .+ 0.5
            Wsq ./= sum(Wsq; dims = 2)
            dsq = shift_share_design(Wsq, g)
            fsq = fit_shift_share(randn(rng, K), dsq)
            worst_sq = max(worst_sq,
                maximum(abs.(Wsq * fsq.X_ring .- fsq.X_tilde)) /
                max(1e-12, maximum(abs, fsq.X_tilde)))

            # Tall, with shares summing to c. Then W*(1/c) = 1, so the
            # projection is g shifted by the regressor's mean over c.
            N = rand(rng, 30:70)
            c = rand(rng) * 2 + 0.5
            W = ss_shares(rng, N, K) .* c
            d = shift_share_design(W, g)
            f = fit_shift_share(randn(rng, N), d)
            ne = W' * (W * f.X_ring .- f.X_tilde)
            worst_ne = max(worst_ne, maximum(abs, ne) /
                max(1e-12, maximum(abs, W' * f.X_tilde)))
            pred = g .- mean(d.X) / c
            worst_c = max(worst_c,
                maximum(abs.(f.X_ring .- pred)) / maximum(abs, pred))
        end
        @test worst_ne < 1e-10
        @test worst_sq < 1e-10
        @test worst_c < 1e-10
        # Measured: 2.99e-15, 2.28e-15, 1.85e-15.
        @test worst_c < 1e-12

        # More shocks than regions leaves the projection
        # underdetermined, and more than one shock with the same
        # exposure pattern leaves it non-unique. Both are refused by
        # name rather than returning a least-norm vector.
        rng = MersenneTwister(5101)
        Wwide = ss_shares(rng, 6, 10)
        dwide = shift_share_design(Wwide, randn(rng, 10))
        @test_throws ArgumentError fit_shift_share(randn(rng, 6), dwide)
        # Renormalise after duplicating the column: overwriting it
        # changes each row's sum, which would trip the incomplete-shares
        # warning for an unrelated reason. Dividing both copies by the
        # same row sum leaves them identical, so the rank guard is still
        # what fires.
        Wdup = ss_shares(rng, 40, 5); Wdup[:, 2] = Wdup[:, 1]
        Wdup ./= sum(Wdup; dims = 2)
        @test Wdup[:, 1] == Wdup[:, 2]
        ddup = shift_share_design(Wdup, randn(rng, 5))
        @test_throws ArgumentError fit_shift_share(randn(rng, 40), ddup)
    end

    @testset "36f. exposure concentration" begin
        # Uniform exposure gives exactly K effective shocks, up to the
        # representability of 1/K.
        for K in (4, 10, 50)
            d = shift_share_design(fill(1.0 / K, 30, K),
                                   randn(MersenneTwister(K), K))
            c = exposure_concentration(d)
            @test c.n_shocks == K
            @test isapprox(c.effective_shocks, K; rtol = 1e-12)
            @test isapprox(c.hhi, 1 / K; rtol = 1e-12)
            @test isapprox(c.max_share, 1 / K; rtol = 1e-12)
        end

        # Concentrating exposure on one shock drives the effective
        # count monotonically toward one.
        rng = MersenneTwister(5201)
        effs = Float64[]
        for frac in (0.2, 0.5, 0.9, 0.99)
            K = 20
            W = fill((1 - frac) / (K - 1), 50, K); W[:, 1] .= frac
            c = exposure_concentration(shift_share_design(W, randn(rng, K)))
            push!(effs, c.effective_shocks)
            @test isapprox(c.max_share, frac; rtol = 1e-10)
            @test c.order[1] == 1
        end
        @test issorted(effs; rev = true)
        @test effs[end] < 1.1
        @test effs[1] > 5

        # The normalisation makes the measure scale-free: multiplying
        # every share by a constant cannot change concentration.
        W = ss_shares(MersenneTwister(5202), 40, 8)
        c1 = exposure_concentration(shift_share_design(W, ones(8)))
        c2 = exposure_concentration(shift_share_design(W .* 3.0, ones(8)))
        @test isapprox(c1.hhi, c2.hhi; rtol = 1e-12)
    end

    @testset "36g. balance test mechanics" begin
        rng = MersenneTwister(5301)
        N = 80; K = 14
        W = ss_shares(rng, N, K)
        d = shift_share_design(W, randn(rng, K))

        # One characteristic: the joint Wald statistic is the squared
        # robust t exactly, which is the internal consistency check on
        # the cross-equation covariance.
        worst = 0.0
        for rep in 1:40
            rngr = MersenneTwister(5400 + rep)
            dd = shift_share_design(W, randn(rngr, K))
            b = balance_test(dd, randn(rngr, N))
            worst = max(worst, abs(b.joint_chi2 - b.t_stats[1]^2) /
                               max(1e-12, b.t_stats[1]^2))
            @test b.joint_df == 1
            @test b.n_shocks == K
            @test 0.0 <= b.p_values[1] <= 1.0
        end
        @test worst < 1e-10
        # Measured: 1.09e-15.
        @test worst < 1e-12

        # Several characteristics at once, with names carried through.
        R = randn(rng, N, 3)
        b3 = balance_test(d, R; names = ["pretrend", "income", "share65"])
        @test length(b3.coefs) == 3
        @test b3.names == ["pretrend", "income", "share65"]
        @test b3.joint_df == 3
        @test all(p -> 0.0 <= p <= 1.0, b3.p_values)
        @test 0.0 <= b3.joint_p <= 1.0

        # A characteristic built to correlate with the shocks must be
        # detected: the test has power, not merely size.
        dpow = shift_share_design(W, randn(MersenneTwister(5501), K))
        rpow = dpow.X .* 2.0 .+ randn(MersenneTwister(5502), N) .* 0.1
        bpow = balance_test(dpow, rpow)
        @test bpow.p_values[1] < 0.01

        # Shock-level controls take one row per shock, and the wrong
        # orientation is refused by name.
        bq = balance_test(d, randn(rng, N); controls = randn(rng, K, 2))
        @test bq.n_shocks == K
        @test_throws DimensionMismatch balance_test(d, randn(rng, N);
                                                    controls = randn(rng, N, 2))
        @test_throws DimensionMismatch balance_test(d, randn(rng, N + 1))
        @test_throws DimensionMismatch balance_test(d, randn(rng, N);
                                                    names = ["a", "b"])
    end

    @testset "36h. chi-square tail against known critical values" begin
        # The upper tail comes from the shared `SpatialUtils._chi2_sf`
        # rather than from SpecialFunctions, so it is checked against the
        # published 5% critical values. This module briefly carried its
        # own closed-form copy, which leaned on `_normcdf` and so was
        # accurate only to ~1.5e-7 on odd df; the shared version is good
        # to 6e-9 there, which is why the tolerance below can be tight.
        for (x, k) in ((3.841459, 1), (5.991465, 2), (7.814728, 3),
                       (9.487729, 4), (11.070498, 5), (18.307038, 10),
                       (31.410433, 20))
            @test isapprox(SS._chi2_sf(x, k), 0.05; atol = 1e-7)
        end
        @test SS._chi2_sf(0.0, 3) == 1.0
        @test SS._chi2_sf(-1.0, 3) == 1.0
        @test SS._chi2_sf(1e4, 5) < 1e-10
    end

    @testset "36i. AKM against HC1 when the dependence is sectoral" begin
        # The reason this module exists. With a sector-mix component in
        # the residuals, heteroskedasticity-robust standard errors are
        # badly undersized and the AKM standard error is not. Measured
        # tables are in docs/AUDIT.md; the assertions here are the
        # loose bounds that would catch a regression without pinning a
        # Monte Carlo rate.
        rng = MersenneTwister(5601)
        N = 300; K = 100
        W = ss_shares(rng, N, K)
        eps = ss_resid(rng, W; mix = 3.0)
        groups = [argmax(@view W[i, :]) for i in 1:N]
        beta = 1.0

        reps = 150
        cov_akm = 0; cov_hc1 = 0; cov_cl = 0
        for _ in 1:reps
            g = randn(rng, K)
            d = shift_share_design(W, g)
            y = d.X .* beta .+ eps
            fa = fit_shift_share(y, d; vcov = :akm)
            fh = fit_shift_share(y, d; vcov = :hc1)
            spec = SpatialEconometrics.ClusterRobustVCOV(groups)
            secl = sqrt(max(0.0, SpatialEconometrics.vcov_cluster(
                reshape(fa.X_tilde, :, 1), residuals(fa), spec)[1, 1]))
            # The point estimate does not depend on the covariance.
            @test fa.beta == fh.beta
            abs(fa.beta - beta) <= 1.96 * fa.se && (cov_akm += 1)
            abs(fa.beta - beta) <= 1.96 * fh.se && (cov_hc1 += 1)
            abs(fa.beta - beta) <= 1.96 * secl && (cov_cl += 1)
        end
        # Measured over 2000 reps: AKM 0.936, HC1 0.732, cluster 0.720.
        @test cov_akm / reps > 0.85
        @test cov_hc1 / reps < 0.88
        @test cov_akm > cov_hc1
        # Clustering on the dominant sector is not a substitute for the
        # AKM correction; it tracks HC1 rather than fixing it.
        @test cov_cl / reps < 0.88

        # The AKM standard error is larger than HC1 here, which is the
        # whole content of the correction on this design.
        g = randn(rng, K)
        d = shift_share_design(W, g)
        y = d.X .* beta .+ eps
        @test fit_shift_share(y, d; vcov = :akm).se >
              fit_shift_share(y, d; vcov = :hc1).se

        # Interface: a 1x1 covariance, consistent accessors, and no
        # pretence of more coefficients than there are.
        f = fit_shift_share(y, d)
        @test size(vcov(f)) == (1, 1)
        @test coef(f) == [f.beta]
        @test stderror(f) == [f.se]
        @test vcov(f)[1, 1] ≈ f.se^2
        @test nobs(f) == N
        @test length(coefnames(f)) == 1
        @test length(residuals(f)) == N
        @test dof_residual(f) == N - size(d.Z, 2) - 1
        @test_throws ArgumentError fit_shift_share(y, d; vcov = :bogus)
        @test_throws DimensionMismatch fit_shift_share(randn(rng, N + 1), d)
    end

    @testset "36j. balance test size under the null" begin
        # The standing rule: a test statistic owes a null-distribution
        # Monte Carlo. The precise rates, and the fact that they are
        # nominal only when the effective number of shocks is large,
        # are recorded in docs/AUDIT.md. Asserted here is the bound
        # that would catch the test breaking outright.
        rng = MersenneTwister(5701)
        N = 300; K = 100
        W = ss_shares(rng, N, K)
        reps = 150
        rej05 = 0; ps = Float64[]
        for _ in 1:reps
            g = randn(rng, K)
            d = shift_share_design(W, g)
            # independent of g, but sectorally structured
            r = ss_resid(rng, W; mix = 1.0)
            b = balance_test(d, r)
            push!(ps, b.p_values[1])
            b.p_values[1] <= 0.05 && (rej05 += 1)
        end
        # Measured over 2000 reps at 100 effective shocks: 0.056.
        @test rej05 / reps < 0.15
        @test mean(ps) > 0.3
        @test all(p -> 0.0 <= p <= 1.0, ps)
        # A p-value distribution concentrated near zero or one would
        # both be wrong; under the null it should be roughly uniform.
        @test 0.2 < count(<=(0.5), ps) / reps < 0.8
    end

    @testset "36k. spatial correlation of shares and regressor" begin
        # Moran's I reused from SpatialDiagnostics. Industries cluster
        # geographically, so spatially correlated shares are the normal
        # case rather than a problem -- what matters is that a spatial
        # correction is the wrong tool here, which the docstring says
        # and this testset documents by construction.
        side = 8; N = side * side; K = 6
        coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
        Wsp = make_queen_weights(side)
        rng = MersenneTwister(5801)

        # Build shares that vary smoothly in space, so the exposure
        # pattern is genuinely autocorrelated. The centres must be
        # distinct: drawing them independently lets two industries land
        # on the same cell, which makes two share columns identical and
        # the shock-level projection non-unique -- which is exactly what
        # fit_shift_share refuses, so it would be a defect in this DGP
        # rather than a finding.
        W = zeros(N, K)
        centres = randperm(rng, N)[1:K]
        for k in 1:K
            c = coords[centres[k], :]
            W[:, k] = [exp(-0.05 * sum(abs2, coords[i, :] .- c))
                       for i in 1:N] .+ 0.05
        end
        W ./= sum(W; dims = 2)
        # Guard the guard: the design this testset builds must be the
        # full-rank one, or the assertions below are vacuous.
        @test rank(W) == K
        d = shift_share_design(W, randn(rng, K))
        y = d.X .* 1.0 .+ randn(rng, N)
        f = fit_shift_share(y, d)

        m = shift_share_moran(d, Wsp)
        @test length(m.share_I) == K
        @test length(m.share_p) == K
        @test isnothing(m.residual)
        @test 1 <= m.worst_share <= K
        @test all(I -> -2.0 <= I <= 2.0, filter(isfinite, m.share_I))
        @test all(p -> 0.0 <= p <= 1.0, m.share_p)
        # Smoothly varying shares are positively autocorrelated.
        @test m.share_I[m.worst_share] > 0.0

        mf = shift_share_moran(d, Wsp; fit = f)
        @test !isnothing(mf.residual)
        @test mf.regressor.I_stat ≈ m.regressor.I_stat
        # Passing a weight matrix of the wrong size is refused by name,
        # since confusing the spatial weights with the exposure shares
        # is the obvious mistake here.
        @test_throws DimensionMismatch shift_share_moran(d,
            make_queen_weights(side + 1))
    end
end
