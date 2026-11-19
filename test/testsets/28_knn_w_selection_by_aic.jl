@testset "28. knn W Selection by AIC" begin
    _progress("28. knn W Selection by AIC")
    # The generating design throughout: coordinates uniform on the unit
    # square, y from a SAR on the knn W at k = 6, rho = 0.5.
    K_STAR = 6
    RHO = 0.5
    function knn_draw(rng, N)
        coords = rand(rng, N, 2)
        X = randn(rng, N, 2)
        y, _ = simulate_spatial(SAR_DGP(RHO, [1.0, -0.5], 0.5),
                                knn_weights(coords, K_STAR), X; rng = rng)
        return coords, X, y
    end

    # ── 28a. structure and the guards ─────────────────────────────────
    @testset "28a. structure" begin
        rng = MersenneTwister(2801)
        coords, X, y = knn_draw(rng, 60)
        sel = select_knn_w(coords, y, X; k_range = 2:10)

        @test sel isa KnnSelectionResult
        @test sel.k_values == collect(2:10)
        @test length(sel.aic) == length(sel.k_values)
        @test length(sel.bic) == length(sel.k_values)
        @test length(sel.loglik) == length(sel.k_values)
        @test length(sel.rho) == length(sel.k_values)
        @test length(sel.converged) == length(sel.k_values)
        @test sel.criterion === :aic
        @test sel.best_k in sel.k_values
        @test size(sel.best_W) == (60, 60)
        @test occursin("Selected", sprint(show, sel))

        # best_W must be exactly the knn W at best_k, not a near miss
        @test sel.best_W ≈ knn_weights(coords, sel.best_k)

        # knn cannot leave an isolate: every unit has exactly k neighbours
        @test all(iszero, sel.n_islands)

        # the criteria must be the ones spatial_aic/spatial_bic would give
        i = findfirst(==(sel.best_k), sel.k_values)
        m = fit_spatial(SAR(), sel.best_W, X, y)
        @test sel.aic[i] ≈ spatial_aic(m) rtol = 1e-8
        @test sel.bic[i] ≈ spatial_bic(m) rtol = 1e-8
        @test sel.loglik[i] ≈ loglikelihood(m) rtol = 1e-8
        @test sel.rho[i] ≈ m.rho rtol = 1e-8

        # guards
        @test_throws ArgumentError select_knn_w(coords, y, X; k_range = 60:80)
        @test_throws ArgumentError select_knn_w(coords, y, X; criterion = :foo)
        @test_throws DimensionMismatch select_knn_w(coords, y[1:end-1], X)
        @test_throws DimensionMismatch select_knn_w(coords, y, X[1:end-1, :])

        # a range that only partly exceeds N is clipped, not rejected
        @test select_knn_w(coords, y, X; k_range = 55:70).k_values ==
              collect(55:59)
    end

    # ── 28b. the curve, and why there is something to search ──────────
    # Measured on one N = 200 draw: AIC 312.7 at k = 2, down to 294.3 at
    # the true k = 6, up to 329.1 at k = 20 — a 34.9-point spread with an
    # interior minimum. The non-monotonicity is the paper's point: too
    # sparse a W misses real neighbours, too dense a W dilutes them.
    @testset "28b. the AIC curve is not monotone" begin
        rng = MersenneTwister(5150)
        coords, X, y = knn_draw(rng, 200)
        sel = select_knn_w(coords, y, X; k_range = 2:20)

        @test all(sel.converged)
        # NOT sel.best_k == K_STAR. 28c measures that at 57% of draws at
        # this N, so pinning it on one draw asserts something the
        # measurement says fails about two draws in five -- it passed on
        # one platform and reported k = 7 on another. The recovery claim
        # belongs to the rate in 28c; what a single draw can carry is the
        # shape of the curve.
        d = diff(sel.aic)
        @test !all(>(0), d) && !all(<(0), d)        # not monotone
        @test maximum(sel.aic) - minimum(sel.aic) > 10.0
        # the minimum is interior, not at either end of the range
        @test sel.best_k != first(sel.k_values)
        @test sel.best_k != last(sel.k_values)

        # rho-hat drifts upward with k: a denser W spreads the same
        # dependence over more neighbours, and rho absorbs it. Counted as
        # upward steps rather than as a correlation with k, because
        # rho-hat is concave in k -- it climbs fast and then flattens, so
        # Pearson against raw k measures the curvature as much as the
        # monotonicity (it is 0.947 here, against 17 of 18 steps up).
        @test count(>(0), diff(sel.rho)) >= length(sel.rho) - 2
        @test sel.rho[findfirst(==(2), sel.k_values)] < RHO
        @test sel.rho[findfirst(==(20), sel.k_values)] > RHO

        # AIC and BIC rank identically here, because with the variant and
        # X fixed the parameter count does not change with k. Documented
        # rather than relied on: it is why the criterion choice is inert
        # in this search.
        selb = select_knn_w(coords, y, X; k_range = 2:20, criterion = :bic)
        @test selb.best_k == sel.best_k
        @test sortperm(sel.aic) == sortperm(selb.bic)
        @test selb.criterion === :bic

        # SDM lags X as well, so the regressors change with k too
        seld = select_knn_w(coords, y, X; k_range = 2:12, variant = SDM())
        @test all(seld.converged)
        @test seld.best_k in 2:12
    end

    # ── 28c. does the search find the generating k? ───────────────────
    # Measured over 300 draws with k searched over 2:20 (5.3% by chance):
    # k-hat is exactly 6 in 42.3% / 57.3% / 77.0% of draws at N = 100 /
    # 200 / 400, and within +/-1 in 68.3% / 86.0% / 95.0%. The median
    # k-hat is 6 at every N. The suite runs the cheap corner of that:
    # N = 200 over 2:12, 40 draws, where three independent seeds gave
    # exact rates of 0.45-0.625 and within-1 rates of 0.825-0.875
    # against a 9.1% chance rate.
    @testset "28c. recovery of the generating k, 40 draws" begin
        R = 40
        rng = MersenneTwister(2803)
        hits = Int[]
        better = 0
        for _ in 1:R
            coords, X, y = knn_draw(rng, 200)
            sel = select_knn_w(coords, y, X; k_range = 2:12)
            push!(hits, sel.best_k)
            # and the point of selecting at all: it must beat a fixed,
            # wrong k on |rho-hat - rho|. At k = 2 the fixed choice is
            # badly biased (-35.1% over 300 draws), so this is nearly
            # always true -- 98.3% there, asserted loosely here.
            r_sel = fit_spatial(SAR(), sel.best_W, X, y).rho
            r_2 = fit_spatial(SAR(), knn_weights(coords, 2), X, y).rho
            abs(r_sel - RHO) < abs(r_2 - RHO) && (better += 1)
        end

        @test count(==(K_STAR), hits) / R >= 0.25          # chance: 0.091
        @test count(k -> abs(k - K_STAR) <= 1, hits) / R >= 0.55
        @test median(hits) == K_STAR
        @test better / R >= 0.80
    end

    # ── 28d. what the wrong k costs, on one draw ──────────────────────
    # The across-draw version, 300 draws at N = 200, is in the
    # select_knn_w docstring: mean rho-hat runs 0.325 at k = 2 (-35.1%
    # bias) through 0.493 at the true k = 6 (-1.4%) to 0.646 at k = 20
    # (+29.1%), while selecting gives 0.502 (+0.5%). The suite checks the
    # direction and ordering on a single draw, which is cheap and still
    # falsifiable.
    @testset "28d. rho-hat moves with k" begin
        rng = MersenneTwister(2804)
        coords, X, y = knn_draw(rng, 200)
        r(k) = fit_spatial(SAR(), knn_weights(coords, k), X, y).rho
        r2, r6, r20 = r(2), r(K_STAR), r(20)
        # The ordering is the property, and it is scale-free: rho-hat
        # rises with k whatever the draw. Absolute margins against the
        # true rho are not -- `r20 > RHO + 0.05` held on one platform and
        # gave 0.5404 on another, because where the whole sequence sits
        # is one draw's sampling noise. The level claims are made across
        # draws in the docstring's bias table, not here.
        @test r2 < r6 < r20                 # monotone in k, both sides
        @test r20 - r2 > 0.05               # and the drift is not trivial
        @test r2 < RHO                      # too sparse understates
    end
end
