@testset "39. Spatio-temporal models" begin
    _progress("39. Spatio-temporal models")
    # As above: implications, never guarded tests.
    ST = SpatialEconometrics.SpatioTemporalModels

    @testset "39a. STAR against explicit normal equations" begin
        rng = MersenneTwister(3_901)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 18
        Wd = Matrix(W)
        X = randn(rng, N * T, 2)
        y = Vector{Float64}(undef, N * T)
        y[1:N] = randn(rng, N)
        for t in 2:T
            prev = y[((t - 2) * N + 1):((t - 1) * N)]
            r = ((t - 1) * N + 1):(t * N)
            y[r] = 0.3 .* prev .+ 0.25 .* (Wd * prev) .+
                   X[r, :] * [0.8, -0.4] .+ randn(rng, N)
        end

        m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)
        @test m isa STARModel
        @test m.T_eff == T - 1
        @test size(m.phi) == (2, 1)
        @test length(m.beta) == 2
        @test nobs(m) == N * (T - 1)

        # Rebuild the design and solve the normal equations directly.
        Teff = T - 1
        D = Matrix{Float64}(undef, N * Teff, 5)
        yv = Vector{Float64}(undef, N * Teff)
        for t in 1:Teff
            src = ((t - 1) * N + 1):(t * N)
            dst = ((t - 1) * N + 1):(t * N)
            tgt = (t * N + 1):((t + 1) * N)
            D[dst, 1] = y[src]
            D[dst, 2] = Wd * y[src]
            D[dst, 3] = X[tgt, 1]
            D[dst, 4] = X[tgt, 2]
            D[dst, 5] .= 1.0
            yv[dst] = y[tgt]
        end
        theta = (D' * D) \ (D' * yv)
        @test isapprox(m.coef, theta; rtol = 1e-9, atol = 1e-11)
        @test isapprox(residuals(m), yv .- D * theta;
                       rtol = 1e-9, atol = 1e-11)
        s2 = sum(abs2, yv .- D * theta) / (N * Teff - 5)
        @test isapprox(m.sigma2, s2; rtol = 1e-10)
        @test isapprox(vcov(m), s2 .* inv(Symmetric(D' * D));
                       rtol = 1e-8, atol = 1e-12)

        # phi is laid out as documented: row k+1 multiplies W^k.
        @test m.phi[1, 1] == m.coef[1]
        @test m.phi[2, 1] == m.coef[2]
        @test "y(t-1)" in coefnames(m)
        @test "W^1 y(t-1)" in coefnames(m)
    end

    @testset "39b. STAR: S = 0 is a non-spatial panel AR" begin
        rng = MersenneTwister(3_902)
        W = cf_grid_W(4, 4); N = size(W, 1); T = 20
        X = randn(rng, N * T, 1)
        y = randn(rng, N * T)
        m = fit_star(y, X, W, N, T; spatial_lags = 0, temporal_lags = 2)
        @test size(m.phi) == (1, 2)
        @test length(m.coef) == 2 + 1 + 1   # two AR, one x, intercept
        # With S = 0 the design never touches W, so changing W cannot
        # change the fit.
        W2 = cf_grid_W(4, 4) .* 0.5
        m2 = fit_star(y, X, sparse(W2), N, T;
                      spatial_lags = 0, temporal_lags = 2)
        @test isapprox(m.coef, m2.coef; rtol = 1e-12, atol = 1e-14)
    end

    @testset "39c. STAR: named failures and the rank guard" begin
        rng = MersenneTwister(3_903)
        W = cf_grid_W(4, 4); N = size(W, 1); T = 12
        X = randn(rng, N * T, 1); y = randn(rng, N * T)
        @test_throws ArgumentError fit_star(y, X, W, N, T; temporal_lags = 0)
        @test_throws ArgumentError fit_star(y, X, W, N, T; spatial_lags = -1)
        @test_throws ArgumentError fit_star(y, X, W, N, T; temporal_lags = T)
        @test_throws DimensionMismatch fit_star(y, X, W, N, T + 1)
        let yn = copy(y); yn[1] = NaN
            @test_throws ArgumentError fit_star(yn, X, W, N, T)
        end
        # A complete graph gives W^2 proportional to W, so the spatial
        # powers are dependent and raising S must be refused rather
        # than failing inside LAPACK.
        let n = 8
            A = ones(n, n) - Matrix(1.0I, n, n)
            Wc = sparse(A ./ sum(A, dims = 2))
            Xc = randn(rng, n * T, 1); yc = randn(rng, n * T)
            @test_throws ArgumentError fit_star(yc, Xc, Wc, n, T;
                                                 spatial_lags = 3,
                                                 temporal_lags = 1)
        end
    end

    @testset "39d. the IRF's long run is the spatial multiplier" begin
        rng = MersenneTwister(3_904)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 20
        Wd = Matrix(W); Im = Matrix(1.0I, N, N)
        X = randn(rng, N * T, 1)
        y = Vector{Float64}(undef, N * T)
        y[1:N] = randn(rng, N)
        for t in 2:T
            prev = y[((t - 2) * N + 1):((t - 1) * N)]
            r = ((t - 1) * N + 1):(t * N)
            y[r] = 0.25 .* prev .+ 0.2 .* (Wd * prev) .+
                   X[r, :] * [0.5] .+ randn(rng, N)
        end
        m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)
        @test m.stationary
        @test m.spectral_radius < 1

        irf = space_time_irf(m; horizon = 300)
        @test irf.psi[1] == Matrix(1.0I, N, N)
        A1 = m.phi[1, 1] .* Im .+ m.phi[2, 1] .* Wd
        acc = zeros(N, N)
        for h in 0:300; acc .+= irf.psi[h + 1]; end
        @test isapprox(acc, inv(Im - A1); rtol = 1e-8, atol = 1e-10)

        cum = space_time_irf(m; horizon = 300, cumulative = true)
        @test isapprox(cum.total[end], sum(inv(Im - A1)) / N;
                       rtol = 1e-8, atol = 1e-10)
        # own + spillover == total, at every horizon, by construction.
        @test all(isapprox.(cum.own .+ cum.spillover, cum.total;
                            rtol = 1e-10, atol = 1e-12))
        # Horizon 0 is purely own: a shock has not yet propagated.
        @test isapprox(irf.own[1], 1.0; rtol = 1e-12)
        @test isapprox(irf.spillover[1], 0.0; atol = 1e-12)

        # store_matrices = false keeps the summaries only.
        let r2 = space_time_irf(m; horizon = 10, store_matrices = false)
            @test isempty(r2.psi)
            @test length(r2.total) == 11
        end
        @test_throws ArgumentError space_time_irf(m; horizon = -1)
        @test occursin("Impulse Response", sprint(show, irf))
    end

    @testset "39e. interactive FE reduces to the audited panel" begin
        # n_factors = 0 removes the factor step entirely, so the
        # estimator must be the package's own SAR panel. The reference
        # is independently written, already-audited code.
        rng = MersenneTwister(3_905)
        W = cf_grid_W(6, 5); N = size(W, 1); T = 20
        Im = Matrix(1.0I, N, N)
        X = randn(rng, N * T, 2)
        Ainv = inv(Im - 0.4 * Matrix(W))
        y = Vector{Float64}(undef, N * T)
        for t in 1:T
            r = ((t - 1) * N + 1):(t * N)
            y[r] = Ainv * (X[r, :] * [1.0, -0.5] .+ randn(rng, N))
        end

        # fit_spatial_panel prepends an intercept under NoFE, so the
        # comparable design carries one explicitly.
        ref = fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T;
                                approach = DirectQML(), method = ExactLU())
        m0 = fit_interactive_fe(y, hcat(ones(N * T), X), W, N, T;
                                n_factors = 0)
        # brent_maximize's own tolerance is 1e-8, which is the binding
        # constraint here rather than anything structural.
        @test isapprox(m0.rho, ref.rho; atol = 1e-6)
        @test isapprox(m0.beta, ref.coef[1:3]; atol = 1e-6)
        @test isapprox(m0.loglik, ref.loglik; rtol = 1e-9)
        @test m0.n_factors == 0
        @test size(m0.factors) == (T, 0)
        @test m0.converged
        @test factor_projection(m0) == zeros(T, T)
    end

    @testset "39f. factors are identified only up to rotation" begin
        rng = MersenneTwister(3_906)
        W = cf_grid_W(6, 5); N = size(W, 1); T = 20
        Im = Matrix(1.0I, N, N)
        Ainv = inv(Im - 0.4 * Matrix(W))
        X = randn(rng, N * T, 2)
        lam = randn(rng, N); f = randn(rng, T)
        y = Vector{Float64}(undef, N * T)
        for t in 1:T
            r = ((t - 1) * N + 1):(t * N)
            y[r] = Ainv * (X[r, :] * [1.0, -0.5] .+ lam .* f[t] .+
                           randn(rng, N))
        end

        m1 = fit_interactive_fe(y, X, W, N, T; n_factors = 1,
                                rng = MersenneTwister(1))
        m2 = fit_interactive_fe(y, X, W, N, T; n_factors = 1, n_starts = 4,
                                rng = MersenneTwister(999))
        # rho and beta are identified.
        @test isapprox(m1.rho, m2.rho; atol = 1e-6)
        @test isapprox(m1.beta, m2.beta; atol = 1e-6)
        # The projection is identified; the factors themselves are not,
        # so only the projection is pinned.
        @test isapprox(factor_projection(m1), factor_projection(m2);
                       atol = 1e-6)
        # A sign flip is the simplest rotation and leaves the
        # projection exactly alone.
        let F = m1.factors, Ff = -m1.factors
            P = F * ((F' * F) \ F')
            Pf = Ff * ((Ff' * Ff) \ Ff')
            @test isapprox(P, Pf; rtol = 1e-12, atol = 1e-14)
        end
        # Normalisation F'F/T = I_r.
        @test isapprox(m1.factors' * m1.factors ./ T, Matrix(1.0I, 1, 1);
                       rtol = 1e-10)
        # The projection is idempotent and symmetric, as a projection.
        let P = factor_projection(m1)
            @test isapprox(P * P, P; rtol = 1e-9, atol = 1e-11)
            @test isapprox(P, P'; rtol = 1e-12, atol = 1e-14)
            @test isapprox(tr(P), 1.0; rtol = 1e-8)   # rank 1
        end
    end

    @testset "39g. interactive FE: structure and named failures" begin
        rng = MersenneTwister(3_907)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 18
        X = randn(rng, N * T, 2); y = randn(rng, N * T)

        m = fit_interactive_fe(y, X, W, N, T; n_factors = 2)
        @test m.n_factors == 2
        @test size(m.factors) == (T, 2)
        @test size(m.loadings) == (N, 2)
        @test length(coef(m)) == 3
        @test coefnames(m)[1] == "rho"
        @test all(isfinite, coef(m))
        @test all(>(0), stderror(m))
        @test length(residuals(m)) == N * T
        @test isapprox(tr(factor_projection(m)), 2.0; rtol = 1e-8)

        @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                       n_factors = -1)
        # Saturating the factor space is refused.
        @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                       n_factors = min(N, T))
        @test_throws ArgumentError fit_interactive_fe(y, X, W, N, T;
                                                       n_starts = 0)
        @test_throws DimensionMismatch fit_interactive_fe(y, X, W, N, T + 1)
        let yn = copy(y); yn[2] = NaN
            @test_throws ArgumentError fit_interactive_fe(yn, X, W, N, T)
        end
        @test occursin("Interactive Fixed Effects", sprint(show, m))
        @test occursin("rotation", sprint(show, m))
    end

    @testset "39h. forecasting" begin
        rng = MersenneTwister(3_908)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 20
        Wd = Matrix(W)
        X = randn(rng, N * T, 1)
        y = Vector{Float64}(undef, N * T)
        y[1:N] = randn(rng, N)
        for t in 2:T
            prev = y[((t - 2) * N + 1):((t - 1) * N)]
            r = ((t - 1) * N + 1):(t * N)
            y[r] = 0.3 .* prev .+ 0.2 .* (Wd * prev) .+
                   X[r, :] * [0.6] .+ randn(rng, N)
        end
        m = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)

        Xf = randn(rng, N * 4, 1)
        hist = y[((T - 1) * N + 1):(T * N)]
        fc = spatial_forecast(m, Xf; h = 4, y_history = hist)
        @test size(fc) == (N, 4)
        @test all(isfinite, fc)

        # The one-step forecast IS the model's own prediction, exactly.
        A1 = m.phi[1, 1] .* Matrix(1.0I, N, N) .+ m.phi[2, 1] .* Wd
        # beta excludes the intercept, which is kept in its own field.
        @test length(m.beta) == size(X, 2)
        @test m.has_intercept
        @test m.intercept == m.coef[end]
        c = m.intercept
        manual = A1 * hist .+ Xf[1:N, :] * m.beta .+ c
        @test isapprox(fc[:, 1], manual; rtol = 1e-10, atol = 1e-12)
        # And the second step iterates on the first, not on the data.
        manual2 = A1 * fc[:, 1] .+ Xf[(N + 1):(2N), :] * m.beta .+ c
        @test isapprox(fc[:, 2], manual2; rtol = 1e-10, atol = 1e-12)

        @test_throws ArgumentError spatial_forecast(m, Xf; h = 0,
                                                     y_history = hist)
        @test_throws ArgumentError spatial_forecast(m, Xf; h = 4)
        @test_throws DimensionMismatch spatial_forecast(m, Xf; h = 4,
                                                         y_history = hist[1:3])
        @test_throws DimensionMismatch spatial_forecast(m, Xf; h = 3,
                                                         y_history = hist)
    end

    @testset "39i. the IFE multiplier matches the static one" begin
        # With no temporal dynamics a shock propagates within the period
        # and stops, so the response is (I - rho W)^-1 and nothing after.
        rng = MersenneTwister(3_909)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 18
        X = randn(rng, N * T, 1); y = randn(rng, N * T)
        m = fit_interactive_fe(y, X, W, N, T; n_factors = 1)
        irf = space_time_irf(m, W; horizon = 5)
        @test isapprox(irf.psi[1],
                       inv(Matrix(1.0I, N, N) - m.rho * Matrix(W));
                       rtol = 1e-10)
        @test all(iszero, irf.psi[2])
        @test all(iszero, irf.total[2:end])
        # Cumulative therefore flattens after horizon 0.
        let c = space_time_irf(m, W; horizon = 5, cumulative = true)
            @test all(isapprox.(c.total[2:end], c.total[1];
                                rtol = 1e-12, atol = 1e-14))
        end
        @test_throws DimensionMismatch space_time_irf(m, cf_grid_W(3, 3);
                                                       horizon = 2)
    end

    # ── 39j ───────────────────────────────────────────────
    # Pfeifer & Deutsch define the spatial lag operator on order-k
    # NEIGHBOURHOODS, not powers of W: w_ij^(k) is "nonzero only if
    # sites i and j are kth order neighbors". `fit_star` has always
    # computed powers. Both are now available; the default is still
    # powers, so these tests pin (i) that the two agree exactly where
    # the conventions agree, and (ii) that they genuinely differ where
    # they do not -- an option that quietly made no difference would be
    # worse than no option.
    @testset "39j. powers of W versus order-k neighbours" begin
        rng = MersenneTwister(3_910)
        W = cf_grid_W(5, 4); N = size(W, 1); T = 40
        Wd = Matrix(W)
        X = randn(rng, N * T, 2)
        y = Vector{Float64}(undef, N * T)
        y[1:N] = randn(rng, N)
        for t in 2:T
            prev = y[((t - 2) * N + 1):((t - 1) * N)]
            r = ((t - 1) * N + 1):(t * N)
            y[r] = 0.30 .* prev .+ 0.25 .* (Wd * prev) .+
                   X[r, :] * [0.8, -0.4] .+ randn(rng, N)
        end

        # At S = 1 the conventions coincide by definition: W^(1) IS the
        # row-standardised first-order neighbour matrix. Identical to
        # the last bit, not merely close -- a tolerance here would hide
        # the case where :orders rebuilds W slightly differently.
        m1p = fit_star(y, X, W, N, T; spatial_lags = 1,
                       spatial_weights = :powers)
        m1o = fit_star(y, X, W, N, T; spatial_lags = 1,
                       spatial_weights = :orders)
        @test m1p.coef == m1o.coef
        @test m1p.loglik == m1o.loglik
        @test m1p.spectral_radius == m1o.spectral_radius
        @test Matrix(m1o.Wk[2]) == Wd
        @test m1o.spatial_weights === :orders
        @test m1p.spatial_weights === :powers

        # S = 0 touches neither, so it cannot depend on the choice.
        @test fit_star(y, X, W, N, T; spatial_lags = 0,
                       spatial_weights = :orders).coef ==
              fit_star(y, X, W, N, T; spatial_lags = 0,
                       spatial_weights = :powers).coef

        m2p = fit_star(y, X, W, N, T; spatial_lags = 2,
                       spatial_weights = :powers)
        m2o = fit_star(y, X, W, N, T; spatial_lags = 2,
                       spatial_weights = :orders)

        # The structural difference. On a rook lattice there are no
        # triangles, so every length-2 walk between distinct sites is a
        # distance-2 pair and the ENTIRE discrepancy is the diagonal:
        # W^2 puts weight on the unit itself and calls it a spatial lag.
        let A2p = Matrix(m2p.Wk[3]), A2o = Matrix(m2o.Wk[3])
            @test count(!iszero, diag(A2p)) == N      # all 20
            @test count(!iszero, diag(A2o)) == 0
            @test count(!iszero, A2p) == count(!iszero, A2o) + N
            # off the diagonal the supports agree here, which is what
            # isolates the diagonal as the whole of the difference
            offp = [i == j ? false : A2p[i, j] != 0
                    for i in 1:N, j in 1:N]
            @test offp == (A2o .!= 0)
            # ... and each is row-standardised in its own right
            @test all(isapprox.(sum(A2o, dims = 2), 1.0; atol = 1e-12))
        end

        # So the fits must differ, and the coefficient on the order-2
        # term is the one that moves.
        @test m2p.coef != m2o.coef
        @test abs(m2p.phi[3, 1] - m2o.phi[3, 1]) > 0.01
        @test "W^2 y(t-1)" in coefnames(m2p)
        @test "W⁽2⁾ y(t-1)" in coefnames(m2o)
        @test !("W^2 y(t-1)" in coefnames(m2o))
        @test occursin("order-k neighbours", sprint(show, m2o))
        @test occursin("powers of W", sprint(show, m2p))

        # Everything built on the fit must use the matrices the fit
        # used. The long-run multiplier is the gate: if the IRF
        # recomputed powers while the design used orders, this breaks.
        for m in (m2p, m2o)
            S = m.spatial_lags
            irf = space_time_irf(m; horizon = 500)
            @test irf.psi[1] == Matrix(1.0I, N, N)
            A1 = ST._star_A(m.phi, m.Wk, N, S, 1)
            @test maximum(abs, sum(irf.psi) .-
                          inv(Matrix(1.0I, N, N) - A1)) < 1e-12
        end

        # L = 2 as well, so the companion block structure is exercised
        # under the orders convention too.
        let mL = fit_star(y, X, W, N, T; spatial_lags = 2,
                          temporal_lags = 2, spatial_weights = :orders)
            @test mL.stationary
            Asum = ST._star_A(mL.phi, mL.Wk, N, 2, 1) .+
                   ST._star_A(mL.phi, mL.Wk, N, 2, 2)
            @test maximum(abs,
                    sum(space_time_irf(mL; horizon = 500).psi) .-
                    inv(Matrix(1.0I, N, N) - Asum)) < 1e-12
        end

        # Pfeifer & Deutsch's stationarity condition is determinantal:
        # every x solving det[x^L I - sum_l A_l x^(L-l)] = 0 must lie
        # inside the unit circle. This package reports a companion
        # spectral radius instead. Assert they are the same statement
        # by evaluating the paper's determinant AT the companion
        # eigenvalues -- it must vanish.
        let mL = fit_star(y, X, W, N, T; spatial_lags = 2,
                          temporal_lags = 2, spatial_weights = :orders),
            L = 2
            As = [ST._star_A(mL.phi, mL.Wk, N, 2, l) for l in 1:L]
            C = zeros(L * N, L * N)
            for l in 1:L
                C[1:N, ((l - 1) * N + 1):(l * N)] = As[l]
            end
            C[(N + 1):(2N), 1:N] = Matrix(1.0I, N, N)
            ev = eigvals(C)
            @test isapprox(maximum(abs, ev), mL.spectral_radius;
                           rtol = 1e-10)
            worst = 0.0
            for x in ev
                M = ComplexF64(x)^L * Matrix{ComplexF64}(I, N, N)
                for l in 1:L
                    M .-= ComplexF64(x)^(L - l) .* As[l]
                end
                worst = max(worst, abs(det(M)))
            end
            @test worst < 1e-20
        end

        # Named failures specific to the option.
        @test_throws ArgumentError fit_star(y, X, W, N, T;
            spatial_weights = :order)
        @test_throws ArgumentError fit_star(y, X, W, N, T;
            spatial_weights = :neighbours)
        # Past the graph's diameter an order matrix is empty, and that
        # is refused with the diameter rather than fitted as zeros.
        # The 5x4 rook lattice has diameter 7.
        let e = try
                    fit_star(y, X, W, N, T; spatial_lags = 20,
                             spatial_weights = :orders)
                    nothing
                catch err
                    err
                end
            @test e isa ArgumentError
            @test occursin("diameter is 7", sprint(showerror, e))
        end
        # Under :powers that same S fails for a different reason --
        # rank deficiency, not emptiness -- so the two paths must not
        # be reporting each other's diagnosis.
        @test_throws ArgumentError fit_star(y, X, W, N, T;
            spatial_lags = 20, spatial_weights = :powers)
    end
end
