@testset "29. Identification guards on the design matrix" begin
    _progress("29. Identification guards on the design matrix")
    SU = SpatialEconometrics.SpatialUtils

    # Two estimators used to meet a rank-deficient design the same way
    # the audit kept finding: `fit_spatial` died inside LAPACK with
    # "Zero pivot found at index 2", which says nothing about the design,
    # and `fit_gravity` returned coefficients of order 1e12 that cancel
    # against each other with a plausible pseudo-R2 beside them. Neither
    # is a numerical hiccup: the coefficient is not identified, and the
    # estimator should say which column is the problem.

    @testset "29a. rank_deficient_columns against known ranks" begin
        # Deterministic designs throughout: a pinned random draw is not
        # a reference, and the streams differ across Julia versions.
        N = 40
        v1 = Float64[sin(i / 3) for i in 1:N]
        v2 = Float64[cos(i / 5) for i in 1:N]
        v3 = Float64[i % 7 for i in 1:N]

        @test isempty(SU.rank_deficient_columns(hcat(v1, v2, v3)))

        # Which column of a dependent set gets deferred is pivoting's
        # choice -- it keeps the largest-norm ones -- so the claim is
        # that exactly one goes, and that dropping it restores full
        # rank. Naming an index would be asserting LAPACK's ordering.
        for A in (hcat(v1, v2, v1), hcat(v1, v2, 2 .* v1 .- 3 .* v2))
            bad = SU.rank_deficient_columns(A)
            @test length(bad) == 1
            keep = setdiff(1:3, bad)
            @test isempty(SU.rank_deficient_columns(A[:, keep]))
            @test rank(A) == 2
        end

        # An intercept duplicated, which is what a caller who adds their
        # own constant column hands to fit_spatial.
        dup_const = hcat(ones(N), ones(N), v1)
        @test length(SU.rank_deficient_columns(dup_const)) == 1

        # Fewer rows than columns: the excess cannot be independent.
        wide = hcat(v1[1:3], v2[1:3], v3[1:3], v1[1:3] .+ v2[1:3])
        @test length(SU.rank_deficient_columns(wide)) == 1

        @test SU.rank_deficient_columns(zeros(N, 2)) == [1, 2]
        @test isempty(SU.rank_deficient_columns(zeros(N, 0)))

        # False positives are the thing to fear in a guard that throws,
        # so: a design that is merely awkward must pass. Correlation of
        # 0.9999 between two columns is identified, and so is a column
        # scaled by 1e6 -- the test is on the conditioning of the span,
        # not on how big the numbers are.
        near = v1 .+ 1e-4 .* v2
        @test cor(v1, near) > 0.9999
        @test isempty(SU.rank_deficient_columns(hcat(v1, near, v3)))
        @test isempty(SU.rank_deficient_columns(hcat(1e6 .* v1, v2, v3)))
    end

    @testset "29b. fit_spatial rejects a design it cannot identify" begin
        W = row_standardize(make_rook_weights(8))
        N = size(W, 1)
        x = Float64[sin(i / 4) for i in 1:N]
        X = reshape(x, N, 1)
        # y built from the model directly, with a deterministic
        # disturbance, so every number here is reproducible on any
        # version.
        eps = Float64[0.3 * cos(i / 2) for i in 1:N]
        rho_true, beta_true = 0.5, 1.5
        y = (I - rho_true * W) \ (x .* beta_true .+ eps)

        # The caller supplied their own intercept, and one is added.
        @test_throws ArgumentError fit_spatial(SAR(), W, hcat(ones(N), x), y)
        # A duplicated covariate.
        @test_throws ArgumentError fit_spatial(SAR(), W, hcat(x, x), y)
        # A third column that is a combination of the first two.
        x2 = Float64[i % 5 for i in 1:N]
        @test_throws ArgumentError fit_spatial(
            SAR(), W, hcat(x, x2, x .- 2 .* x2), y)
        # SDM builds W*X into the design, so a duplicate there is caught
        # on the augmented matrix rather than on what the caller passed.
        @test_throws ArgumentError fit_spatial(SDM(), W, hcat(x, x), y)

        # The message has to name the problem, or it is no better than
        # the LAPACK one it replaces.
        err = try
            fit_spatial(SAR(), W, hcat(ones(N), x), y); nothing
        catch e
            sprint(showerror, e)
        end
        @test occursin("not identified", err)
        @test occursin("intercept", err)
        @test occursin("fit_spatial:", err)

        # And a design that is fine stays fine: the guard is a
        # precondition, not a change to the estimator.
        m = fit_spatial(SAR(), W, X, y)
        @test isapprox(m.rho, rho_true; atol = 0.2)
        @test isapprox(m.coef[2], beta_true; atol = 0.2)
    end

    @testset "29c. fit_gravity rejects what the fixed effects absorb" begin
        No, Nd = 10, 8
        N = No * Nd
        o = repeat(1:No, inner = Nd)
        d = repeat(1:Nd, outer = No)
        # Deterministic flows from a known log-link DGP, so the control
        # below is a statement about this test's data and not a draw.
        # Both covariates have to vary within origin AND within
        # destination, or the fixed effects absorb one and the guard
        # fires on the control itself. Modular functions of the pair
        # index do that without an RNG.
        logd = Float64[1.0 + ((o[n] * d[n]) % 11) / 4 for n in 1:N]
        lang = Float64[((o[n] + 2 * d[n]) % 5 == 0) ? 1.0 : 0.0 for n in 1:N]
        Xg = hcat(logd, lang)
        ao = Float64[0.3 * sin(i) for i in 1:No]
        ad = Float64[0.2 * cos(i) for i in 1:Nd]
        eta = Xg * [-0.9, 0.4] .+ 3.0
        for n in 1:N; eta[n] += ao[o[n]] + ad[d[n]]; end
        yg = round.(exp.(eta))

        # Control first: the guard must not block a design that is fine.
        m = fit_gravity(PoissonGravity(), yg, Xg, o, d;
                        fe = BilateralFE(), maxiter = 300, tol = 1e-12)
        @test m.converged
        @test isapprox(m.coef[1], -0.9; atol = 0.05)
        @test isapprox(m.coef[2], 0.4; atol = 0.05)

        # A duplicated regressor used to reach LAPACK and throw
        # SingularException from inside the solve.
        @test_throws ArgumentError fit_gravity(
            PoissonGravity(), yg, hcat(logd, logd), o, d; fe = BilateralFE())

        # The quiet one, and the reason the per-column check was not
        # enough: each column survives demeaning on its own, but their
        # sum is a pure origin-level term that the fixed effects absorb.
        # Before the guard this returned coefficients of order 1e12.
        u = Float64[0.5 + ((3i) % 13) / 7 for i in 1:N]
        orig_level = Float64[o[n] for n in 1:N]
        pair = hcat(u, orig_level .- u)
        @test maximum(abs, pair[:, 1]) > 1e-6     # neither column is
        @test maximum(abs, pair[:, 2]) > 1e-6     # absorbed on its own
        @test_throws ArgumentError fit_gravity(
            PoissonGravity(), yg, pair, o, d; fe = OriginFE())

        # The negative binomial goes through the same core, so it is
        # guarded too.
        @test_throws ArgumentError fit_gravity(
            NegBinGravity(), yg, hcat(logd, logd), o, d; fe = BilateralFE())
    end

    @testset "29d. the same guard at every other entry point" begin
        # Before this sweep, nine of these died inside LAPACK with a
        # message about a zero pivot and six returned quietly: the
        # minimum-norm solve splits the unidentified coefficient across
        # the dependent columns, so fit_spatial_error handed back 0.8833
        # twice where the truth was one coefficient of 1.77, with NaN
        # standard errors beside it.
        W = row_standardize(make_rook_weights(8))
        N = size(W, 1)
        x = Float64[sin(i / 4) for i in 1:N]
        eps = Float64[0.2 * cos(i / 3) for i in 1:N]
        y = (I - 0.4 * W) \ (1.0 .+ 2.0 .* x .+ eps)
        Xb = hcat(x, x)                       # the duplicated column
        yb = Float64.(y .> median(y))
        yo = Int.(1 .+ (y .> median(y)) .+ (y .> quantile(y, 0.75)))
        coords = hcat(repeat(1.0:8.0, inner = 8), repeat(1.0:8.0, outer = 8))
        labels = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]

        @test_throws ArgumentError fit_spatial_error(W, Xb, y)
        @test_throws ArgumentError fit_slx(W, Xb, y)
        @test_throws ArgumentError fit_sac(W, W, Xb, y; method = ExactLU())
        @test_throws ArgumentError fit_sdem(W, Xb, y; method = ExactLU())
        @test_throws ArgumentError fit_spatial_regime(W, Xb, y,
            ExogenousRegimes(labels))
        @test_throws ArgumentError fit_gwr(Xb, y, coords;
            bandwidth = AdaptiveBandwidth(30))
        @test_throws ArgumentError fit_mgwr(Xb, y, coords; inference = false)
        @test_throws ArgumentError fit_spatial_probit(W, Xb, yb)
        @test_throws ArgumentError fit_bayesian_spatial_probit(W, Xb, yb;
            n_samples = 20, burnin = 5)
        @test_throws ArgumentError fit_spatial_ordered_probit(W, Xb, yo;
            n_samples = 20, burnin = 5, n_categories = 3)
        @test_throws ArgumentError fit_esf(Xb, y, W; family = :gaussian,
            max_eigenvectors = 5)
        @test_throws ArgumentError fit_re_esf(Xb, y, W)

        # Panels. The design that gets solved is the within-transformed
        # one, so that is what the guard has to look at.
        L = 5; Np = L * L; Tp = 4; NT = Np * Tp
        Wp = row_standardize(make_rook_weights(L))
        xp = Float64[sin(i / 5) for i in 1:NT]
        Sp = lu(sparse(1.0I, Np, Np) - 0.3 * Wp)
        mu = Float64[0.4 * cos(i) for i in 1:Np]
        yp = zeros(NT)
        for t in 1:Tp
            idx = ((t - 1) * Np + 1):(t * Np)
            yp[idx] = Sp \ (mu .+ 1.2 .* xp[idx] .+
                            Float64[0.1 * sin(i + t) for i in 1:Np])
        end
        Xpb = hcat(xp, xp)

        @test_throws ArgumentError fit_spatial_panel(SAR(), IndivFE(), Wp,
            Xpb, yp, Np, Tp)
        @test_throws ArgumentError fit_spatial_error(IndivFE(), Wp, Xpb,
            yp, Np, Tp)
        @test_throws ArgumentError fit_dynamic_spatial_panel(SAR(),
            IndivFE(), Wp, Xpb, yp, Np, Tp)
        en = Float64[0.3 * cos(i / 2) for i in 1:NT]
        z1 = Float64[sin(i / 7) for i in 1:NT]
        z2 = Float64[cos(i / 11) for i in 1:NT]
        @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), Wp, Xpb,
            reshape(en, NT, 1), hcat(z1, z2), yp, Np, Tp)
        # Duplicated instruments are unidentified too, and they fail in a
        # different place: inv(H'H), before the structural solve.
        @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), Wp,
            reshape(xp, NT, 1), reshape(en, NT, 1), hcat(z1, z1), yp, Np, Tp)

        # Not a duplicate at all: a regressor that is constant within
        # each unit, which individual effects sweep out entirely. This
        # one returned a coefficient before -- the column the caller
        # passed has plenty of variation, it is only the transformed
        # design that is short a rank.
        xinv = repeat(Float64[0.7 * sin(i) for i in 1:Np], Tp)
        @test_throws ArgumentError fit_spatial_panel(SAR(), IndivFE(), Wp,
            hcat(xp, xinv), yp, Np, Tp)
        # and it is fine without those fixed effects, which is the proof
        # that the guard is reading the transform and not the column.
        @test fit_spatial_panel(SAR(), NoFE(), Wp, hcat(xp, xinv), yp,
                                Np, Tp) isa SpatialPanelModel

        # Dynamic GMM carries the lagged response beside the covariates.
        Ng, Tg = 25, 6
        Xg1 = Float64[sin(i / 3 + t) for i in 1:Ng, t in 1:Tg]
        Xten = zeros(Ng, Tg, 2)
        Xten[:, :, 1] = Xg1
        Xten[:, :, 2] = Xg1
        Yg = zeros(Ng, Tg)
        Yg[:, 1] = Float64[0.5 * cos(i) for i in 1:Ng]
        for t in 2:Tg
            Yg[:, t] = 0.4 .* Yg[:, t - 1] .+ 0.8 .* Xg1[:, t] .+
                       Float64[0.05 * sin(i * t) for i in 1:Ng]
        end
        @test_throws ArgumentError fit_dynamic_gmm(SystemGMM(), Yg, Xten;
                                                   max_lag = 3)
    end

    @testset "29e. the search interval for ρ against W's spectrum (AUDIT 33)" begin
        SU = SpatialEconometrics.SpatialUtils
        rng33 = MersenneTwister(33)
        co33 = rand(rng33, 400, 2)
        Wr33 = make_rook_weights(20)
        Wq33 = make_queen_weights(20)
        # Binary rook, r(W) ≈ 3.96. Typed explicitly: on Julia 1.10,
        # `Float64.(W .> 0)` on a sparse matrix has eltype Any.
        Wb33 = SparseMatrixCSC{Float64, Int}(Wr33 .> 0)
        Wi33 = copy(Wb33); Wi33[1, :] .= 0; Wi33[:, 1] .= 0; dropzeros!(Wi33)
        Wd33 = distance_weights(co33, 0.1; row_standardize = false)

        # The bounds contain the radius computed densely, for row-standardised,
        # bipartite, island-bearing and unscaled W alike.
        for W in (Wr33, Wq33, knn_weights(co33, 6), Wb33, Wi33, Wd33)
            lo, hi = SU._spectral_radius_bounds(W)
            r = maximum(abs, eigvals(Matrix(W)))
            @test lo - 1e-9 <= r <= hi + 1e-9
        end
        @test SU._spectral_radius_bounds(Wr33) == (1.0, 1.0)
        Wneg = copy(Wb33); Wneg[1, 2] = -2.0
        @test SU._spectral_radius_bounds(Wneg) == (0.0, maximum(sum(abs, Wneg; dims = 2)))

        draw33(W, ρ, s) = (r = MersenneTwister(s); n = size(W, 1); x = randn(r, n);
                           (reshape(x, :, 1), (I - ρ * Matrix(W)) \ (1 .+ 2 .* x .+ randn(r, n))))
        # Brute force on the dense matrix: the profile likelihood on a grid,
        # then refined, with no LU, no Brent and no package code.
        function dense_rho(W, X, y, lo, hi)
            Xa = [ones(length(y)) X]; Wy = Matrix(W) * y; Wm = Matrix(W)
            f(ρ) = (e = (y .- ρ .* Wy) .- Xa * (Xa \ (y .- ρ .* Wy));
                    logabsdet(I - ρ * Wm)[1] - length(y) / 2 * log(sum(abs2, e)))
            g = range(lo, hi; length = 2001)
            k = argmax(f.(g)); a, b = g[max(k - 1, 1)], g[min(k + 1, end)]
            for _ in 1:200
                m1 = a + (b - a) / 3; m2 = b - (b - a) / 3
                f(m1) < f(m2) ? (a = m1) : (b = m2)
            end
            (a + b) / 2
        end

        # (b) An unscaled W: the default interval crosses 1/r(W), and the
        # Chebyshev traces would extrapolate. Both refuse.
        X33, y33 = draw33(Wb33, 0.2, 7)
        @test_throws ArgumentError fit_spatial(SAR(), Wb33, X33, y33)
        @test_throws ArgumentError fit_spatial(SAR(), Wb33, X33, y33; rho_bounds = (-0.25, 0.25))
        @test_throws ArgumentError SU.precompute_chebyshev_traces(Wb33, ChebyshevApprox(order = 6))
        m33 = fit_spatial(SAR(), Wb33, X33, y33; method = ExactLU(), rho_bounds = (-0.25, 0.25))
        @test m33.rho ≈ dense_rho(Wb33, X33, y33, -0.25, 0.25) atol = 1e-5
        # The Chebyshev expansion is undefined for |ρ| ≥ 1.
        @test_throws ArgumentError fit_spatial(SAR(), Wr33, X33, y33; rho_bounds = (-1.2, 0.99))

        # Every other entry point with a default interval refuses the same W.
        @test_throws ArgumentError fit_spatial_error(Wb33, X33, y33)
        @test_throws ArgumentError fit_sac(Wb33, Wb33, X33, y33)
        @test_throws ArgumentError fit_sdem(Wb33, X33, y33)
        @test_throws ArgumentError fit_spatial_regime(Wb33, X33, y33,
                                                      ExogenousRegimes(repeat([1, 2], 200)))
        Xp33 = randn(MersenneTwister(8), 400 * 3, 1); yp33 = randn(MersenneTwister(9), 400 * 3)
        @test_throws ArgumentError fit_spatial_panel(SAR(), TwoWayFE(), Wb33, Xp33, yp33, 400, 3)

        # (a) A row-standardised queen W at ρ = −1.2: admissible, since
        # 1/λ_min ≈ −1.92, but below the default interval. The exact fit lands
        # on −0.995 and says so; widened, it matches the dense maximiser.
        Xq33, yq33 = draw33(Wq33, -1.2, 3)
        mq = @test_logs (:warn, r"lower edge") match_mode = :any fit_spatial(SAR(), Wq33, Xq33, yq33;
                                                                           method = ExactLU())
        @test mq.rho ≈ -0.995 atol = 1e-5
        lo33 = 1 / minimum(real, eigvals(Matrix(Wq33))) + 1e-3
        mw = @test_logs min_level = Base.CoreLogging.Warn fit_spatial(SAR(), Wq33, Xq33, yq33;
                                method = ExactLU(), rho_bounds = (lo33, 0.995))
        @test mw.rho ≈ dense_rho(Wq33, Xq33, yq33, lo33, 0.995) atol = 1e-5
        @test mw.rho < -1

        # A fit well inside the interval stays silent.
        Xs33, ys33 = draw33(Wr33, 0.5, 2)
        @test_logs min_level = Base.CoreLogging.Warn fit_spatial(SAR(), Wr33, Xs33, ys33)

        # (c) The polish: a default (Chebyshev) fit reports the exact maximum.
        # Against ExactLU, ρ̂ to 1e-6 (ten times Brent's 1e-7 tolerance) and
        # ℓ to a relative 1e-8. Measured on 1.13 and 1.10: at most 3.4e-8 and
        # 5e-15. With the polish disabled, every cell fails both, by at least
        # 3.9e-4 and 7.8e-7, so a search left unpolished is caught.
        function sem33(W, λ, s)
            r = MersenneTwister(s); n = size(W, 1); x = randn(r, n)
            reshape(x, :, 1), 1 .+ 2 .* x .+ (I - λ * Matrix(W)) \ randn(r, n)
        end
        function panel33(W, ρ, T, s)
            r = MersenneTwister(s); n = size(W, 1); A = lu(sparse(I - ρ * W)); μ = randn(r, n)
            X = randn(r, n * T); y = similar(X)
            for t in 1:T
                idx = (t - 1) * n + 1:t * n
                y[idx] = A \ (μ .+ 0.3t .+ 2 .* X[idx] .+ randn(r, n))
            end
            reshape(X, :, 1), y
        end
        relgap(a, b) = abs(a - b) / abs(b)
        for (W, ρ) in ((Wq33, -0.8), (make_rook_weights(30), 0.9))
            X, y = draw33(W, ρ, 11)
            a = fit_spatial(SAR(), W, X, y); b = fit_spatial(SAR(), W, X, y; method = ExactLU())
            @test abs(a.rho - b.rho) < 1e-6
            @test relgap(a.loglik, b.loglik) < 1e-8
            X, y = sem33(W, ρ, 12)
            a = fit_spatial_error(W, X, y); b = fit_spatial_error(W, X, y; method = ExactLU())
            @test abs(a.lambda - b.lambda) < 1e-6
            @test relgap(a.loglik, b.loglik) < 1e-8
            X, y = panel33(W, ρ, 3, 13); n = size(W, 1)
            a = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, n, 3)
            b = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, n, 3; method = ExactLU())
            @test abs(a.rho - b.rho) < 1e-6
            @test relgap(a.loglik, b.loglik) < 1e-8
        end
    end
end
