@testset "37. Spatial econometrics for networks" begin
    _progress("37. Spatial econometrics for networks")
    # Bramoulle, Djebbari and Fortin (2009) is unusually good to test
    # against: their identification results are statements about the
    # rank of integer-structured matrices, so the gates here are exact
    # integers and machine-precision identities rather than tolerances.
    # Every verdict asserted below is one the paper states in words,
    # and several of the algebraic identities are printed in it.
    NW = SpatialEconometrics.SpatialNetworks

    # Erdos-Renyi graph with no isolates, which the rank results for the
    # non-identification half assume.
    #
    # Isolates are repaired rather than rejected. At n = 150 and density
    # 0.02 the chance that a draw happens to have none is about 1 in
    # 2,000, so rejection sampling here does not terminate in any sane
    # number of tries -- and burns a 150-node sparse build on each one.
    # Attaching each isolate to one random other node is O(isolates),
    # always succeeds, and changes the density by less than a rounding.
    function nw_er(rng, n, d)
        e = [(i, j) for i in 1:n for j in (i + 1):n if rand(rng) < d]
        deg = zeros(Int, n)
        for (i, j) in e
            deg[i] += 1; deg[j] += 1
        end
        for i in 1:n
            if deg[i] == 0
                j = rand(rng, 1:(n - 1))
                j >= i && (j += 1)
                push!(e, (i, j))
                deg[i] += 1; deg[j] += 1
            end
        end
        return network_weights(e, n)
    end

    nw_star(n) = network_weights([(1, j) for j in 2:n], n)
    nw_line(n) = network_weights([(i, i + 1) for i in 1:(n - 1)], n)

    @testset "37a. network weight construction" begin
        A = [0 1 1; 1 0 0; 1 0 0]
        G = network_weights(A)
        @test size(G) == (3, 3)
        @test vec(sum(G, dims = 2)) ≈ ones(3)
        @test G[1, 2] ≈ 0.5
        @test G[2, 1] ≈ 1.0
        # :none leaves the adjacency alone.
        @test network_weights(A; normalize = :none) == sparse(Float64.(A))

        # An edge list must be told how many units there are, because
        # isolates never appear in one.
        Ge = network_weights([(1, 2), (1, 3)], 4)
        @test size(Ge) == (4, 4)
        @test iszero(Ge[4, :])
        @test Matrix(Ge[1:3, 1:3]) ≈ Matrix(G)

        # Directed graphs are supported and are not symmetrised.
        Gd = network_weights([(1, 2), (2, 3)], 3; directed = true)
        @test Gd[1, 2] ≈ 1.0
        @test Gd[2, 1] == 0.0
        @test iszero(Gd[3, :])

        # Weighted edges, and duplicates summed.
        Gw = network_weights([(1, 2, 3.0), (1, 3, 1.0)], 3)
        @test Gw[1, 2] ≈ 0.75
        Gdup = network_weights([(1, 2), (1, 2), (1, 3)], 3; normalize = :none)
        @test Gdup[1, 2] ≈ 2.0

        # Named failures rather than silent degeneracy.
        @test_throws DimensionMismatch network_weights(zeros(2, 3))
        @test_throws ArgumentError network_weights([0 1; 1 1])      # self-link
        @test_throws ArgumentError network_weights([0 -1; -1 0])    # negative
        @test_throws ArgumentError network_weights(A; normalize = :bogus)
        @test_throws ArgumentError network_weights([(1, 9)], 3)     # out of range
        @test_throws ArgumentError network_weights([(1, 2, -1.0)], 3)
        # A self-link is allowed when it is asked for by name.
        @test network_weights([0 1; 1 1]; allow_self = true)[2, 2] ≈ 0.5
    end

    @testset "37b. the published algebraic identities" begin
        # Manski (1993): the group mean INCLUDES the unit, and then
        # G^2 = G exactly -- whatever the group sizes are. The paper
        # states this as the reason his model is not identified.
        for sizes in ([5, 5, 5], [4, 5, 6], [3, 7, 11])
            G = group_weights(sizes; include_self = true)
            @test maximum(abs, Matrix(G * G - G)) < 1e-15
        end

        # Moffitt (2001) / Lee (2007): the mean EXCLUDES the unit, and
        # within a block of size s the paper gives
        #   G^2 = I/(s-1) + ((s-2)/(s-1)) G.
        for s in (3, 5, 8)
            G = group_weights([s, s, s])
            n = 3s
            pred = (1 / (s - 1)) * Matrix(1.0I, n, n) +
                   ((s - 2) / (s - 1)) * Matrix(G)
            @test maximum(abs, Matrix(G * G) - pred) < 1e-15
        end

        # Their page 47, two groups of sizes s1 and s2: G^3 is an exact
        # combination of I, G and G^2, which is why two group sizes do
        # not suffice under network fixed effects.
        for (s1, s2) in ((4, 7), (3, 5), (6, 11))
            G = Matrix(group_weights([s1, s2]))
            n = s1 + s2
            d = (s1 - 1) * (s2 - 1)
            pred = (1 / d) * Matrix(1.0I, n, n) +
                   ((s1 + s2 - 3) / d) * G +
                   ((s1 * s2 - 2 * (s1 + s2) + 3) / d) * (G * G)
            @test maximum(abs, G * G * G - pred) < 1e-14
        end

        # Complete bipartite graphs, of which the star is one, satisfy
        # G^3 = G. The paper names them as the leading failure of
        # Proposition 4.
        for n in (5, 9, 14)
            G = Matrix(nw_star(n))
            @test maximum(abs, G * G * G - G) < 1e-15
        end
        Gb = Matrix(network_weights([(i, j) for i in 1:4 for j in 5:9], 9))
        @test maximum(abs, Gb * Gb * Gb - Gb) < 1e-15
    end

    @testset "37c. the rank condition, verdict by verdict" begin
        # Proposition 2: equal group sizes are not identified; at least
        # two different sizes are.
        @test !peer_identification(group_weights([5, 5, 5])).identified
        @test !peer_identification(group_weights([8, 8])).identified
        @test !peer_identification(group_weights([3, 3, 3, 3])).identified
        @test peer_identification(group_weights([4, 6])).identified
        @test peer_identification(group_weights([3, 9])).identified
        @test peer_identification(group_weights([4, 5, 6])).identified

        # Manski's convention fails whatever the sizes, since G^2 = G.
        for sizes in ([5, 5, 5], [4, 5, 6], [3, 7, 11])
            id = peer_identification(group_weights(sizes; include_self = true))
            @test !id.identified
            @test id.rank == 2
        end

        # The ranks themselves, not just the verdicts.
        @test peer_identification(group_weights([5, 5, 5])).rank == 2
        @test peer_identification(group_weights([4, 6])).rank == 3

        # Proposition 4 is strictly more demanding: two group sizes give
        # rank 3 where 4 is needed, and three sizes give 4. This is the
        # paper's "three different classroom sizes are now necessary".
        for sizes in ([4, 6], [3, 9], [5, 12])
            @test peer_identification(group_weights(sizes)).identified
            id4 = peer_identification(group_weights(sizes); fixed_effects = true)
            @test id4.rank == 3
            @test !id4.identified
            @test id4.required == 4
        end
        for sizes in ([4, 6, 9], [3, 5, 7], [2, 5, 11, 13])
            id4 = peer_identification(group_weights(sizes); fixed_effects = true)
            @test id4.rank == 4
            @test id4.identified
        end

        # Stars pass Proposition 1 and fail Proposition 4, since
        # G^3 = G but G^2 differs from both I and G.
        for n in (5, 9, 14)
            @test peer_identification(nw_star(n)).identified
            @test !peer_identification(nw_star(n); fixed_effects = true).identified
        end

        # Intransitive triads identify: the paper's own line example.
        @test peer_identification(nw_line(8)).identified
        @test peer_identification(nw_line(8); fixed_effects = true).identified
    end

    @testset "37d. the rank tolerance is measured, not assumed" begin
        # The Gram route never forms the n^2 x m matrix, at the cost of
        # squaring the condition number. Its singular values must match
        # a dense SVD of the explicit matrix.
        let rng = MersenneTwister(4113), worst = 0.0
            for G in (group_weights([4, 6, 9]), group_weights([5, 5, 5]),
                      nw_star(9), nw_line(12), nw_er(rng, 60, 0.08))
                n = size(G, 1)
                mats = Any[sparse(1.0I, n, n), G, G * G, G * G * G]
                sg = NW._vec_svals(mats)
                sd = svdvals(hcat((vec(Matrix(m)) for m in mats)...))
                worst = max(worst, maximum(abs.(sg .- sd)) / sd[1])
            end
            # Degenerate designs sit at the sqrt(eps) noise floor; that
            # is the whole reason the tolerance is where it is.
            @test worst < 1e-7
        end

        # The regression this guards. Two equal groups of 8 are NOT
        # identified (Proposition 2), but the spurious singular value
        # the Gram route produces for them is 1.97e-8 -- above a 1e-8
        # tolerance, which therefore reports them as identified. The
        # default was moved to 1e-6 on that measurement.
        @test NW.DEFAULT_RANK_TOL == 1e-6
        @test !peer_identification(group_weights([8, 8])).identified
        @test peer_identification(group_weights([8, 8]); tol = 1e-8).identified
        @test peer_identification(group_weights([8, 8])).gap > 1e-8
        @test peer_identification(group_weights([8, 8])).gap < 1e-6

        # Identified structures sit far above the tolerance, which is
        # the other half of the separation.
        for G in (group_weights([4, 6]), nw_line(12), nw_star(9))
            @test peer_identification(G).gap > 1e-4
        end
    end

    @testset "37e. diameter and Corollary 1" begin
        # Corollary 1: a diameter of 3 or more is sufficient for
        # identification under network fixed effects.
        #
        # Two deliberate choices here, both of them scar tissue.
        #
        # First, the corollary is an implication, and it is asserted as
        # one rather than as a `@test` behind an `if` guard. A guarded
        # assertion that fails its guard does not fail -- it does not
        # run, the suite stays green, and the only trace is that the
        # assertion count is one lower. That is strictly worse than a
        # red test.
        #
        # Second, every structure in the tuple is deterministic. An
        # Erdos-Renyi draw stood here and was the thing that found the
        # problem: `rand(rng)` over a MersenneTwister does not produce
        # the same edge set on every Julia version -- the same seed gave
        # 326 edges on 1.13.2 and 288 on 1.10.12 -- and at 288 the draw
        # came out disconnected, so on 1.10 the guard failed and the
        # assertion silently vanished. A 4x4 grid (diameter 6) keeps a
        # structure that is neither a line nor a ring without betting
        # the gate on an RNG stream.
        let grid4 = network_weights(
                vcat([((r - 1) * 4 + c, (r - 1) * 4 + c + 1)
                      for r in 1:4 for c in 1:3],
                     [((r - 1) * 4 + c, r * 4 + c)
                      for r in 1:3 for c in 1:4]), 16)
            for G in (nw_line(8), nw_line(30), grid4,
                      network_weights(vcat([(i, i + 1) for i in 1:9],
                                           [(10, 1)]), 10))
                id = peer_identification(G; fixed_effects = true)
                @test !(id.diameter >= 3 && id.connected) || id.identified
            end
            # ... and the grid genuinely enters the corollary rather
            # than satisfying the implication vacuously.
            let id = peer_identification(grid4; fixed_effects = true)
                @test id.connected
                @test id.diameter == 6
            end
        end
        # The star has diameter 2 and fails, which is the paper's own
        # illustration of where the corollary stops.
        let id = peer_identification(nw_star(9); fixed_effects = true)
            @test id.diameter == 2
            @test id.connected
            @test !id.identified
        end
        @test nw_line(8) |> G -> peer_identification(G).diameter == 7
        # Disconnected structures are flagged rather than reported as
        # though every pair were reachable.
        @test !peer_identification(group_weights([4, 6])).connected
        # Opting out returns a sentinel, not a wrong number.
        @test peer_identification(nw_line(8); compute_diameter = false).diameter == -1
    end

    @testset "37f. 2SLS against explicit normal equations" begin
        # The estimator is checked against the formula as printed in
        # their Section 4, written out independently here.
        let rng = MersenneTwister(31337)
            n = 240
            G = nw_er(rng, n, 0.02)
            Gd = Matrix(G)
            X = randn(rng, n, 2)
            y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))

            Gy = Gd * y; GX = Gd * X; G2X = Gd * GX
            Xr = hcat(ones(n), Gy, X, GX)
            S  = hcat(ones(n), X, GX, G2X)
            P  = S * inv(S'S) * S'
            th = inv(Xr' * P * Xr) * (Xr' * P * y)

            f = fit_peer_effects(y, X, G)
            @test maximum(abs, coef(f) .- th) < 1e-10
            @test f.beta ≈ th[2]
            @test f.alpha ≈ th[1]
            @test f.gamma ≈ th[3:4]
            @test f.delta ≈ th[5:6]
            @test length(residuals(f)) == n
            @test nobs(f) == n

            # The fixed-effects branch is their eq. (10): the within
            # local transformation (I - G) applied throughout, with
            # S = [(I-G)X (I-G)GX (I-G)G^2X] as the paper writes it.
            M  = Matrix(1.0I, n, n) - Gd
            Xrf = hcat(M * Gy, M * X, M * GX)
            Sf  = hcat(M * X, M * GX, M * G2X)
            Pf  = Sf * inv(Sf'Sf) * Sf'
            thf = inv(Xrf' * Pf * Xrf) * (Xrf' * Pf * (M * y))

            ff = fit_peer_effects(y, X, G; fixed_effects = true)
            @test maximum(abs, coef(ff) .- thf) < 1e-10
            @test ff.alpha === nothing        # swept out by the transform
            @test ff.beta ≈ thf[1]
        end
    end

    @testset "37g. the identification check is not optional" begin
        let rng = MersenneTwister(9)
            G = group_weights([6, 6, 6])        # equal sizes: not identified
            n = size(G, 1)
            X = randn(rng, n, 2)
            y = randn(rng, n)
            # The fit refuses, and says why rather than failing inside
            # the linear algebra.
            @test_throws ArgumentError fit_peer_effects(y, X, G)

            # `check = false` does not rescue an exactly degenerate
            # network, and that is Bramoulle, Djebbari and Fortin's
            # Appendix B rather than a limitation of the override: when
            # I, G and G^2 are dependent, E(Gy|x) is perfectly collinear
            # with (iota, x, Gx), so the instrument block is collinear
            # with the included regressors and no estimate exists. The
            # design-rank guard is what fires, naming the instruments.
            @test_throws ArgumentError fit_peer_effects(y, X, G;
                                                        check = false)
            # The equivalence itself, both directions. E(Gy|x) is the
            # expected endogenous regressor, so the systematic part of
            # the reduced form with the error dropped.
            nw_EGy = function (Gm, Xm)
                nm = size(Gm, 1)
                sys = 1.0 .+ Xm * [1.0, -0.5] .+ Gm * Xm * [0.8, 0.3]
                return Gm * ((Matrix(1.0I, nm, nm) - 0.4 * Gm) \ sys)
            end
            # Dependent  =>  E(Gy|x) collinear with (iota, x, Gx).
            let Gd = Matrix(G), E = nw_EGy(Matrix(G), X)
                Z = hcat(ones(n), X, Gd * X)
                @test maximum(abs, E .- Z * (Z \ E)) < 1e-12
            end
            # Independent  =>  it is not collinear.
            let Gi = group_weights([4, 6])
                ni = size(Gi, 1)
                Xi = randn(rng, ni, 2)
                Gid = Matrix(Gi)
                E = nw_EGy(Gid, Xi)
                Zi = hcat(ones(ni), Xi, Gid * Xi)
                @test peer_identification(Gi).identified
                @test maximum(abs, E .- Zi * (Zi \ E)) > 1e-6
            end

            # The override does work where it is meant to: a network
            # whose design still has rank.
            let Gn = nw_er(rng, 120, 0.03)
                nn = size(Gn, 1)
                Xn = randn(rng, nn, 2)
                Gnd = Matrix(Gn)
                sys = 1.0 .+ Xn * [1.0, -0.5] .+ Gnd * Xn * [0.8, 0.3]
                yn = (Matrix(1.0I, nn, nn) - 0.4 * Gnd) \ (sys .+ randn(rng, nn))
                fo = fit_peer_effects(yn, Xn, Gn; check = false)
                @test isfinite(fo.beta)
                @test fo.ident.identified
            end
            # Two group sizes pass Proposition 1 but not Proposition 4,
            # so the same network is accepted one way and refused the
            # other -- the clearest statement that the fixed-effects
            # model asks for more.
            G2 = group_weights([5, 9])
            n2 = size(G2, 1)
            X2 = randn(rng, n2, 2); y2 = randn(rng, n2)
            @test fit_peer_effects(y2, X2, G2) isa PeerEffectsFit
            @test_throws ArgumentError fit_peer_effects(y2, X2, G2;
                                                        fixed_effects = true)
        end
    end

    @testset "37h. first-stage strength and the vcov options" begin
        let rng = MersenneTwister(606)
            n = 260
            Gsp = nw_er(rng, n, 0.015)      # sparse: strong instruments
            Gdn = nw_er(rng, n, 0.20)       # dense: weak ones
            mk = G -> begin
                Gd = Matrix(G); X = randn(rng, n, 2)
                y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                    (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
                (y, X)
            end
            ys, Xs = mk(Gsp); yd, Xd = mk(Gdn)
            fs = fit_peer_effects(ys, Xs, Gsp; check = false)
            fd = fit_peer_effects(yd, Xd, Gdn; check = false)
            # The direction is the finding; the magnitudes live in
            # docs/AUDIT.md.
            @test fs.first_stage_F > fd.first_stage_F
            @test fs.first_stage_F > 10

            # The three covariance options differ but share a point
            # estimate: only the uncertainty changes.
            fh = fit_peer_effects(ys, Xs, Gsp; check = false,
                                  vcov = :homoskedastic)
            @test coef(fh) ≈ coef(fs)
            @test stderror(fh) != stderror(fs)
            @test all(>(0), stderror(fs))
            @test all(>(0), stderror(fh))
            @test size(vcov(fs)) == (length(coef(fs)), length(coef(fs)))

            # :transform corrects for the correlation the within-local
            # transformation induces, so it is meaningless without it
            # and is refused by name.
            @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                        vcov = :transform)
            ft = fit_peer_effects(ys, Xs, Gsp; check = false,
                                  fixed_effects = true, vcov = :transform)
            @test all(>(0), stderror(ft))
            @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                        vcov = :bogus)
            @test_throws ArgumentError fit_peer_effects(ys, Xs, Gsp;
                                                        instrument_order = 4)
            @test_throws DimensionMismatch fit_peer_effects(ys, Xs[1:10, :], Gsp)
            @test_throws DimensionMismatch fit_peer_effects(ys, Xs, Gsp;
                                                            varnames = ["a"])
        end
    end

    @testset "37i. Sargan mechanics" begin
        let rng = MersenneTwister(1212)
            n = 240
            G = nw_er(rng, n, 0.02); Gd = Matrix(G)
            X = randn(rng, n, 2)
            y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))

            f2 = fit_peer_effects(y, X, G; instrument_order = 2)
            f3 = fit_peer_effects(y, X, G; instrument_order = 3)
            # K excluded instruments against one endogenous regressor
            # leaves K - 1 over-identifying restrictions.
            @test f2.sargan_df == 1
            @test f3.sargan_df == 3
            @test 0 <= f2.sargan_p <= 1
            @test f2.sargan >= 0
            # The statistic agrees with its own definition, computed here.
            let S = hcat(ones(n), X, Gd * X, Gd * Gd * X),
                u = residuals(f2)
                @test f2.sargan ≈ dot(u, S * (S \ u)) / (dot(u, u) / n) rtol = 1e-10
            end

            # With one characteristic the model is exactly identified at
            # order 2, and a Sargan test does not exist. Reporting df 0
            # beats reporting a p-value of 1 as though it meant something.
            f1 = fit_peer_effects(y, X[:, 1:1], G; instrument_order = 2)
            @test f1.sargan_df == 0
            @test f1.sargan == 0
        end
    end

    @testset "37j. multiplex reduces to the audited single-channel fit" begin
        let rng = MersenneTwister(8080)
            n = 150
            G = nw_er(rng, n, 0.06)
            X = randn(rng, n, 2)
            y = (Matrix(1.0I, n, n) - 0.5 * Matrix(G)) \
                (2.0 .+ X * [1.0, -0.7] .+ randn(rng, n))

            m1 = fit_multiplex_sar(y, X, [G])
            sar = fit_spatial(SAR(), G, X, y; method = ExactLU(),
                              max_exact_n = 10_000)
            # The reference is independently written, already-audited
            # code, not a rearrangement of the same expression. The
            # maximised log-likelihood is the sharp comparison; rho and
            # beta agree to the Brent tolerance both searches use.
            @test abs(m1.loglik - sar.loglik) < 1e-10
            @test abs(m1.rho[1] - sar.rho) < 1e-7
            @test maximum(abs, m1.beta .- sar.coef[1:(end - 1)]) < 1e-6
            @test abs(m1.rho_se[1] - sar.rho_se) < 1e-7
            @test maximum(abs, m1.beta_se .-
                          sqrt.(diag(sar.vcov))[1:(end - 1)]) < 1e-6
            @test m1.converged
            @test m1.channel_rank == 1
            @test nobs(m1) == n
            @test length(coefnames(m1)) == length(coef(m1))
        end
    end

    @testset "37k. multiplex coordinate ascent against a grid" begin
        # Coordinate ascent is exact along each axis but can stall on a
        # ridge. The profile likelihood is written out independently
        # here and maximised by brute force over the stationary region.
        let rng = MersenneTwister(2024)
            n = 120
            W1 = nw_er(rng, n, 0.05)
            W2 = nw_er(rng, n, 0.05)
            X = randn(rng, n, 2)
            A = Matrix(1.0I, n, n) - 0.35 * Matrix(W1) - 0.25 * Matrix(W2)
            y = A \ (1.0 .+ X * [1.0, -0.6] .+ randn(rng, n))

            Xa = hcat(ones(n), X)
            function prof(r1, r2)
                abs(r1) + abs(r2) >= 0.995 && return -Inf
                B = Matrix(1.0I, n, n) - r1 * Matrix(W1) - r2 * Matrix(W2)
                e = B * y
                u = e .- Xa * (Xa \ e)
                s2 = dot(u, u) / n
                return -0.5n * (log(2π) + 1) + logabsdet(B)[1] - 0.5n * log(s2)
            end

            m = fit_multiplex_sar(y, X, [W1, W2])
            step = 0.01
            best = -Inf
            for r1 in -0.95:step:0.95, r2 in -0.95:step:0.95
                v = prof(r1, r2)
                v > best && (best = v)
            end
            @test m.converged
            # The ascent must not be beaten by the grid beyond what the
            # grid spacing can resolve.
            @test m.loglik >= best - 1e-6
            @test abs(m.rho[1] - 0.35) < 0.25
            @test abs(m.rho[2] - 0.25) < 0.25
            @test m.loglik ≈ prof(m.rho[1], m.rho[2]) rtol = 1e-10
            @test m.channel_rank == 2
            @test sum(abs, m.rho) < 0.995
        end
    end

    @testset "37l. multiplex named failures" begin
        let rng = MersenneTwister(606)
            n = 60
            W1 = nw_er(rng, n, 0.08)
            X = randn(rng, n, 2); y = randn(rng, n)
            # Two identical channels cannot have their rhos told apart,
            # which is the same algebra as the peer-effects rank
            # condition and is refused the same way.
            @test_throws ArgumentError fit_multiplex_sar(y, X, [W1, W1])
            @test_throws ArgumentError fit_multiplex_sar(y, X, [W1, 2.0 .* W1])
            @test_throws ArgumentError fit_multiplex_sar(y, X, [])
            @test_throws DimensionMismatch fit_multiplex_sar(y, X,
                [W1, nw_er(rng, n + 1, 0.08)])
            @test_throws ArgumentError fit_multiplex_sar(y, X, [W1];
                                                         rho_bound = 1.5)
            # An intercept is added, so one in X is a rank failure
            # named by column rather than a singular matrix.
            @test_throws ArgumentError fit_multiplex_sar(y, hcat(ones(n), X), [W1])
        end
    end

    @testset "37m. residual dependence after a fit" begin
        let rng = MersenneTwister(4004)
            n = 200
            G = nw_er(rng, n, 0.02); Gd = Matrix(G)
            X = randn(rng, n, 2)
            y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
            f = fit_peer_effects(y, X, G)
            m = network_moran(f, G)
            @test m isa MoranResult
            @test isfinite(m.I_stat)
            @test -1.5 <= m.I_stat <= 1.5
            # A different channel may be passed, which is the point:
            # it asks whether dependence survives along a route the
            # model did not include.
            other = nw_er(rng, n, 0.02)
            @test network_moran(f, other) isa MoranResult
            @test_throws DimensionMismatch network_moran(f, nw_er(rng, n + 5, 0.02))
        end
    end

    @testset "37n. show methods run" begin
        let rng = MersenneTwister(5150), io = IOBuffer()
            n = 150
            G = nw_er(rng, n, 0.02); Gd = Matrix(G)
            X = randn(rng, n, 2)
            y = (Matrix(1.0I, n, n) - 0.4 * Gd) \
                (1.0 .+ X * [1.0, -0.5] .+ Gd * X * [0.8, 0.3] .+ randn(rng, n))
            show(io, peer_identification(G))
            show(io, peer_identification(group_weights([5, 5, 5])))
            show(io, fit_peer_effects(y, X, G))
            show(io, fit_multiplex_sar(y, X, [G]))
            s = String(take!(io))
            @test occursin("NOT IDENTIFIED", s)
            @test occursin("reflection problem", s)
            @test occursin("first-stage F", s)
        end
    end
end
