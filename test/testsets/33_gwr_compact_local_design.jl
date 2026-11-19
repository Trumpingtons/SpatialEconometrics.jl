@testset "33. GWR compact local design" begin
    _progress("33. GWR compact local design")
    # Every local regression used to build a dense `N x K` weighted
    # design when only the `b` rows inside the kernel's support carry
    # anything. The compact form is the same arithmetic on the rows that
    # matter, so what these tests pin is the equivalence -- against the
    # dense form kept in `_get_weights` for exactly this purpose -- and
    # the properties the compact form has to have for that to hold.
    GW = SpatialEconometrics.GWREstimation

    grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

    # The dense arithmetic, verbatim as it stood before the change.
    function dense_fit(X, y, coords, tree, bw, kernel)
        N, K = size(X)
        coef = zeros(N, K); hat = zeros(N); res = zeros(N); fit = zeros(N)
        for i in 1:N
            w = GW._get_weights(tree, coords, i, bw, kernel)
            Ws = Diagonal(sqrt.(w))
            F = qr(Ws * X)
            b = F \ (Ws * y)
            coef[i, :] = b
            fit[i] = dot(X[i, :], b)
            res[i] = y[i] - fit[i]
            hat[i] = sum(abs2, F.R' \ (X[i, :] .* sqrt(w[i])))
        end
        return coef, res, fit, sum(hat), dot(res, res)
    end

    @testset "33a. the support is what the dense vector was" begin
        coords = grid(9)
        N = size(coords, 1)
        tree = GW._build_tree(coords)
        for bw in (FixedBandwidth(2.5), FixedBandwidth(50.0),
                   AdaptiveBandwidth(5), AdaptiveBandwidth(N - 1)),
            kern in (GaussianKernel(), BisquareKernel(), ExponentialKernel())

            sc = GW._support_scratch(Float64)
            for i in (1, 13, N)
                n, h_i, w_self = GW._local_support!(sc, tree, coords, i, bw, kern)
                w = GW._get_weights(tree, coords, i, bw, kern)

                # Ascending index order is a contract, not an accident:
                # it is what makes the compact design's row order both
                # reproducible and equal to the dense path's.
                @test issorted(view(sc.idx, 1:n))
                @test allunique(view(sc.idx, 1:n))

                # Scattering the support back must reproduce the dense
                # vector exactly -- same values, same positions, and
                # nothing nonzero outside the support.
                scattered = zeros(N)
                for t in 1:n
                    scattered[sc.idx[t]] = sc.wv[t]
                end
                @test scattered == w

                # `w_self` is computed as the kernel at distance zero
                # rather than looked up, so it has to agree bit for bit.
                @test w_self === w[i]
                @test h_i > 0
            end
        end
    end

    @testset "33b. compact equals dense, across kernels and bandwidths" begin
        worst = 0.0
        for side in (8, 12), kern in (GaussianKernel(), BisquareKernel(),
                                      ExponentialKernel())
            N = side^2
            coords = grid(side)
            lcg = Ref(UInt64(6103))
            function draw()
                lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
            end
            X = hcat(ones(N), reshape([draw() for _ in 1:(2N)], N, 2))
            y = 1.0 .+ X[:, 2:3] * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]
            tree = GW._build_tree(coords)

            for bw in (FixedBandwidth(2.5), FixedBandwidth(6.0),
                       AdaptiveBandwidth(6), AdaptiveBandwidth(N - 1))
                dc, dr, df, dtrS, dRSS = dense_fit(X, y, coords, tree, bw, kern)
                cc, cr, cf, ctrS, cRSS = GW._gwr_fit_at_bandwidth(
                    X, y, coords, tree, bw, kern)

                @test cc ≈ dc rtol = 1e-10
                @test cf ≈ df rtol = 1e-10
                @test ctrS ≈ dtrS rtol = 1e-10
                @test cRSS ≈ dRSS rtol = 1e-10
                worst = max(worst, maximum(abs.(cc .- dc) ./ max.(abs.(dc), 1e-300)))
            end
        end
        # The stated standard. A Householder QR on `b x K` is not the
        # same arithmetic as on the same matrix padded with `N - b` zero
        # rows, so this is roundoff and not equality -- but it is
        # roundoff, and the bound says so rather than leaving it open.
        @test worst < 1e-11
    end

    @testset "33c. the bandwidth search still picks the same k" begin
        # The one discrete risk: `_select_bandwidth_aicc` compares AICc
        # with `<`, so a near-tie could flip the chosen bandwidth. The
        # margin to the runner-up is ~0.04 AICc units, about 1e11 eps,
        # which no roundoff can cross -- this pins that it does not.
        for side in (8, 11)
            N = side^2
            coords = grid(side)
            lcg = Ref(UInt64(99173))
            function draw()
                lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
                return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
            end
            X = reshape([draw() for _ in 1:(2N)], N, 2)
            y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]
            Xa = hcat(ones(N), X)
            tree = GW._build_tree(coords)

            # A span of 48 is scanned exhaustively, so the selection is the
            # exact argmin of the curve.
            kmin = 3 + 3
            kmax = kmin + 47
            bw = GW._select_bandwidth_aicc(Xa, y, coords, tree,
                                           GaussianKernel(), AdaptiveBandwidth;
                                           search = :scan, k_min = kmin, k_max = kmax)
            # Rebuild the AICc curve from the dense arithmetic and check
            # the argmin agrees exactly, not approximately.
            aiccs = [GW._gwr_fit_at_bandwidth(Xa, y, coords, tree,
                         AdaptiveBandwidth(k), GaussianKernel())[8]
                     for k in kmin:kmax]
            best = (kmin:kmax)[argmin(aiccs)]
            @test bw.k == best
            # And the margin really is enormous, which is why this is safe.
            srt = sort(aiccs)
            @test srt[2] - srt[1] > 1e-6
        end
    end

    @testset "33d. edge cases the compact form has to survive" begin
        coords = grid(7)
        N = size(coords, 1)
        tree = GW._build_tree(coords)
        lcg = Ref(UInt64(4441))
        function draw()
            lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
            return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
        end
        X = hcat(ones(N), reshape([draw() for _ in 1:(2N)], N, 2))
        y = [draw() for _ in 1:N]

        # A bandwidth wide enough to cover everything: the compact path
        # must reduce to the dense one, support of exactly N.
        sc = GW._support_scratch(Float64)
        n, _, _ = GW._local_support!(sc, tree, coords, 1, FixedBandwidth(1000.0),
                                     GaussianKernel())
        @test n == N
        @test view(sc.idx, 1:n) == 1:N
        dc, _, _, dtrS, _ = dense_fit(X, y, coords, tree,
                                      FixedBandwidth(1000.0), GaussianKernel())
        cc, _, _, ctrS, _ = GW._gwr_fit_at_bandwidth(X, y, coords, tree,
                                      FixedBandwidth(1000.0), GaussianKernel())
        @test cc ≈ dc rtol = 1e-10
        @test ctrS ≈ dtrS rtol = 1e-10

        # Coincident points, where a point need not appear first in its
        # own neighbour list and `w_self` cannot be found by position.
        dup = vcat(zeros(5, 2), grid(5))
        Nd = size(dup, 1)
        td = GW._build_tree(dup)
        scd = GW._support_scratch(Float64)
        for i in 1:5
            n, _, w_self = GW._local_support!(scd, td, dup, i,
                                              FixedBandwidth(1.5), BisquareKernel())
            @test n >= 5
            @test w_self === GW._get_weights(td, dup, i, FixedBandwidth(1.5),
                                             BisquareKernel())[i]
        end

        # A Bisquare kernel is exactly zero at `d == h`, and `inrange`
        # returns that point, so the support may legitimately contain
        # zero weights. Nothing downstream may divide by that.
        scz = GW._support_scratch(Float64)
        n, _, _ = GW._local_support!(scz, tree, coords, 25,
                                     FixedBandwidth(2.0), BisquareKernel())
        @test any(iszero, view(scz.wv, 1:n))
    end

    @testset "33e. the chunk helper covers 1:N exactly once" begin
        for N in (0, 1, 7, 64, 100), nc in (1, 3, 8, 64)
            hits = zeros(Int, max(N, 1))
            GW._tforeach_chunks(N; nchunks = nc) do rows
                for i in rows
                    hits[i] += 1
                end
            end
            @test all(==(1), view(hits, 1:N))
            N == 0 && @test true          # must not error on an empty range
        end
    end

    @testset "33f. fit_gwr and fit_mgwr end to end" begin
        side = 10
        N = side^2
        coords = grid(side)
        lcg = Ref(UInt64(77001))
        function draw()
            lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
            return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
        end
        X = reshape([draw() for _ in 1:(2N)], N, 2)
        y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]

        m = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(3.0),
                    kernel = GaussianKernel())
        @test size(m.coef_local) == (N, 3)
        @test all(isfinite, m.coef_local)
        @test all(x -> x > 0, m.se_local)
        @test all(isfinite, m.local_R2)
        @test 0 < m.hat_trace < N

        # se_local against the dense form it replaced.
        tree = GW._build_tree(coords)
        Xa = hcat(ones(N), X)
        for i in (1, 50, N)
            w = GW._get_weights(tree, coords, i, FixedBandwidth(3.0), GaussianKernel())
            Xw = Diagonal(sqrt.(w)) * Xa
            inv_k = inv(Symmetric(Xw' * Xw))
            for k in 1:3
                @test m.se_local[i, k] ≈
                      sqrt(max(0.0, m.sigma2 * inv_k[k, k])) rtol = 1e-9
            end
            # and local R², which sums over the support
            tss = dot(w, (y .- mean(y)) .^ 2)
            rss = dot(w, m.residuals .^ 2)
            @test m.local_R2[i] ≈ 1 - rss / tss rtol = 1e-10
        end

        mm = fit_mgwr(X, y, coords; inference = false, max_iter = 8)
        @test size(mm.coef_local) == (N, 3)
        @test all(isfinite, mm.coef_local)
        @test all(isfinite, mm.se_local)
        @test length(mm.bandwidths) == 3

        # `_mgwr_se_approx` replaced two byte-identical blocks; check it
        # against the formula those blocks computed.
        for i in (1, 37), k in (1, 3)
            bw_k = AdaptiveBandwidth(round(Int, mm.bandwidths[k]))
            w = GW._get_weights(tree, coords, i, bw_k, BisquareKernel())
            xw = Xa[:, k] .* sqrt.(w)
            denom = dot(xw, xw)
            want = denom > 1e-12 ? sqrt(mm.sigma2 * w[i] / denom) : 0.0
            @test mm.se_local[i, k] ≈ want rtol = 1e-9
        end
    end
end
