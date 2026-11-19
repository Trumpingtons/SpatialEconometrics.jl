@testset "32. Performance invariants" begin
    _progress("32. Performance invariants")
    # Correctness tests do not notice an accidental quadratic: the old
    # `row_standardize!` rescanned every nonzero in the matrix once per
    # row, so it was O(N * nnz) rather than O(nnz), and every assertion
    # about its output passed throughout. What catches that is a bound
    # on how long it may take.

    @testset "32a. row_standardize is linear in the nonzeros" begin
        for side in (20, 40)
            W = SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
            Wr = row_standardize(W)
            rs = vec(sum(Wr; dims = 2))
            @test all(x -> isapprox(x, 1.0; atol = 1e-12), rs)
            @test nnz(Wr) == nnz(W)
            @test W == SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
        end

        # A row with no neighbours stays zero rather than dividing by it.
        Wgap = sparse([1, 1, 2], [2, 3, 1], [1.0, 2.0, 3.0], 4, 4)
        Wg = row_standardize(Wgap)
        @test vec(sum(Wg; dims = 2)) ≈ [1.0, 1.0, 0.0, 0.0]

        # The row sum is of absolute values, so a signed W normalises by
        # magnitude -- which is what the original did, and what the
        # linear rewrite has to keep doing.
        Wneg = sparse([1, 2, 2], [2, 1, 3], [-1.0, 2.0, -3.0], 3, 3)
        @test nonzeros(row_standardize(Wneg)) ≈ [2 / 5, -1.0, -3 / 5]

        # The timing bound. At N = 8100 the quadratic version took 310 ms
        # and the linear one takes 0.19 ms, so 50 ms sits 260x above the
        # new cost and 6x below the old. That margin is what makes a
        # wall-clock assertion safe to run on a shared CI runner; a
        # tighter bound would be measuring the runner, not the code.
        Wbig = SparseMatrixCSC{Float64, Int}(make_rook_weights(90))
        @test size(Wbig, 1) == 8100
        row_standardize(Wbig)                      # compile first
        t = @elapsed row_standardize(Wbig)
        @test t < 0.05
    end

    @testset "32b. the probes-contiguous trace kernel" begin
        SU = SpatialEconometrics.SpatialUtils

        # The kernel is a hand-written gather, so it gets checked against
        # the thing it replaced rather than only against itself: the same
        # probe block, multiplied by `W` with `mul!`, and the same traces
        # read off by `dot`. Agreement has to be at roundoff and no
        # better, since the two sum the outer loop in different orders.
        for side in (6, 11), p in (1, 4, 37)
            W = SparseMatrixCSC{Float64, Int}(make_rook_weights(side))
            N = size(W, 1)
            mp = 12
            got = SU.precompute_neumann_traces(W; max_power = mp,
                                               n_probes = p,
                                               rng = MersenneTwister(4))
            # Reference: draw the probes column-major exactly as the
            # documented contract says, then walk the powers densely.
            V0 = Matrix{Float64}(undef, N, p)
            rand!(MersenneTwister(4), V0, (-1.0, 1.0))
            want = zeros(mp + 1)
            want[1] = 1.0
            want[2] = tr(W) / N
            Vc = W * V0
            for m in 2:mp
                Vc = W * Vc
                want[m + 1] = dot(V0, Vc) / (N * p)
            end
            @test got ≈ want rtol = 1e-10
            @test got[1] == 1.0
            @test got[2] == tr(W) / N
        end

        # The probe set is part of the contract: a caller who passes a
        # seed gets the column-major Rademacher draw, not whatever
        # layout happens to be fastest inside. If that ever changes,
        # every reported impact and standard error moves by a Monte
        # Carlo error, which is exactly the surprise this pins down.
        V = Matrix{Float64}(undef, 9, 5)
        rand!(MersenneTwister(13), V, (-1.0, 1.0))
        @test SU._rademacher_probes(Float64, 9, 5, MersenneTwister(13)) ==
              permutedims(V)
        @test all(x -> abs(x) == 1.0, V)

        # Same seed, same answer, bit for bit -- the reason the kernel
        # reduces over a per-row array in index order instead of letting
        # threads accumulate into one scalar.
        W2 = SparseMatrixCSC{Float64, Int}(make_rook_weights(14))
        a = SU.precompute_neumann_traces(W2; max_power = 20, n_probes = 8,
                                         rng = MersenneTwister(1))
        b = SU.precompute_neumann_traces(W2; max_power = 20, n_probes = 8,
                                         rng = MersenneTwister(1))
        @test all(a .=== b)

        # An asymmetric W, checked the same way. Worth stating why this
        # case needs the reference comparison and not a cleverer test:
        # tr((Wᵀ)ᵐ) = tr(Wᵐ) for every m, so a gather that walked the
        # matrix the wrong way round would still be an unbiased
        # estimator of the right traces and no test of the trace values
        # alone could see it. What pins the direction is agreeing with
        # `mul!(V, W, ...)` realisation for realisation.
        Wa = SparseMatrixCSC{Float64, Int}(
            sparse([1, 2, 3, 1, 4, 4], [2, 3, 1, 3, 1, 2],
                   [0.5, 0.25, 0.75, 0.5, 0.3, 0.7], 4, 4))
        @test !issymmetric(Matrix(Wa))
        p = 16
        got = SU.precompute_neumann_traces(Wa; max_power = 9, n_probes = p,
                                           rng = MersenneTwister(6))
        V0 = Matrix{Float64}(undef, 4, p)
        rand!(MersenneTwister(6), V0, (-1.0, 1.0))
        want = zeros(10)
        want[1] = 1.0
        want[2] = tr(Wa) / 4
        Vc = Wa * V0
        for m in 2:9
            Vc = Wa * Vc
            want[m + 1] = dot(V0, Vc) / (4 * p)
        end
        @test got ≈ want rtol = 1e-10
        # And it would not have agreed with the transpose: same probes,
        # a different realisation, even though the expectation is equal.
        Vt = permutedims(Matrix(Wa)) * V0
        Vt = permutedims(Matrix(Wa)) * Vt
        @test !isapprox(dot(V0, Vt) / (4 * p), got[4]; rtol = 1e-6)
    end

    @testset "32c. threaded GWR writes disjoint rows" begin
        # The three local loops in `fit_gwr` were threaded. Each writes
        # only row `i` of its output, and the reductions that follow --
        # `tr(S)` and `RSS` -- are serial passes afterwards, so the fit
        # is the same whatever the thread count. A test cannot vary the
        # thread count, so what it checks instead is the property that
        # makes that true: the threaded result equals the same arithmetic
        # done serially, here by rebuilding one local regression by hand.
        side = 12
        N = side^2
        coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
        lcg = Ref(UInt64(20251030))
        function draw()
            lcg[] = 6364136223846793005 * lcg[] + 1442695040888963407
            return Float64(lcg[] >> 11) / Float64(UInt64(1) << 53) * 2 - 1
        end
        X = reshape([draw() for _ in 1:(2N)], N, 2)
        y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* [draw() for _ in 1:N]

        m = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(4.0),
                    kernel = GaussianKernel())
        @test size(m.coef_local) == (N, 3)
        @test all(isfinite, m.coef_local)
        @test all(isfinite, m.se_local)
        @test all(x -> x > 0, m.se_local)

        # Row 7 of coef_local, rebuilt serially from the definition.
        i = 7
        Xa = hcat(ones(N), X)
        # Note the truncation: `_get_weights` evaluates the kernel only
        # at the points `inrange` returns, so a Gaussian kernel is cut
        # off hard at one bandwidth rather than decaying over all N. The
        # reference has to do the same or it is weighting a different
        # neighbourhood. That cut-off is the estimator's own existing
        # choice, not something this pass introduced.
        dist = [sqrt(sum((coords[i, :] .- coords[j, :]) .^ 2)) for j in 1:N]
        w = [d <= 4.0 ? exp(-0.5 * (d / 4.0)^2) : 0.0 for d in dist]
        @test count(!iszero, w) < N
        Wh = Diagonal(sqrt.(w))
        beta = qr(Wh * Xa) \ (Wh * y)
        @test m.coef_local[i, :] ≈ beta rtol = 1e-10

        # And local R-squared, which had its two loop-invariant vectors
        # hoisted out of the threaded loop. Same number, computed once.
        tss = dot(w, (y .- mean(y)) .^ 2)
        rss = dot(w, m.residuals .^ 2)
        @test m.local_R2[i] ≈ 1 - rss / tss rtol = 1e-10

        # `_euclid` replaced `sqrt(sum((a .- b) .^ 2))`, which it has to
        # match exactly and not merely closely, since it feeds the kernel.
        GW = SpatialEconometrics.GWREstimation
        for (a, b) in ((1, 2), (5, 99), (40, 40), (1, N))
            @test GW._euclid(coords, a, b) ===
                  sqrt(sum((coords[a, :] .- coords[b, :]) .^ 2))
        end
        # Three columns too: GWR asserts two, but the helper is generic.
        c3 = [0.5 -1.25 2.0; 3.0 0.125 -4.5; -1.0 1.0 1.0]
        @test GW._euclid(c3, 1, 2) ≈ sqrt(sum((c3[1, :] .- c3[2, :]) .^ 2))
    end

    @testset "32d. knn_weights keeps k non-self neighbours" begin
        # The selection used to re-derive, per candidate, how many
        # non-self neighbours the list held -- an allocating `filter` and
        # a `count` over a fresh slice inside the inner loop. A running
        # counter decides the same thing; the rule is what the test pins.
        grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
        for side in (7, 20), k in (1, 3, 12)
            W = knn_weights(grid(side), k; row_standardize = false)
            N = side^2
            @test size(W) == (N, N)
            @test all(==(k), vec(sum(W .!= 0; dims = 2)))
            @test all(iszero, diag(W))
            @test nnz(W) == N * k
        end

        # Coincident points: self can be absent from its own neighbour
        # list, which is the case the old `length(filter(...)) >= k`
        # guard existed to handle. Six points stacked at the origin.
        dup = vcat(zeros(6, 2), Float64[1 0; 0 1; 2 2; -1 -1; 3 0; 0 3])
        for k in (2, 5)
            W = knn_weights(dup, k; row_standardize = false)
            @test all(==(k), vec(sum(W .!= 0; dims = 2)))
            @test all(iszero, diag(W))
        end

        # binary, symmetrize and row_standardize still compose.
        Wb = knn_weights(grid(10), 4; binary = true, row_standardize = false)
        @test all(==(1.0), nonzeros(Wb))
        Wr = knn_weights(grid(10), 4)
        @test all(x -> isapprox(x, 1.0; atol = 1e-12), vec(sum(Wr; dims = 2)))
        Wsym = knn_weights(grid(10), 4; symmetrize = true, row_standardize = false)
        @test Wsym == permutedims(Wsym)
    end

    @testset "32e. the log-determinant sweep reuses its factorisation" begin
        SMS = SpatialEconometrics.SpatialModelSelection
        W = SparseMatrixCSC{Float64, Int}(make_rook_weights(14))
        N = size(W, 1)
        Isp = sparse(1.0I, N, N)

        # The closure writes values into a fixed pattern and calls `lu!`
        # instead of building `I - rho*W` and factorising from scratch.
        # Exactness is the point of this branch, so the standard here is
        # bit-identical and not merely close.
        f = SMS._logdet_closure(W, N, ExactLU(), typemax(Int))
        for rho in (-0.995, -0.4, 0.0, 0.17, 0.5, 0.9, 0.995)
            @test f(rho) === first(logabsdet(lu(Isp - rho * W)))
        end
        # Called repeatedly and out of order: the value buffer must be
        # rewritten each time, not accumulated into.
        @test f(0.5) === f(0.5)
        @test f(0.3) === first(logabsdet(lu(Isp - 0.3 * W)))

        # The pattern helper must hold the structural union of I and W,
        # including positions where `I - W` would cancel to exactly zero.
        Wone = sparse([1, 2, 2, 3], [2, 1, 2, 3], [1.0, 0.5, 1.0, 1.0], 3, 3)
        A, iv, wv = SMS._affine_pattern(Wone, 3)
        @test nnz(A) == 5          # three diagonal, plus (1,2) and (2,1)
        nzv = nonzeros(A)
        for rho in (0.0, 0.25, -0.75)
            for t in eachindex(nzv)
                nzv[t] = iv[t] - rho * wv[t]
            end
            @test Matrix(A) ≈ Matrix(1.0I(3) - rho * Wone)
        end
    end

    @testset "32f. the Chebyshev recursion has a memory ceiling" begin
        SU = SpatialEconometrics.SpatialUtils
        grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
        unbounded = typemax(Int) ÷ 4

        for (Wraw, q) in ((make_rook_weights(20), 6),
                          (make_rook_weights(20), 12),
                          (row_standardize(knn_weights(grid(14), 12)), 10))
            Ws = SparseMatrixCSC{Float64, Int}(Wraw)
            m = ChebyshevApprox(order = q, n_probes = 0)
            ref = SU.precompute_chebyshev_traces(Ws, m; max_fill_bytes = unbounded)
            # A ceiling of zero forces the blocked path from order 2 on.
            blocked = SU.precompute_chebyshev_traces(Ws, m; max_fill_bytes = 0)
            @test length(blocked) == q + 1
            @test blocked[1] == size(Ws, 1)       # tr(T_0) = tr(I) = N
            @test blocked[2] == tr(Ws)
            # Same traces to roundoff: the two sum over columns in
            # different groupings, so this is reordering rather than a
            # different quantity.
            @test blocked ≈ ref rtol = 1e-10
            # What actually matters is the log-determinant it feeds.
            for rho in (-0.9, -0.25, 0.0, 0.45, 0.9)
                @test SU.chebyshev_logdet(rho, blocked) ≈
                      SU.chebyshev_logdet(rho, ref) atol = 1e-8
            end
            # A generous ceiling must leave the fast path untouched, bit
            # for bit: the default is not allowed to perturb anything.
            @test all(SU.precompute_chebyshev_traces(Ws, m) .=== ref)
        end

        # Block size: at least 16 columns, never more than N, otherwise
        # as many as three dense N-vectors per column allow.
        @test SU._cheb_block_size(1000, 0, Float64) == 16
        @test SU._cheb_block_size(1000, typemax(Int) ÷ 4, Float64) == 1000
        @test SU._cheb_block_size(1000, 3 * 1000 * 8 * 50, Float64) == 50

        # A block size that does not divide N must leave no column out:
        # the trace is a sum over all N, so a short final block would
        # show up as a wrong answer rather than as an error.
        W11 = SparseMatrixCSC{Float64, Int}(make_rook_weights(11))   # N = 121
        m5 = ChebyshevApprox(order = 5, n_probes = 0)
        @test SU.precompute_chebyshev_traces(W11, m5; max_fill_bytes = 0) ≈
              SU.precompute_chebyshev_traces(W11, m5;
                                             max_fill_bytes = unbounded) rtol = 1e-10

        # The probe branch is untouched by any of this.
        a = SU.precompute_chebyshev_traces(W11,
                ChebyshevApprox(order = 6, n_probes = 8, rng = MersenneTwister(5));
                max_fill_bytes = 0)
        b = SU.precompute_chebyshev_traces(W11,
                ChebyshevApprox(order = 6, n_probes = 8, rng = MersenneTwister(5)))
        @test all(a .=== b)
    end
end
