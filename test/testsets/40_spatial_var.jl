# ═══════════════════════════════════════════════════════════════════
# 40. Spatial VAR — Di Giacinto (2010)
# ═══════════════════════════════════════════════════════════════════

@testset "40. Spatial VAR" begin
    _progress("40. Spatial VAR")
    SV = SpatialEconometrics.SpVAR

    # A ring is regular, so row-standardising keeps W⁽¹⁾ symmetric —
    # which the outward/inward STIR gate needs. See 40g.
    function sv_ring(n)
        I_ = Int[]; J_ = Int[]
        for i in 1:n
            push!(I_, i); push!(J_, mod1(i + 1, n))
            push!(I_, i); push!(J_, mod1(i - 1, n))
        end
        return sparse(I_, J_, ones(2n), n, n)
    end

    function sv_path(n)
        I_ = Int[]; J_ = Int[]
        for i in 1:(n - 1)
            push!(I_, i); push!(J_, i + 1)
            push!(I_, i + 1); push!(J_, i)
        end
        return sparse(I_, J_, ones(2 * (n - 1)), n, n)
    end

    SV_R1, SV_R2 = 0.35, -0.20
    SV_PS = (0.40, 0.15)
    SV_A = ((0.30, 0.10), (0.15, -0.05), (-0.10, 0.08), (0.25, 0.12))

    # A homogeneous SpVAR(1,1) with K = 2, built from the same W⁽ˡ⁾ the
    # estimator will use. `spatial = false` zeroes every spatial
    # coefficient, giving the nested panel VAR of eqs. (15)–(18).
    function sv_sim(rng, Wl, N, T; burn = 150, spatial = true)
        Wd = Matrix(Wl[2]); Im = Matrix(1.0I, N, N); Z = zeros(N, N)
        C0, C1 = if spatial
            ([Im - SV_R1 * Wd  Z;
              -(SV_PS[1] * Im + SV_PS[2] * Wd)  Im - SV_R2 * Wd],
             [SV_A[1][1] * Im + SV_A[1][2] * Wd   SV_A[2][1] * Im + SV_A[2][2] * Wd;
              SV_A[3][1] * Im + SV_A[3][2] * Wd   SV_A[4][1] * Im + SV_A[4][2] * Wd])
        else
            ([Im Z; -SV_PS[1] * Im Im],
             [SV_A[1][1] * Im  SV_A[2][1] * Im;
              SV_A[3][1] * Im  SV_A[4][1] * Im])
        end
        F = lu(C0); n = 2N
        y = zeros(n); Y = Matrix{Float64}(undef, N * T, 2)
        for t in 1:(T + burn)
            y = F \ (C1 * y + randn(rng, n))
            if t > burn
                tt = t - burn
                for i in 1:N
                    Y[(tt - 1) * N + i, 1] = y[i]
                    Y[(tt - 1) * N + i, 2] = y[N + i]
                end
            end
        end
        return Y, C0, C1
    end

    @testset "40a. order-l weights are not powers of W" begin
        # Hand-checkable on a path: order l has exactly 2(n-l) ordered
        # pairs, since |i-j| = l for i = 1..n-l in each direction.
        for n in (5, 9, 12)
            G = sv_path(n)
            for l in 1:(n - 1)
                Wl = order_weights(G, l; row_standardize = false)
                @test nnz(Wl) == 2 * (n - l)
                @test all(iszero, diag(Matrix(Wl)))
            end
            # The orders partition every ordered pair exactly once.
            tot = sum(nnz(order_weights(G, l; row_standardize = false))
                      for l in 1:(n - 1))
            @test tot == n * (n - 1)
        end
        # order 0 is the identity, whatever W is.
        @test order_weights(sv_path(7), 0) == sparse(1.0I, 7, 7)

        # And the trap: W² is a different object. On a 4x4 rook lattice
        # it is non-zero on the diagonal and has more non-zeros.
        let nx = 4, ny = 4, I_ = Int[], J_ = Int[]
            idx(i, j) = (j - 1) * nx + i
            for i in 1:nx, j in 1:ny, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                a, b = i + di, j + dj
                if 1 <= a <= nx && 1 <= b <= ny
                    push!(I_, idx(i, j)); push!(J_, idx(a, b))
                end
            end
            A = sparse(I_, J_, ones(length(I_)), 16, 16)
            P = A * A
            O = order_weights(A, 2; row_standardize = false)
            @test count(!iszero, diag(Matrix(P))) == 16
            @test count(!iszero, diag(Matrix(O))) == 0
            @test nnz(P) > nnz(O)
            # W² keeps first-order neighbours where triangles exist; on
            # a bipartite lattice there are none, so test the claim on a
            # graph that has one.
            let G = sparse([1, 2, 1, 3, 2, 3, 1, 4],
                           [2, 1, 3, 1, 3, 2, 4, 1], ones(8), 4, 4),
                Gp = G * G,
                G2 = order_weights(G, 2; row_standardize = false)
                kept_p = count(ij -> G[ij...] != 0 && Gp[ij...] != 0,
                               [(i, j) for i in 1:4, j in 1:4 if i != j])
                kept_o = count(ij -> G[ij...] != 0 && G2[ij...] != 0,
                               [(i, j) for i in 1:4, j in 1:4 if i != j])
                @test kept_p == 6
                @test kept_o == 0
            end
        end
        # A disconnected pair has no order-2 link, so the list refuses
        # rather than handing back a structurally empty W⁽²⁾.
        @test_throws ArgumentError order_weights_list(
            sparse([1, 2], [2, 1], ones(2), 2, 2), 2)
        @test_throws ArgumentError order_weights(sv_path(5), -1)
    end

    # `binary = false` weights each order-l link by 1/d, where d is
    # the number of distinct SHORTEST paths. On a rook lattice every
    # shortest path is monotone, so d between (1,1) and (a+1,b+1) is
    # exactly the binomial C(a+b, a) -- an independent closed form,
    # not a tolerance. Checked because the keyword is documented, and
    # a documented keyword no run has exercised is the thing this
    # project's standard exists to prevent.
    @testset "40a2. binary = false counts shortest paths" begin
        G = let nx = 4, ny = 4
            n = nx * ny
            I_ = Int[]
            J_ = Int[]
            idx(i, j) = (j - 1) * nx + i
            for i in 1:nx, j in 1:ny,
                (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                a, b = i + di, j + dj
                if 1 <= a <= nx && 1 <= b <= ny
                    push!(I_, idx(i, j))
                    push!(J_, idx(a, b))
                end
            end
            sparse(I_, J_, ones(length(I_)), n, n)
        end
        gidx(i, j) = (j - 1) * 4 + i
        nchoosek(n, k) = factorial(n) ÷ (factorial(k) * factorial(n - k))
        for (a, b) in ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
            Wl = order_weights(G, a + b; row_standardize = false,
                               binary = false)
            @test isapprox(Wl[gidx(1, 1), gidx(a + 1, b + 1)],
                           1 / nchoosek(a + b, a); atol = 1e-12)
        end
        # A 4-ring: the opposite unit is reached by exactly two
        # shortest paths, so the weight is 1/2 and there are N of them.
        let W2 = order_weights(sv_ring(4), 2; row_standardize = false,
                               binary = false)
            @test isapprox(W2[1, 3], 0.5; atol = 1e-14)
            @test nnz(W2) == 4
        end
        # A 6-ring: order 2 is single-path, order 3 is the diameter and
        # is reached two ways.
        let W2 = order_weights(sv_ring(6), 2; row_standardize = false,
                               binary = false),
            W3 = order_weights(sv_ring(6), 3; row_standardize = false,
                               binary = false)
            @test isapprox(W2[1, 3], 1.0; atol = 1e-14)
            @test isapprox(W2[1, 5], 1.0; atol = 1e-14)
            @test isapprox(W3[1, 4], 0.5; atol = 1e-14)
        end
        # The two settings differ only in the VALUES: same pattern,
        # and `binary = true` is all ones before standardisation.
        for l in 1:6
            Wb = order_weights(G, l; row_standardize = false,
                               binary = true)
            Wp = order_weights(G, l; row_standardize = false,
                               binary = false)
            @test nnz(Wb) == nnz(Wp)
            @test Wb.rowval == Wp.rowval
            @test all(isone, nonzeros(Wb))
        end
    end

    @testset "40b. the order condition reproduces the paper" begin
        # Di Giacinto's own worked numbers, for the heterogeneous
        # parameterisation his eq. (13) counts: K=2 needs N≥3 at s=1
        # and N≥4 at s=2. Integer bounds, not tolerances.
        for (K, s, want) in ((2, 1, 3), (2, 2, 4))
            @test spvar_order_condition(want, K, s;
                                        homogeneous = false).n_min == want
            @test spvar_order_condition(want, K, s;
                                        homogeneous = false).satisfied
            @test !spvar_order_condition(want - 1, K, s;
                                         homogeneous = false).satisfied
        end
        # The free-coefficient count derived here, N(s+1)K(K-1)/2 + NsK,
        # must equal the paper's printed NK(K-1)/2 + sNK(K+1)/2.
        for N in (3, 5, 12, 40), K in (2, 3, 4), s in (1, 2, 3)
            @test SV._n_free_c0(N, K, s, false) ==
                  div(N * K * (K - 1), 2) + div(s * N * K * (K + 1), 2)
        end
        # The bound falls as K rises, which is the paper's remark.
        @test spvar_order_condition(3, 3, 2; homogeneous = false).n_min <=
              spvar_order_condition(3, 2, 2; homogeneous = false).n_min
        # Homogeneous coefficients do not scale with N, so the
        # condition binds only for tiny N.
        @test spvar_order_condition(4, 2, 1; homogeneous = true).satisfied
        @test spvar_order_condition(50, 2, 1; homogeneous = true).n_free <
              spvar_order_condition(50, 2, 1; homogeneous = false).n_free
        @test_throws ArgumentError spvar_order_condition(0, 2, 1)
        @test occursin("order condition",
                       sprint(show, spvar_order_condition(12, 2, 1)))
    end

    @testset "40c. |C₀| factors over the diagonal blocks" begin
        # This identity (from block triangularity) is what makes the
        # concentrated likelihood separable, so it is pinned exactly.
        rng = MersenneTwister(40_301)
        N, T = 10, 200
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
        blk(k) = ((k - 1) * N + 1):(k * N)
        full = logabsdet(m.C0)[1]
        parts = sum(logabsdet(m.C0[blk(k), blk(k)])[1] for k in 1:m.K)
        @test isapprox(full, parts; rtol = 1e-12, atol = 1e-12)
        # C₀ is block LOWER triangular: the (1,2) block is exactly zero.
        @test all(iszero, m.C0[blk(1), blk(2)])
        @test !all(iszero, m.C0[blk(2), blk(1)])
    end

    @testset "40d. the concentrated likelihood is separable" begin
        rng = MersenneTwister(40_401)
        N, T = 10, 200
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
        cst = -(N * m.K * m.T_eff) / 2 * log(2π) - (N * m.K * m.T_eff) / 2
        @test isapprox(sum(m.loglik_eq) + cst, m.loglik;
                       rtol = 1e-12, atol = 1e-10)
        @test length(m.loglik_eq) == m.K
    end

    @testset "40e. s = 0 is the nested panel VAR" begin
        # fit_nested_panel_var is written independently of fit_spvar —
        # no shared solver — so this is a real cross-check, not a
        # tautology. With s = 0 every W⁽ˡ⁾ block collapses to W⁽⁰⁾ = I,
        # A⁽⁰⁾ₖₖ = I and the Jacobian vanishes: eqs. (15)-(18) exactly.
        rng = MersenneTwister(40_501)
        N, T = 9, 180
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m0 = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 0)
        nv = fit_nested_panel_var(Y, N, T; ordering = [1, 2],
                                  homogeneous = true)
        @test isapprox(m0.loglik, nv.loglik; rtol = 1e-10, atol = 1e-10)
        for k in 1:2
            @test isapprox(m0.theta[k], vec(nv.coefs[k]);
                           rtol = 1e-8, atol = 1e-10)
        end
        @test all(isempty, m0.rho)
        # No Jacobian at s = 0: log|C₀| = 0 because C₀ is unit
        # triangular in the blocks.
        @test isapprox(logabsdet(m0.C0)[1], 0.0; atol = 1e-12)
        # The heterogeneous nested VAR has MORE parameters, so it is not
        # nested in the homogeneous SpVAR and must not be LR-tested
        # against it. Recorded as an assertion so the claim is pinned.
        let nvh = fit_nested_panel_var(Y, N, T; ordering = [1, 2],
                                       homogeneous = false)
            @test nvh.n_params > nv.n_params
            @test nvh.loglik >= nv.loglik
        end
        @test occursin("Nested panel VAR", sprint(show, nv))
    end

    @testset "40f. IRF sums to the long-run multiplier" begin
        rng = MersenneTwister(40_601)
        N, T = 10, 200
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
        @test m.stationary
        psi = spvar_irf(m; horizon = 400)
        @test psi[1] ≈ inv(m.C0)
        acc = zeros(2N, 2N)
        for P in psi
            acc .+= P
        end
        @test isapprox(acc, inv(m.C0 - m.C[1]); rtol = 1e-8, atol = 1e-10)
        # Convergence is monotone in the horizon, not accidental.
        errs = Float64[]
        for H in (5, 20, 80)
            a = zeros(2N, 2N)
            for P in spvar_irf(m; horizon = H)
                a .+= P
            end
            push!(errs, maximum(abs, a .- inv(m.C0 - m.C[1])))
        end
        @test issorted(errs; rev = true)
        @test_throws ArgumentError spvar_irf(m; horizon = -1)
    end

    @testset "40g. outward and inward STIR" begin
        # They coincide exactly when W⁽ˡ⁾ is symmetric and K = 1, since
        # the blocks are then commuting polynomials in W. A RING keeps
        # that under row standardisation; a PATH does not, and that is
        # asserted too rather than assumed.
        rng = MersenneTwister(40_701)
        N, T = 12, 300
        for (regular, Wl) in ((true, order_weights_list(sv_ring(N), 1)),
                              (false, order_weights_list(sv_path(N), 1)))
            W1 = Wl[2]
            asym = maximum(abs, Matrix(W1) .- Matrix(W1)')
            @test (asym <= 1e-12) == regular
            Wd = Matrix(W1); Im = Matrix(1.0I, N, N)
            C0 = Im - 0.3 * Wd; C1 = 0.3 * Im + 0.1 * Wd
            F = lu(C0); y = zeros(N)
            Y1 = Matrix{Float64}(undef, N * T, 1)
            for t in 1:(T + 150)
                y = F \ (C1 * y + randn(rng, N))
                if t > 150
                    for i in 1:N
                        Y1[(t - 151) * N + i, 1] = y[i]
                    end
                end
            end
            m1 = fit_spvar(Y1, W1, N, T; ordering = [1], s = 1, Wl = Wl)
            st = space_time_stir(m1; horizon = 6)
            gap = maximum(abs, st.outward .- st.inward)
            if regular
                @test gap < 1e-10
            else
                @test gap > 1e-4
            end
        end
        # Order 0 uses W⁽⁰⁾ = I, so the outward STIR there is the mean
        # diagonal of the response block: the own-location effect.
        let Wl = order_weights_list(sv_ring(10), 1)
            Y, = sv_sim(MersenneTwister(40_702), Wl, 10, 200)
            m = fit_spvar(Y, Wl[2], 10, 200; ordering = [1, 2], s = 1,
                          Wl = Wl)
            st = space_time_stir(m; horizon = 3)
            psi = spvar_irf(m; horizon = 3)
            blk(k) = ((k - 1) * 10 + 1):(k * 10)
            for h in 0:3, k in 1:2, r in 1:2
                @test isapprox(st.outward[h + 1, 1, k, r],
                    mean(diag(psi[h + 1][blk(k), blk(r)]));
                    rtol = 1e-10, atol = 1e-12)
            end
            # Cumulative accumulates; horizon 0 is unchanged.
            let sc = space_time_stir(m; horizon = 5, cumulative = true)
                @test sc.outward[1, :, :, :] ≈ st.outward[1, :, :, :]
                @test isapprox(sc.outward[3, 2, 1, 1],
                    sum(st.outward[1:3, 2, 1, 1]);
                    rtol = 1e-10, atol = 1e-12)
            end
            # store_local averages to the global version, eqs. (31)-(32).
            let sl = space_time_stir(m; horizon = 2, store_local = true)
                @test size(sl.local_outward, 5) == 10
                for h in 0:2, k in 1:2, r in 1:2, l in 1:2
                    @test isapprox(sl.outward[h + 1, l, k, r],
                        mean(sl.local_outward[h + 1, l, k, r, :]);
                        rtol = 1e-10, atol = 1e-12)
                end
            end
            @test occursin("STIR", sprint(show, st))
        end
    end

    @testset "40h. forecasts, and y_history is required" begin
        rng = MersenneTwister(40_801)
        N, T, H = 10, 250, 4
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T + H)
        m = fit_spvar(Y[1:(N * T), :], Wl[2], N, T; ordering = [1, 2],
                      s = 1, Wl = Wl)
        hist = Y[(N * (T - m.p) + 1):(N * T), :]
        f = spvar_forecast(m, H; y_history = hist)
        @test size(f) == (N * H, 2)
        @test all(isfinite, f)
        # Beats a no-change forecast on this stationary DGP.
        act = Y[(N * T + 1):(N * (T + H)), :]
        naive = repeat(hist, H, 1)
        @test sqrt(mean(abs2, f .- act)) < sqrt(mean(abs2, naive .- act))
        @test_throws ArgumentError spvar_forecast(m, H)
        @test_throws ArgumentError spvar_forecast(m, 0;
                                                  y_history = hist)
        @test_throws DimensionMismatch spvar_forecast(m, H;
            y_history = hist[:, 1:1])
        # One step ahead is exactly the reduced form applied to history.
        let f1 = spvar_forecast(m, 1; y_history = hist),
            Ct = lu(m.C0) \ m.C[1],
            v = Vector{Float64}(undef, 2N)
            for k in 1:2, i in 1:N
                v[(k - 1) * N + i] =
                    hist[i, m.ordering[k]] - m.means[i, k]
            end
            pred = Ct * v
            for k in 1:2, i in 1:N
                @test isapprox(f1[i, m.ordering[k]],
                    pred[(k - 1) * N + i] + m.means[i, k];
                    rtol = 1e-10, atol = 1e-12)
            end
        end
    end

    @testset "40i. the ordering is an assumption, and it binds" begin
        rng = MersenneTwister(40_901)
        N, T = 12, 300
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        # No default: identification of C₀ rests on it.
        @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T)
        @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
                                             ordering = [1, 1])
        @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
                                             ordering = [1, 2, 3])
        # Names work, and agree with the equivalent integer ordering.
        let ma = fit_spvar(Y, Wl[2], N, T; ordering = [2, 1], s = 1,
                           Wl = Wl, varnames = ["g", "y"]),
            mb = fit_spvar(Y, Wl[2], N, T; ordering = ["y", "g"], s = 1,
                           Wl = Wl, varnames = ["g", "y"])
            @test ma.ordering == mb.ordering == [2, 1]
            @test isapprox(ma.loglik, mb.loglik; rtol = 1e-12)
            @test ma.varnames == ["y", "g"]
        end
        @test_throws ArgumentError fit_spvar(Y, Wl[2], N, T;
            ordering = ["g", "nope"], varnames = ["g", "y"])
        # Reversing it changes the answer. The magnitude is measured in
        # docs/AUDIT.md; here only that it is not a rounding error.
        let m1 = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1,
                           Wl = Wl),
            m2 = fit_spvar(Y, Wl[2], N, T; ordering = [2, 1], s = 1,
                           Wl = Wl)
            @test abs(m1.loglik - m2.loglik) > 1.0
            s1 = space_time_stir(m1; horizon = 8)
            s2 = space_time_stir(m2; horizon = 8)
            @test maximum(abs, s1.outward) != maximum(abs, s2.outward)
        end
    end

    @testset "40j. refusals are named" begin
        rng = MersenneTwister(41_001)
        N, T = 10, 120
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        W1 = Wl[2]
        @test_throws DimensionMismatch fit_spvar(Y, W1, N, T + 1;
                                                  ordering = [1, 2])
        @test_throws DimensionMismatch fit_spvar(Y, sv_ring(N + 1), N, T;
                                                  ordering = [1, 2])
        @test_throws ArgumentError fit_spvar(Y, W1, N, T;
                                              ordering = [1, 2], p = 0)
        @test_throws ArgumentError fit_spvar(Y, W1, N, T;
                                              ordering = [1, 2], s = -1)
        # T_per > p + 1, with Y shaped to match so the row check
        # does not fire first and mask it.
        @test_throws ArgumentError fit_spvar(Y[1:(N * 2), :], W1, N, 2;
                                              ordering = [1, 2])
        @test_throws DimensionMismatch fit_spvar(Y, W1, N, T;
            ordering = [1, 2], s = 1, Wl = Wl[1:1])
        let Ybad = copy(Y); Ybad[1, 1] = Inf
            @test_throws ArgumentError fit_spvar(Ybad, W1, N, T;
                                                  ordering = [1, 2])
        end
        @test_throws DimensionMismatch fit_spvar(Y, W1, N, T;
            ordering = [1, 2], varnames = ["only one"])
        # The order condition refuses before any estimation happens.
        @test_throws ArgumentError fit_spvar(randn(2 * 40, 2),
            sparse([1, 2], [2, 1], ones(2), 2, 2), 2, 40;
            ordering = [1, 2], s = 2)
        # s = 0 has nothing to test for spatial interaction.
        let m0 = fit_spvar(Y, W1, N, T; ordering = [1, 2], s = 0)
            @test_throws ArgumentError lr_spatial_interaction(m0, Y, W1)
        end
        let m = fit_spvar(Y, W1, N, T; ordering = [1, 2], s = 1, Wl = Wl)
            @test_throws DimensionMismatch lr_spatial_interaction(
                m, Y[:, 1:1], W1)
        end
    end

    @testset "40k. the LR test and its degrees of freedom" begin
        rng = MersenneTwister(41_101)
        N, T = 12, 300
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl)
        lr = lr_spatial_interaction(m, Y, Wl[2])
        # df counted from the parameter vectors must match the formula
        # s(K(K+1)/2 + K²p) derived from the block structure.
        @test lr.df == 1 * (div(2 * 3, 2) + 4 * 1)
        @test lr.stat > 0
        @test lr.loglik_full >= lr.loglik_restricted
        @test 0 <= lr.p_value <= 1
        # On a DGP with real spatial interaction it rejects decisively.
        @test lr.p_value < 1e-6
        # And the formula holds across shapes.
        for (K, p, s) in ((2, 1, 1), (2, 2, 1), (3, 1, 1))
            nf = sum(s + s * (k - 1) + s * K * p for k in 1:K)
            @test nf == s * (div(K * (K + 1), 2) + K^2 * p)
        end
    end

    # ── 40m ────────────────────────────────────────────────────────
    # The K = 1 reductions. `fit_spvar` absorbs μ by demeaning each
    # unit's LEVEL series once, over t = p+1..T_per, which is what keeps
    # (I − ρW)ỹ exact when s ≥ 1: the spatial lag of a demeaned series
    # is not the demeaned spatial lag, so demeaning the constructed
    # columns separately would break the structural form.
    #
    # At s = 0 there is no spatial term, and the exact profiled FIML
    # instead demeans the lagged and current sub-series by their OWN
    # means, which differ by (y_T − y_1)/(T−1). So the two disagree in
    # finite samples. The rate was measured rather than guessed, against
    # an independently written profiled FIML, N = 10, φ = 0.55,
    # heteroskedastic λ ∈ [0.5, 1.5]:
    #
    #   T  =  150  300  600  1200  2400  4800
    #   gap = 6.23e-5 1.45e-5 4.42e-6 1.11e-6 3.90e-7 5.06e-8
    #
    # The ratio per doubling averages about 4, not 2, and T²·gap is
    # roughly constant in 1.2–2.3 while T·gap falls steadily. So the
    # two ways of concentrating μ agree to **second** order, not first
    # -- a stronger result than the O(1/T) first guessed, and the reason
    # this is a documented property rather than a defect.
    @testset "40m. the K = 1 reductions" begin
        function sv_ar1(rng, N, T, phi, lam)
            mu = randn(rng, N)
            y = zeros(N)
            Y = Matrix{Float64}(undef, N * T, 1)
            for t in 1:(T + 200)
                y = mu .+ phi .* y .+ sqrt.(lam) .* randn(rng, N)
                if t > 200
                    Y[((t - 201) * N + 1):((t - 200) * N), 1] = y
                end
            end
            return Y
        end
        # An independently written exact profiled FIML: μ concentrated
        # by the likelihood's own first-order condition, λ by eq (21).
        function sv_ar1_fiml(Y, N, T; iters = 4000)
            yv = reshape(view(Y, :, 1), N, T)
            yl = yv[:, 1:(T - 1)]
            yc = yv[:, 2:T]
            lam = ones(N)
            b = 0.0
            for _ in 1:iters
                num = 0.0
                den = 0.0
                for i in 1:N
                    xi = yl[i, :] .- mean(yl[i, :])
                    zi = yc[i, :] .- mean(yc[i, :])
                    num += sum(xi .* zi) / lam[i]
                    den += sum(xi .* xi) / lam[i]
                end
                b = num / den
                for i in 1:N
                    a = mean(yc[i, :]) - b * mean(yl[i, :])
                    r = yc[i, :] .- a .- b .* yl[i, :]
                    lam[i] = sum(abs2, r) / (T - 1)
                end
            end
            return b, lam
        end

        Nr = 10
        Wr = sv_ring(Nr)
        lamr = collect(range(0.5, 1.5; length = Nr))
        gaps = Float64[]
        for Tr in (300, 1200)
            Yr = sv_ar1(MersenneTwister(4041), Nr, Tr, 0.55, lamr)
            mr = fit_spvar(Yr, Wr, Nr, Tr; ordering = [1], p = 1, s = 0,
                           varnames = ["y"])
            br, lmr = sv_ar1_fiml(Yr, Nr, Tr)
            @test mr.converged
            # Both are estimating the same φ, so they must agree to
            # well inside sampling error at either T.
            @test abs(br - mr.theta[1][1]) < 1e-4
            @test maximum(abs, lmr .- vec(mr.lambda)) < 1e-4
            push!(gaps, abs(br - mr.theta[1][1]))
        end
        # Assert the IMPLICATION of second-order agreement: a fourfold
        # increase in T must shrink the gap by far more than fourfold.
        # Measured ratio 13.1 against the 5 asserted, so the margin is
        # 2.6x -- checked rather than assumed, per testset 38i.
        @test gaps[1] / gaps[2] > 5.0

        # s = 1, K = 1: the single-equation spatial case recovers its
        # own parameters. This is the specification `fit_star` cannot
        # express, since it has a contemporaneous Wyₜ.
        let Tr = 300
            WLr = order_weights_list(Wr, 1)
            W1r = Matrix(WLr[2])
            rng = MersenneTwister(4042)
            Fr = lu(Matrix(1.0I, Nr, Nr) .- 0.40 .* W1r)
            mur = randn(rng, Nr)
            y = zeros(Nr)
            Y2 = Matrix{Float64}(undef, Nr * Tr, 1)
            for t in 1:(Tr + 200)
                y = Fr \ (mur .+ 0.30 .* y .+ 0.10 .* (W1r * y) .+
                          randn(rng, Nr))
                if t > 200
                    Y2[((t - 201) * Nr + 1):((t - 200) * Nr), 1] = y
                end
            end
            m2 = fit_spvar(Y2, Wr, Nr, Tr; ordering = [1], p = 1, s = 1,
                           Wl = WLr, varnames = ["y"])
            @test m2.converged
            @test m2.K == 1
            @test isapprox(m2.rho[1][1], 0.40; atol = 0.06)
            @test isapprox(m2.theta[1][1], 0.30; atol = 0.05)
            @test isapprox(m2.theta[1][2], 0.10; atol = 0.05)
            @test m2.stationary
            # At K = 1 the structural and reduced forms differ only by
            # the Jacobian, so |C₀| is a single determinant.
            @test isapprox(logabsdet(m2.C0)[1],
                           logabsdet(Matrix(1.0I, Nr, Nr) .-
                                     m2.rho[1][1] .* W1r)[1];
                           atol = 1e-10)

            # `fit_dynamic_spatial_panel` builds its own time lag, so
            # handing it y(t−1) as a regressor is exactly collinear
            # with that lag and it refuses. Pinned, because the refusal
            # is the reason these two estimators cannot be compared
            # coefficient by coefficient.
            yv = reshape(view(Y2, :, 1), Nr, Tr)
            @test_throws ArgumentError fit_dynamic_spatial_panel(
                SAR(), IndivFE(), Wr,
                reshape(vec(yv[:, 1:(Tr - 1)]), :, 1),
                vec(yv[:, 2:Tr]), Nr, Tr - 1)
        end
    end

    # ── 40n ────────────────────────────────────────────────────────
    # p > 1. Every other subset here runs at the p = 1 default, so the
    # multi-lag path -- the loop over C_h, the (t-2) coefficient names,
    # the Kp.N companion matrix, and the longer history a forecast
    # needs -- was shipping unexercised. The companion matrix is the
    # part that matters: at p = 1 it IS the reduced-form matrix, so its
    # block structure is only non-trivial from p = 2.
    @testset "40n. p > 1, and rho_bounds" begin
        Np, Tp = 12, 600
        Wp = sv_ring(Np)
        WLp = order_weights_list(Wp, 1)
        W1p = Matrix(WLp[2])
        Imp = Matrix(1.0I, Np, Np)
        Zp = zeros(Np, Np)
        C0p = [Imp - 0.30 * W1p            Zp;
               -(0.35 * Imp + 0.12 * W1p)  Imp + 0.18 * W1p]
        C1p = [0.25 * Imp + 0.08 * W1p   0.12 * Imp - 0.04 * W1p;
               -0.08 * Imp + 0.06 * W1p  0.20 * Imp + 0.10 * W1p]
        C2p = [-0.12 * Imp + 0.05 * W1p  0.06 * Imp;
               0.05 * Imp                -0.10 * Imp + 0.04 * W1p]
        Fp = lu(C0p)
        # Assert the DGP is stationary rather than assuming it: the
        # companion form at p = 2 is 2Kp.N square.
        let comp = [Fp \ C1p  Fp \ C2p;
                    Matrix(1.0I, 2Np, 2Np)  zeros(2Np, 2Np)]
            @test maximum(abs, eigvals(comp)) < 1
        end
        rngp = MersenneTwister(2026)
        y1 = zeros(2Np)
        y2 = zeros(2Np)
        Yp = Matrix{Float64}(undef, Np * Tp, 2)
        for t in 1:(Tp + 300)
            yy = Fp \ (C1p * y1 + C2p * y2 + randn(rngp, 2Np))
            y2 = y1
            y1 = yy
            if t > 300
                k = t - 300
                for i in 1:Np
                    Yp[(k - 1) * Np + i, 1] = yy[i]
                    Yp[(k - 1) * Np + i, 2] = yy[Np + i]
                end
            end
        end

        mp = fit_spvar(Yp, Wp, Np, Tp; ordering = [1, 2], p = 2, s = 1,
                       Wl = WLp, varnames = ["a", "b"])
        @test mp.converged
        @test mp.p == 2
        @test length(mp.C) == 2
        @test mp.T_eff == Tp - 2
        @test mp.stationary
        @test isapprox(mp.rho[1][1], 0.30; atol = 0.05)
        @test isapprox(mp.rho[2][1], -0.18; atol = 0.05)
        # The second lag's coefficients must be named, and named with
        # the right lag index -- the obvious way for a multi-lag design
        # to be silently wrong is to label the wrong column.
        @test any(contains("(t-2)"), mp.coefnames_eq[1])
        @test any(contains("(t-1)"), mp.coefnames_eq[1])
        @test count(contains("(t-2)"), mp.coefnames_eq[1]) ==
              count(contains("(t-1)"), mp.coefnames_eq[1])

        # The long-run multiplier now has TWO lag matrices in it, so
        # this fails if the C_h loop drops or double-counts one.
        let psi = spvar_irf(mp; horizon = 600)
            acc = zeros(2Np, 2Np)
            for P in psi
                acc .+= P
            end
            @test maximum(abs, acc .- inv(mp.C0 - mp.C[1] - mp.C[2])) <
                  1e-12
        end

        # A p = 2 forecast needs two periods of history, and must
        # refuse one rather than quietly padding.
        let hist2 = Yp[(end - 2Np + 1):end, :]
            f = spvar_forecast(mp, 3; y_history = hist2)
            @test size(f) == (Np * 3, 2)
            @test all(isfinite, f)
        end
        @test_throws DimensionMismatch spvar_forecast(mp, 3;
            y_history = Yp[(end - Np + 1):end, :])

        # `rho_bounds` is a documented keyword, so it is exercised:
        # bounded away from the unconstrained optimum, the estimate
        # must sit ON the boundary rather than outside it.
        let mb = fit_spvar(Yp, Wp, Np, Tp; ordering = [1, 2], p = 2,
                           s = 1, Wl = WLp, rho_bounds = (0.5, 0.9))
            @test mb.rho[1][1] >= 0.5 - 1e-8
            @test mb.rho[1][1] <= 0.9 + 1e-8
            @test isapprox(mb.rho[1][1], 0.5; atol = 1e-4)
            # and a bounded fit is worse, since the truth is outside
            @test mb.loglik < mp.loglik
        end
    end

    @testset "40l. accessors and show" begin
        rng = MersenneTwister(41_201)
        N, T = 10, 200
        Wl = order_weights_list(sv_ring(N), 1)
        Y, = sv_sim(rng, Wl, N, T)
        m = fit_spvar(Y, Wl[2], N, T; ordering = [1, 2], s = 1, Wl = Wl,
                      varnames = ["g", "y"])
        @test nobs(m) == N * m.T_eff
        @test loglikelihood(m) == m.loglik
        @test size(residuals(m)) == (N * m.T_eff, 2)
        @test length(coef(m)) == length(coefnames(m))
        @test length(coef(m)) == length(stderror(m))
        @test size(vcov(m), 1) == length(coef(m))
        @test all(>=(0), stderror(m))
        @test issymmetric(vcov(m))
        @test size(m.lambda) == (N, 2)
        @test all(>(0), m.lambda)
        @test size(m.means) == (N, 2)
        @test m.T_eff == T - m.p
        @test m.homogeneous
        @test occursin("Spatial VAR", sprint(show, m))
        @test occursin("g", sprint(show, m))
        # Residual variances recover the unit structural variances the
        # DGP used, as T grows: the DGP draws u ~ N(0, I).
        @test 0.6 < mean(m.lambda) < 1.5
    end
end
