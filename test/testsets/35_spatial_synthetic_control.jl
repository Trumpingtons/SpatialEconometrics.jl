@testset "35. Spatial synthetic control" begin
    _progress("35. Spatial synthetic control")
    # The gate here is the same one testset 34 uses: algebra where the
    # quantity is algebraic. The donor weights solve a convex program,
    # so the reference is not a tolerance on recovery but the exact
    # optimum computed independently by enumerating every support --
    # which is feasible for a small donor pool and is the only check
    # that distinguishes a converged solver from a nearly converged
    # one.
    SC = SpatialEconometrics.SpatialCausal

    scgrid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

    # Exact optimum of  min ||x1 - X0 w||^2 + lam d'w  s.t. w >= 0,
    # sum(w) = 1, by enumerating all 2^J - 1 supports and solving the
    # equality-constrained least squares on each. O(2^J), so J stays
    # small; written out here rather than reusing anything from src so
    # that it is a genuinely independent reference.
    function scm_brute(X0::Matrix{Float64}, x1::Vector{Float64},
                       d::Vector{Float64}, lam::Float64)
        J = size(X0, 2)
        best_w = Float64[]; best_f = Inf
        objf(w) = sum(abs2, x1 .- X0 * w) + lam * dot(d, w)
        for mask in 1:(2^J - 1)
            S = [j for j in 1:J if (mask >> (j - 1)) & 1 == 1]
            k = length(S)
            Xs = X0[:, S]
            A = [2 .* (Xs' * Xs)  ones(k); ones(1, k)  0.0]
            rank(A) < k + 1 && continue
            sol = nothing
            try
                sol = A \ vcat(2 .* (Xs' * x1) .- lam .* d[S], 1.0)
            catch
                continue
            end
            v = sol[1:k]
            all(>=(-1e-11), v) || continue
            w = zeros(J); w[S] .= max.(v, 0.0)
            sw = sum(w); sw > 0 && (w ./= sw)
            f = objf(w)
            f < best_f - 1e-14 && (best_f = f; best_w = w)
        end
        return best_w, best_f
    end

    @testset "35a. simplex projection" begin
        # The projection is characterised by subtracting one common
        # threshold and clipping: so on the support, v - w must be
        # constant, and nowhere may w go negative.
        rng = MersenneTwister(3501)
        worst_sum = 0.0; worst_neg = 0.0; worst_spread = 0.0
        for _ in 1:3000
            J = rand(rng, 1:12)
            v = randn(rng, J) .* exp(randn(rng) * 2)
            w = copy(v); SC._project_simplex!(w)
            worst_sum = max(worst_sum, abs(sum(w) - 1.0))
            worst_neg = max(worst_neg, -minimum(w))
            sup = w .> 0
            if any(sup) && !all(sup)
                c = (v .- w)[sup]
                worst_spread = max(worst_spread, maximum(c) - minimum(c))
            end
        end
        @test worst_sum < 1e-13
        @test worst_neg == 0.0
        @test worst_spread < 1e-13

        # Against brute force in low dimension: no randomly drawn
        # simplex point may be closer to v than the projection is.
        rng2 = MersenneTwister(3502)
        worst_excess = -Inf
        for _ in 1:400
            J = rand(rng2, 2:4)
            v = randn(rng2, J) .* 2
            w = copy(v); SC._project_simplex!(w)
            dw = sum(abs2, v .- w)
            best = Inf
            for _ in 1:2000
                c = rand(rng2, J); c ./= sum(c)
                best = min(best, sum(abs2, v .- c))
            end
            worst_excess = max(worst_excess, dw - best)
        end
        @test worst_excess < 0.0

        # Degenerate inputs. A single donor must take all the weight,
        # and a point already on the simplex must be returned
        # unchanged.
        w1 = [-5.0]; SC._project_simplex!(w1)
        @test w1 == [1.0]
        w2 = [0.2, 0.3, 0.5]; w2c = copy(w2)
        SC._project_simplex!(w2)
        @test maximum(abs, w2 .- w2c) < 1e-15
        @test_throws ArgumentError SC._project_simplex!(Float64[])
    end

    @testset "35b. solver against the exact optimum" begin
        # J <= T0, so X0 has full column rank and the optimum is
        # unique: the weights themselves must agree, not merely the
        # objective.
        rng = MersenneTwister(3503)
        max_obj = 0.0; max_w = 0.0; max_kkt = 0.0; max_viol = 0.0
        nprob = 0
        for _ in 1:120
            J = rand(rng, 3:6)
            T0 = rand(rng, (J + 2):(J + 8))
            X0 = randn(rng, T0, J)
            x1 = randn(rng, T0)
            d = abs.(randn(rng, J))
            lam = rand(rng) < 0.5 ? 0.0 : rand(rng) * 2.0
            w, f, _, kkt, viol = SC._solve_pensynth(X0, x1, d, lam)
            wb, fb = scm_brute(X0, x1, d, lam)
            isempty(wb) && continue
            nprob += 1
            max_obj = max(max_obj, f - fb)
            max_w = max(max_w, maximum(abs, w .- wb))
            max_kkt = max(max_kkt, kkt); max_viol = max(max_viol, viol)
        end
        @test nprob > 100
        @test max_obj < 1e-12        # measured 3.55e-15
        @test max_w < 1e-10          # measured 1.31e-14
        @test max_kkt < 1e-12        # measured 9.68e-15
        @test max_viol == 0.0

        # J > T0 is the realistic case -- more donors than
        # pre-periods -- and there the optimum is a face, not a point.
        # Only the objective is well defined, which is itself the
        # argument for the penalty.
        rng2 = MersenneTwister(3504)
        max_obj2 = 0.0; n2 = 0
        for _ in 1:80
            T0 = rand(rng2, 2:4); J = rand(rng2, 5:7)
            X0 = randn(rng2, T0, J); x1 = randn(rng2, T0)
            d = abs.(randn(rng2, J))
            lam = rand(rng2) < 0.5 ? 0.0 : rand(rng2) * 1.5
            w, f, = SC._solve_pensynth(X0, x1, d, lam)
            wb, fb = scm_brute(X0, x1, d, lam)
            isempty(wb) && continue
            n2 += 1
            max_obj2 = max(max_obj2, f - fb)
        end
        @test n2 > 60
        @test max_obj2 < 1e-12       # measured 1.78e-15
    end

    @testset "35c. exact convex combination" begin
        # The treated unit's path IS 0.6 * donor_a + 0.4 * donor_b at
        # every period, so with no noise the weights are identified
        # and the gap is zero throughout. Machine precision, not a
        # recovery tolerance.
        side = 8; N = side^2; T_per = 20; T0 = 14
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3505)
        Y = randn(rng, N, T_per)
        a, b = 50, 61
        Y[1, :] .= 0.6 .* Y[a, :] .+ 0.4 .* Y[b, :]

        scm = fit_synthetic_control(Y, 1, T0; W = W, coords = coords)
        qa = findfirst(==(a), scm.donors)
        qb = findfirst(==(b), scm.donors)
        @test qa !== nothing && qb !== nothing
        @test scm.weights[1, qa] ≈ 0.6 atol = 1e-10
        @test scm.weights[1, qb] ≈ 0.4 atol = 1e-10
        truth = [j == a ? 0.6 : j == b ? 0.4 : 0.0 for j in scm.donors]
        @test maximum(abs, scm.weights[1, :] .- truth) < 1e-10
        @test abs(sum(scm.weights[1, :]) - 1.0) < 1e-13
        @test minimum(scm.weights) >= 0.0
        @test maximum(abs, scm.gaps) < 1e-9
        @test abs(scm.att) < 1e-9
        @test scm.kkt_gap[1] < 1e-12
        @test scm.kkt_violation[1] < 1e-12

        # The gap is a definition, so it holds exactly rather than to
        # a tolerance.
        syn = (Y[scm.donors, :])' * scm.weights[1, :]
        @test maximum(abs, scm.gaps[1, :] .- (Y[1, :] .- syn)) == 0.0
        @test maximum(abs, scm.synthetic[1, :] .- syn) == 0.0
        # residuals() is the pre-period slice of the gaps.
        @test size(StatsAPI.residuals(scm)) == (1, T0)
        @test maximum(abs, StatsAPI.residuals(scm) .- scm.gaps[:, 1:T0]) == 0.0
        @test StatsAPI.coef(scm) === donor_weights(scm)
        @test StatsAPI.coefnames(scm) == ["unit $(j)" for j in scm.donors]
        @test StatsAPI.nobs(scm) == T0
        # No standard errors exist for this estimator, and the type
        # must not pretend otherwise.
        @test_throws MethodError StatsAPI.vcov(scm)
        @test_throws MethodError StatsAPI.stderror(scm)
    end

    @testset "35d. penalty limits" begin
        rng = MersenneTwister(3506)
        T0 = 10; J = 6
        X0 = randn(rng, T0, J); x1 = randn(rng, T0)
        d = [5.0, 1.0, 9.0, 3.0, 7.0, 4.0]

        # lam = 0 is the unpenalised program.
        w0, f0, = SC._solve_pensynth(X0, x1, d, 0.0)
        wn, fn, = SC._solve_pensynth(X0, x1, zeros(J), 0.0)
        @test maximum(abs, w0 .- wn) < 1e-10
        @test abs(f0 - fn) < 1e-12

        # As lam grows the support must collapse into argmin d. With
        # distinct distances that is the single nearest donor.
        for lam in (50.0, 1e4, 1e8)
            w, = SC._solve_pensynth(X0, x1, d, lam)
            @test argmax(w) == argmin(d)
            @test w[argmin(d)] ≈ 1.0 atol = 1e-12
        end

        # With ties -- the usual case on a lattice, where several
        # donors sit at the same distance -- the support must lie
        # within the tied set, which is the general statement.
        rng2 = MersenneTwister(3507)
        bad = 0
        for _ in 1:40
            Jq = rand(rng2, 4:8); T0q = rand(rng2, 5:10)
            X0q = randn(rng2, T0q, Jq); x1q = randn(rng2, T0q)
            dq = abs.(randn(rng2, Jq)) .+ 0.1
            wq, = SC._solve_pensynth(X0q, x1q, dq, 1e8)
            dmin = minimum(dq)
            all(q -> dq[q] <= dmin + 1e-9, findall(>(1e-9), wq)) || (bad += 1)
        end
        @test bad == 0

        # Monotonicity of the penalty term: a larger lam cannot buy a
        # larger distance-weighted average.
        side = 8; N = side^2; T_per = 16; T0b = 12
        W = make_rook_weights(side); coords = scgrid(side)
        rng3 = MersenneTwister(3508)
        Yb = randn(rng3, N, T_per)
        prev = Inf
        for lam in (0.0, 0.05, 0.5, 5.0)
            s = fit_synthetic_control(Yb, 28, T0b; W = W, coords = coords,
                                      penalty = lam)
            dn = [sqrt(sum(abs2, coords[28, :] .- coords[j, :]))
                  for j in s.donors]
            avg = dot(dn, s.weights[1, :])
            @test avg <= prev + 1e-8
            prev = avg
        end
    end

    @testset "35e. donor pool screening" begin
        side = 8; N = side^2; T_per = 16; T0 = 12
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3509)
        Y = randn(rng, N, T_per)

        # Unit 28 sits at (4,4) on the lattice; its rook neighbours are
        # 20, 27, 29 and 36 and none of them may be a donor.
        s = fit_synthetic_control(Y, 28, T0; W = W, coords = coords)
        @test sort(s.excluded_neighbours) == [20, 27, 29, 36]
        @test !(28 in s.donors)
        @test isempty(intersect(s.donors, s.excluded_neighbours))
        @test length(s.donors) == N - 1 - 4
        @test s.donor_pool_rule === :exclude_neighbours

        # :all keeps every untreated unit, neighbours included, and
        # records no exclusions.
        sa = fit_synthetic_control(Y, 28, T0; donor_pool = :all,
                                   penalty_metric = :none)
        @test length(sa.donors) == N - 1
        @test isempty(sa.excluded_neighbours)
        @test !(28 in sa.donors)

        # An explicit pool is honoured, and the treated unit is removed
        # from it even if the caller left it in.
        se = fit_synthetic_control(Y, 28, T0; donor_pool = [1, 2, 3, 28],
                                   penalty_metric = :none)
        @test se.donors == [1, 2, 3]
        @test se.donor_pool_rule === :explicit

        # Several treated units: none of them may be a donor, and the
        # exclusion set is the union of their neighbourhoods.
        sm = fit_synthetic_control(Y, [28, 30], T0; W = W, coords = coords)
        @test isempty(intersect(sm.donors, [28, 30]))
        @test 29 in sm.excluded_neighbours      # neighbours both
        @test 22 in sm.excluded_neighbours      # neighbours 30 only
        @test size(sm.weights) == (2, length(sm.donors))

        # The default needs W, and says so rather than silently
        # falling back to a pool that includes contaminated units.
        err = nothing
        try
            fit_synthetic_control(Y, 28, T0)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("needs `W`", err.msg)

        # Guards.
        @test_throws ArgumentError fit_synthetic_control(Y, 28, T0;
                                                         donor_pool = :nonsense)
        @test_throws ArgumentError fit_synthetic_control(Y, 28, T_per)
        @test_throws ArgumentError fit_synthetic_control(Y, 28, 0; W = W,
                                                         coords = coords)
        @test_throws ArgumentError fit_synthetic_control(Y, N + 1, T0; W = W,
                                                         coords = coords)
        @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                         coords = coords,
                                                         penalty = -1.0)
        @test_throws ArgumentError fit_synthetic_control(Y, [28, 28], T0; W = W,
                                                         coords = coords)
        # :spatial penalty without coords.
        @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                         penalty = 0.5)
        # Treating every unit leaves no donors at all.
        @test_throws ArgumentError fit_synthetic_control(Y, collect(1:N), T0;
                                                         donor_pool = :all)
    end

    @testset "35f. donor-pool contamination, exactly" begin
        # The point of screening the donor pool, made algebraic rather
        # than statistical. The spillover lands only in the
        # post-treatment periods, so the weights -- fitted on the
        # pre-period -- are untouched by it. The contamination a
        # neighbour carries into the synthetic control is therefore
        # exactly its weight times its exposure times the spillover,
        # and the bias follows without a tolerance.
        #
        # The design makes the trap unavoidable: the treated unit has a
        # path that is an exact half-and-half combination of two of its
        # own neighbours, which are in turn exact copies of two far
        # corners. An unscreened fit can reach a perfect pre-period fit
        # through the neighbours, and does.
        side = 9; N = side^2; T_per = 20; T0 = 14
        W = make_rook_weights(side); coords = scgrid(side)
        Wt = SpatialEconometrics.row_standardize(W)
        i = 41                               # (5,5): interior, 4 neighbours
        nbs = findall(!iszero, Matrix(W)[i, :])
        @test nbs == [32, 40, 42, 50]
        nb1, nb2 = nbs[1], nbs[2]
        a, b = 1, 9                          # far corners, exposure 0
        tau = 3.0
        D = zeros(N); D[i] = 1.0
        E = Wt * D
        @test E[nb1] == 0.25 && E[nb2] == 0.25
        @test E[a] == 0.0 && E[b] == 0.0 && E[i] == 0.0

        rng = MersenneTwister(3510)
        base = randn(rng, N, T_per)
        base[nb1, :] .= base[a, :]
        base[nb2, :] .= base[b, :]
        base[i, :]   .= 0.5 .* base[a, :] .+ 0.5 .* base[b, :]

        # The bias is linear in the spillover, with slope
        # -(weight on contaminated donors) * exposure. Pinned at four
        # spillover sizes, each to machine precision.
        for spill in (0.0, 1.0, 2.0, 4.0)
            Y = copy(base)
            for t in (T0 + 1):T_per
                Y[i, t] += tau
                Y[:, t] .+= spill .* E
            end
            s_all = fit_synthetic_control(Y, i, T0; donor_pool = :all,
                                          coords = coords, penalty = 0.01)
            wd = Dict(j => s_all.weights[1, q]
                      for (q, j) in enumerate(s_all.donors))
            # It reaches a perfect pre-period fit through the two
            # contaminated neighbours, taking half from each.
            @test s_all.rmspe_pre[1] < 1e-12
            @test wd[nb1] ≈ 0.5 atol = 1e-9
            @test wd[nb2] ≈ 0.5 atol = 1e-9
            # And is wrong by exactly the contamination it absorbed.
            contam = sum(s_all.weights[1, q] * spill * E[j]
                         for (q, j) in enumerate(s_all.donors))
            @test contam ≈ spill * 0.25 atol = 1e-9
            @test s_all.att ≈ tau - contam atol = 1e-12
            @test s_all.att ≈ tau - spill * 0.25 atol = 1e-9
        end

        # The screened fit cannot touch the neighbours, so it is
        # right -- and its pre-period fit is *worse*. A good
        # pre-period fit is not evidence that a synthetic control is
        # the right one, which is the reason this testset exists.
        spill = 2.0
        Y = copy(base)
        for t in (T0 + 1):T_per
            Y[i, t] += tau
            Y[:, t] .+= spill .* E
        end
        s_screen = fit_synthetic_control(Y, i, T0; W = W, coords = coords,
                                         penalty = 0.01)
        s_all = fit_synthetic_control(Y, i, T0; donor_pool = :all,
                                      coords = coords, penalty = 0.01)
        @test sort(s_screen.excluded_neighbours) == nbs
        @test abs(s_screen.att - tau) < 0.01
        @test s_all.att ≈ tau - 0.5 atol = 1e-9
        @test abs(s_all.att - tau) > 0.49
        @test s_all.rmspe_pre[1] < s_screen.rmspe_pre[1]

        # The weights are fitted pre-period, so adding a post-period
        # spillover cannot change them at all. This is what makes the
        # identity above exact rather than approximate.
        Y_nospill = copy(base)
        for t in (T0 + 1):T_per
            Y_nospill[i, t] += tau
        end
        s_ns = fit_synthetic_control(Y_nospill, i, T0; donor_pool = :all,
                                     coords = coords, penalty = 0.01)
        @test maximum(abs, s_ns.weights .- s_all.weights) == 0.0
        @test s_ns.att ≈ tau atol = 1e-9
        @test (s_all.att - s_ns.att) ≈ -0.5 atol = 1e-9

        # The spillover onto the excluded neighbours is recoverable
        # from the same clean pool -- but only for the neighbours the
        # pool can actually fit, and the pre-period RMSPE says which
        # those are before any gap is read.
        #
        # Units nb1 and nb2 are exact copies of the far corners a and
        # b, which are in the clean pool, so their synthetic controls
        # are exact and their gaps are the spillover alone. The other
        # two neighbours have no such counterpart, fit badly, and
        # their gaps are not the spillover. That is why
        # `SpilloverGaps` carries `rmspe_pre` per unit rather than
        # only the average.
        s_exact = fit_synthetic_control(Y, i, T0; W = W, coords = coords,
                                        penalty = 0.0)
        sg = spillover_gaps(s_exact, Y; W = W, coords = coords)
        @test sort(sg.units) == nbs
        @test length(sg.units) == 4
        @test size(sg.gaps) == (4, T_per)
        @test maximum(abs, sg.spillover_path .-
                           vec(mean(sg.gaps; dims = 1))) < 1e-14
        @test sg.spillover ≈ mean(sg.spillover_path[(T0 + 1):T_per]) atol = 1e-14

        good = [q for (q, u) in enumerate(sg.units) if u in (nb1, nb2)]
        poor = [q for (q, u) in enumerate(sg.units) if !(u in (nb1, nb2))]
        @test length(good) == 2 && length(poor) == 2
        # The two that can be fitted are fitted exactly, and return
        # the spillover to machine precision.
        for q in good
            @test sg.rmspe_pre[q] < 1e-13
            @test maximum(abs, sg.gaps[q, 1:T0]) < 1e-13
            @test sg.gaps[q, T_per] ≈ spill * 0.25 atol = 1e-10
            @test mean(sg.gaps[q, (T0 + 1):T_per]) ≈ spill * 0.25 atol = 1e-10
        end
        # The two that cannot are off by more than the spillover
        # itself, and their pre-period fit is three orders of
        # magnitude worse -- visible before the gap is looked at.
        for q in poor
            @test sg.rmspe_pre[q] > 0.4
            @test abs(sg.gaps[q, T_per] - spill * 0.25) > 0.5
        end
        @test minimum(sg.rmspe_pre[poor]) > 1e10 * maximum(sg.rmspe_pre[good])
        # So the unweighted average across all four is pulled off the
        # truth, by much more than the well-fitted pair are. The fix
        # is the user reading `rmspe_pre`, not the function silently
        # dropping units.
        #
        # This aggregate is deliberately bounded rather than pinned:
        # it depends on the two gaps that cannot be fitted, and those
        # follow the `randn(::MersenneTwister)` stream, which differs
        # between Julia minor versions. Measured 0.5452 on 1.13.2 and
        # 0.5887 on 1.10.12 — the well-fitted pair returns exactly
        # 0.500000000000 on both, which is why that is what gets
        # pinned above and this does not.
        @test 0.52 < sg.spillover < 0.62
        @test abs(sg.spillover - spill * 0.25) >
              10 * abs(mean(sg.gaps[good, (T0 + 1):T_per]) - spill * 0.25)
        # Asking for spillovers when nothing was excluded is an error,
        # not an empty result that would read as "no spillover".
        @test_throws ArgumentError spillover_gaps(s_all, Y)
        @test_throws DimensionMismatch spillover_gaps(s_screen,
                                                      Y[:, 1:(T_per - 1)])
    end

    @testset "35g. aggregation over treated units" begin
        side = 8; N = side^2; T_per = 18; T0 = 13
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3511)
        Y = randn(rng, N, T_per)
        tr = [11, 30, 54]
        s = fit_synthetic_control(Y, tr, T0; W = W, coords = coords,
                                  penalty = 0.05)
        @test s.treated == tr
        @test size(s.weights) == (3, length(s.donors))
        # Each row is a point on the simplex.
        for k in 1:3
            @test abs(sum(s.weights[k, :]) - 1.0) < 1e-13
            @test minimum(s.weights[k, :]) >= 0.0
        end
        # The ATT path is the mean of the per-unit gaps, exactly, and
        # the ATT is its post-period mean.
        @test maximum(abs, s.att_path .- vec(mean(s.gaps; dims = 1))) == 0.0
        @test s.att ≈ mean(s.att_path[(T0 + 1):T_per]) atol = 1e-15
        # Per-unit gaps are per-unit definitions.
        for k in 1:3
            syn = (Y[s.donors, :])' * s.weights[k, :]
            @test maximum(abs, s.gaps[k, :] .- (Y[tr[k], :] .- syn)) == 0.0
            @test s.rmspe_pre[k] ≈ sqrt(mean(abs2, s.gaps[k, 1:T0])) atol = 1e-15
            @test s.rmspe_post[k] ≈
                  sqrt(mean(abs2, s.gaps[k, (T0 + 1):T_per])) atol = 1e-15
            @test s.ratio[k] ≈ s.rmspe_post[k] / s.rmspe_pre[k] atol = 1e-13
        end
        @test StatsAPI.nobs(s) == 3 * T0

        # A 0/1 indicator and an index vector must select the same
        # units and give the same answer.
        ind = zeros(N); ind[tr] .= 1.0
        s2 = fit_synthetic_control(Y, ind, T0; W = W, coords = coords,
                                   penalty = 0.05)
        @test s2.treated == tr
        @test maximum(abs, s2.weights .- s.weights) == 0.0
        bind = falses(N); bind[tr] .= true
        s3 = fit_synthetic_control(Y, bind, T0; W = W, coords = coords,
                                   penalty = 0.05)
        @test s3.treated == tr
    end

    @testset "35h. predictors and standardisation" begin
        side = 8; N = side^2; T_per = 16; T0 = 12
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3512)
        Y = randn(rng, N, T_per)

        # An explicit predictor matrix is used in place of the
        # pre-period path, and changes the number of fitted moments.
        P = randn(rng, N, 4)
        sp = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                   predictors = P)
        @test StatsAPI.nobs(sp) == 4
        @test size(sp.weights) == (1, length(sp.donors))

        # Standardisation must be scale-free: multiplying one
        # predictor by a constant cannot change the weights once each
        # is scaled to unit variance, and must change them when it is
        # not.
        P2 = copy(P); P2[:, 1] .*= 1000.0
        s_std_a = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                        predictors = P, standardize = true)
        s_std_b = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                        predictors = P2, standardize = true)
        @test maximum(abs, s_std_a.weights .- s_std_b.weights) < 1e-8
        s_raw_a = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                        predictors = P, standardize = false)
        s_raw_b = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                        predictors = P2, standardize = false)
        @test maximum(abs, s_raw_a.weights .- s_raw_b.weights) > 1e-3

        # Covariate-distance penalty is available and differs from the
        # spatial one.
        sc = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                   penalty = 0.2,
                                   penalty_metric = :covariate)
        @test sc.penalty_metric === :covariate
        @test abs(sum(sc.weights[1, :]) - 1.0) < 1e-13
        # penalty = 0 records :none whatever the metric asked for,
        # since no distance is consulted.
        sz = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                   penalty = 0.0,
                                   penalty_metric = :spatial)
        @test sz.penalty_metric === :none

        @test_throws ArgumentError fit_synthetic_control(Y, 28, T0; W = W,
                                                         coords = coords,
                                                         penalty = 0.2,
                                                         penalty_metric = :bogus)
        @test_throws DimensionMismatch fit_synthetic_control(Y, 28, T0; W = W,
                                                             coords = coords,
                                                             predictors = randn(rng, N + 1, 3))
        # Haversine is accepted for lon/lat coordinates.
        sh = fit_synthetic_control(Y, 28, T0; W = W, coords = coords,
                                   penalty = 0.01,
                                   metric = HaversineDistance())
        @test abs(sum(sh.weights[1, :]) - 1.0) < 1e-13
    end

    @testset "35i. placebo test mechanics" begin
        side = 7; N = side^2; T_per = 14; T0 = 10
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3513)
        F = randn(rng, 2, T_per); L = rand(rng, N, 2)
        Y = L * F .+ 0.05 .* randn(rng, N, T_per)
        i = 25
        for t in (T0 + 1):T_per
            Y[i, t] += 4.0
        end
        s = fit_synthetic_control(Y, i, T0; W = W, coords = coords)
        J = length(s.donors)

        pt = placebo_test(s, Y; W = W, coords = coords)
        # One treated unit and J donors gives exactly J assignments,
        # so the test is exact and the p-value lives on a grid of
        # 1/(J+1).
        @test pt.exact
        @test pt.n_placebo == J
        @test pt.n_possible == Float64(J)
        @test pt.m_treated == 1
        @test pt.n_donors == J
        @test pt.min_attainable_p ≈ 1 / (J + 1) atol = 1e-15
        @test length(pt.ratio_placebo) == J
        @test pt.ratio_observed == s.ratio_aggregate
        # The p-value is the rank formula, recomputed here rather than
        # taken on trust.
        @test pt.p_value ≈
              (1 + count(>=(pt.ratio_observed), pt.ratio_placebo)) / (1 + J) atol = 1e-15
        @test pt.p_value >= pt.min_attainable_p
        # A planted effect this large should land at the floor.
        @test pt.p_value ≈ pt.min_attainable_p atol = 1e-15

        # Sampling path: cap the assignments below the number
        # possible and the test reports itself as inexact.
        sm = fit_synthetic_control(Y, [25, 27], T0; W = W, coords = coords)
        Jm = length(sm.donors)
        pts = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(1),
                           W = W, coords = coords)
        @test !pts.exact
        @test pts.n_placebo == 50
        @test pts.n_possible ≈ Jm * (Jm - 1) / 2 rtol = 1e-12
        @test pts.min_attainable_p ≈ 1 / 51 atol = 1e-15
        # Seeded, so it repeats exactly.
        pts2 = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(1),
                            W = W, coords = coords)
        @test pts2.p_value == pts.p_value
        @test maximum(abs, pts2.ratio_placebo .- pts.ratio_placebo) == 0.0
        # And a different seed generally gives a different sample.
        pts3 = placebo_test(sm, Y; n_placebo = 50, rng = MersenneTwister(9),
                            W = W, coords = coords)
        @test pts3.n_placebo == 50

        # _n_subsets against binomial, including the overflow guard.
        @test SC._n_subsets(10, 3) ≈ 120.0 atol = 1e-9
        @test SC._n_subsets(5, 0) == 1.0
        @test SC._n_subsets(5, 6) == 0.0
        @test SC._n_subsets(44, 1) == 44.0
        @test isfinite(SC._n_subsets(2000, 900)) == false
        # _enumerate_subsets really enumerates, without repeats.
        su = SC._enumerate_subsets([3, 7, 9, 11], 2)
        @test length(su) == 6
        @test length(unique(su)) == 6
        @test all(v -> length(v) == 2 && issorted(v), su)
        @test [3, 7] in su && [9, 11] in su

        @test_throws DimensionMismatch placebo_test(s, Y[:, 1:(T_per - 1)];
                                                    W = W, coords = coords)
        @test_throws ArgumentError placebo_test(s, Y; n_placebo = 0,
                                                W = W, coords = coords)
        # No clean donor to permute against.
        stiny = fit_synthetic_control(Y, 1, T0; donor_pool = [2],
                                      penalty_metric = :none)
        @test_throws ArgumentError placebo_test(stiny, Y; donor_pool = [2])
    end

    @testset "35j. placebo size under the null" begin
        # A p-value is a test statistic, so the standing rule in
        # docs/AUDIT.md applies: its null distribution is measured,
        # not assumed. Here a short version of the study recorded in
        # the ledger -- enough replications to catch a grossly
        # miscalibrated test without making the suite slow.
        side = 6; N = side^2; T_per = 12; T0 = 9
        W = make_rook_weights(side); coords = scgrid(side)
        rng = MersenneTwister(3514)
        ps = Float64[]
        minp = NaN
        for _ in 1:60
            F = randn(rng, 2, T_per); L = rand(rng, N, 2)
            Y = L * F .+ 0.3 .* randn(rng, N, T_per)   # no effect at all
            s = fit_synthetic_control(Y, 15, T0; W = W, coords = coords)
            pt = placebo_test(s, Y; W = W, coords = coords)
            push!(ps, pt.p_value)
            minp = pt.min_attainable_p
        end
        # Every p-value is a valid probability on the attainable grid.
        @test all(p -> minp - 1e-12 <= p <= 1.0 + 1e-12, ps)
        # With no effect the test must not reject wholesale. This is a
        # loose bound by design: the precise size, and the fact that
        # it is not nominal, is measured in docs/AUDIT.md rather than
        # asserted here.
        @test count(<=(0.10), ps) / length(ps) < 0.5
        @test mean(ps) > 0.1
    end
end
