@testset "34. Spatial causal inference" begin
    _progress("34. Spatial causal inference")
    # The verification standard here is algebra, not a recovery
    # tolerance: the two-period design first-differences to an exact
    # OLS problem, so with no noise the estimator must return the
    # generating parameters to machine precision. Anything looser would
    # not distinguish a correct decomposition from a nearly correct one,
    # which is the failure shape of audit findings 13-16.
    SC = SpatialEconometrics.SpatialCausal

    grid(side) = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

    # A contiguous treated block: exposure then varies among controls
    # (some border the block, most do not), which is what identifies a
    # spillover.
    function block_design(side, lo, hi)
        N = side^2
        coords = grid(side)
        D = zeros(N)
        for i in 1:N
            r, c = coords[i, 1], coords[i, 2]
            (lo <= r <= hi && lo <= c <= hi) && (D[i] = 1.0)
        end
        return N, coords, D
    end

    @testset "34a. exposure mapping is what it claims to be" begin
        side = 8
        N, coords, D = block_design(side, 3, 5)
        W = make_rook_weights(side)
        Wt = row_standardize(W)

        em = exposure(D, W)
        @test size(em.E) == (N, 1)
        @test em.order == 1
        @test em.labels == ["WD"]
        @test em.standardized

        # Row-standardized: exposure is the fraction of neighbours
        # treated, so it is bounded and matches a hand computation.
        @test all(0.0 .<= em.E[:, 1] .<= 1.0)
        for i in (1, 20, N)
            nbrs = findall(!iszero, W[i, :])
            @test em.E[i, 1] ≈ sum(D[nbrs] .* Wt[i, nbrs]) rtol = 1e-14
        end
        @test em.E[:, 1] ≈ Wt * D rtol = 1e-14

        # Unstandardized gives a count, not a fraction.
        emc = exposure(D, W; standardize = false)
        @test emc.E[:, 1] ≈ W * D rtol = 1e-14
        @test !emc.standardized

        # Second order is W² D, and it is a genuinely different column.
        em2 = exposure(D, W; order = 2)
        @test size(em2.E) == (N, 2)
        @test em2.labels == ["WD", "W2D"]
        @test em2.E[:, 1] ≈ Wt * D rtol = 1e-14
        @test em2.E[:, 2] ≈ Wt * (Wt * D) rtol = 1e-14
        @test !(em2.E[:, 1] ≈ em2.E[:, 2])

        # A non-binary treatment makes the direct/indirect split
        # meaningless, and degenerate designs have nothing to estimate.
        @test_throws ArgumentError exposure([0.0, 0.5, 1.0], make_rook_weights(2)[1:3, 1:3])
        @test_throws ArgumentError exposure(zeros(N), W)
        @test_throws ArgumentError exposure(ones(N), W)
        @test_throws DimensionMismatch exposure(zeros(N + 1), W)
        @test_throws ArgumentError exposure(D, W; order = 0)
    end

    @testset "34b. interference report finds what identifies δ" begin
        side = 10
        N, coords, D = block_design(side, 4, 6)
        W = make_rook_weights(side)
        em = exposure(D, W)
        rep = check_interference(em)

        @test rep.n_treated == count(isone, D)
        @test rep.n_control == N - rep.n_treated
        @test rep.n_treated + rep.n_control == N
        @test rep.identified
        # A block in a 10x10 grid leaves plenty of untouched controls.
        @test rep.n_control_unexposed > 0
        @test rep.exposure_var_control > 0
        @test rep.mean_exposure_treated > rep.mean_exposure_control

        E = em.E[:, 1]
        @test rep.exposure_var_treated ≈ var(E[D .== 1.0]) rtol = 1e-14
        @test rep.exposure_var_control ≈ var(E[D .== 0.0]) rtol = 1e-14
        @test rep.mean_exposure_treated ≈ mean(E[D .== 1.0]) rtol = 1e-14
        @test rep.mean_exposure_control ≈ mean(E[D .== 0.0]) rtol = 1e-14
        @test rep.exposure_cor ≈ cor(D, E) rtol = 1e-12
        @test occursin("identified", rep.message)

        # No control has a treated neighbour: δ has no baseline, and
        # the report must say so rather than leave it to be discovered.
        Wfar = make_rook_weights(side)
        Dlone = zeros(N); Dlone[1] = 1.0
        # Remove the treated unit's links so nothing is exposed at all.
        Wiso = sparse(Wfar)
        Wiso[1, :] .= 0.0; Wiso[:, 1] .= 0.0
        dropzeros!(Wiso)
        em_iso = exposure(Dlone, Wiso)
        rep_iso = check_interference(em_iso)
        @test !rep_iso.identified
        @test rep_iso.n_control_unexposed == rep_iso.n_control
        @test occursin("no control", lowercase(rep_iso.message))

        # show must not error on either branch
        @test occursin("Interference", sprint(show, rep))
        @test occursin("identified", sprint(show, rep_iso))
    end

    @testset "34c. two-period DiD is exact, not approximate" begin
        side = 12
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        em = exposure(D, W)
        E = em.E[:, 1]

        γ, τ, δ = 0.7, 2.5, -1.25
        μ = collect(range(-1.0, 1.0, length = N))

        y = zeros(N, 2)
        y[:, 1] .= μ
        y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E

        m = fit_spatial_did(y, D, W; vcov = :ols)

        Ē_T = mean(E[D .== 1.0])
        Ē_C = mean(E[D .== 0.0])

        # The generating parameters, to machine precision.
        @test m.att_direct ≈ τ rtol = 1e-12
        @test m.att_spill_treated ≈ δ * Ē_T rtol = 1e-12
        @test m.att_spill_control ≈ δ * Ē_C rtol = 1e-12
        @test m.att_total_treated ≈ τ + δ * Ē_T rtol = 1e-12
        @test m.coef[1] ≈ γ rtol = 1e-12
        @test m.coef[2] ≈ τ rtol = 1e-12
        @test m.coef[3] ≈ δ rtol = 1e-12
        @test norm(m.residuals) < 1e-10

        # The decomposition is an identity, so it holds exactly rather
        # than to a tolerance -- both are the same linear form in θ̂.
        @test m.att_total_treated == m.att_direct + m.att_spill_treated

        # First-differencing removes the unit effect, so an arbitrary
        # change to it cannot move any reported effect.
        y2 = copy(y)
        shift = collect(range(-50.0, 50.0, length = N))
        y2[:, 1] .+= shift
        y2[:, 2] .+= shift
        m2 = fit_spatial_did(y2, D, W; vcov = :ols)
        @test m2.att_direct ≈ m.att_direct rtol = 1e-12
        @test m2.att_spill_control ≈ m.att_spill_control rtol = 1e-12

        # StatsAPI surface and show
        @test nobs(m) == N
        @test length(coef(m)) == 3
        @test size(vcov(m)) == (3, 3)
        @test coefnames(m) == ["(Intercept)", "D", "WD"]
        @test dof_residual(m) == N - 3
        @test length(stderror(m)) == 3
        @test occursin("Difference-in-Differences", sprint(show, m))
        @test m.design === :two_period
        @test m.n_units == N && m.n_periods == 2

        # Second-order exposure recovers both rings exactly.
        δ2 = 0.4
        em2 = exposure(D, W; order = 2)
        y3 = zeros(N, 2)
        y3[:, 1] .= μ
        y3[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* em2.E[:, 1] .+ δ2 .* em2.E[:, 2]
        m3 = fit_spatial_did(y3, D, W; vcov = :ols, order = 2)
        @test m3.coef[2] ≈ τ rtol = 1e-10
        @test m3.coef[3] ≈ δ rtol = 1e-10
        @test m3.coef[4] ≈ δ2 rtol = 1e-10
        @test m3.coefnames == ["(Intercept)", "D", "WD", "W2D"]
        # The reported spillover sums over both rings.
        want_t = δ * mean(em2.E[D .== 1.0, 1]) + δ2 * mean(em2.E[D .== 1.0, 2])
        @test m3.att_spill_treated ≈ want_t rtol = 1e-10
    end

    @testset "34d. staggered rollout, exact under homogeneity" begin
        side = 10
        N = side^2
        T_per = 8
        coords = grid(side)
        W = make_rook_weights(side)
        Wt = row_standardize(W)

        adopt = fill(0, N)
        for i in 1:N
            c = coords[i, 2]
            if c <= 2.5;     adopt[i] = 3
            elseif c <= 5.0; adopt[i] = 5
            elseif c <= 7.5; adopt[i] = 7
            end
        end
        Dpath = zeros(N, T_per)
        for i in 1:N, t in 1:T_per
            adopt[i] != 0 && t >= adopt[i] && (Dpath[i, t] = 1.0)
        end

        μ = collect(range(-1.0, 1.0, length = N))
        λ = [0.0, 0.3, 0.1, -0.2, 0.4, 0.25, -0.1, 0.5]
        τ, δ = 2.0, -0.8

        y = zeros(N, T_per)
        for i in 1:N, t in 1:T_per
            E_it = dot(view(Wt, i, :), view(Dpath, :, t))
            y[i, t] = μ[i] + λ[t] + τ * Dpath[i, t] + δ * E_it
        end

        m = fit_spatial_did(y, Dpath, W; vcov = :ols)
        @test m.design === :staggered
        @test m.n_periods == T_per
        # The within transform absorbs μ_i and γ_t exactly, so with
        # homogeneous effects this is as exact as the two-period path.
        @test m.att_direct ≈ τ rtol = 1e-12
        @test m.coef[2] ≈ δ rtol = 1e-12
        @test norm(m.residuals) < 1e-10
        # No intercept: the two-way transform already removed it, and a
        # column of ones would be collinear with the transform.
        @test m.coefnames == ["D", "WD"]
        @test nobs(m) == N * T_per

        # Unit and period effects are absorbed, so shifting either
        # cannot move the estimate.
        y2 = copy(y)
        for i in 1:N, t in 1:T_per
            y2[i, t] += 10.0 * μ[i] - 3.0 * λ[t]
        end
        m2 = fit_spatial_did(y2, Dpath, W; vcov = :ols)
        @test m2.att_direct ≈ τ rtol = 1e-10
        @test m2.coef[2] ≈ δ rtol = 1e-10

        # A length-N vector cannot describe an 8-period design, and
        # saying so beats silently treating it as two periods.
        @test_throws ArgumentError fit_spatial_did(y, zeros(N), W; vcov = :ols)
        @test_throws ArgumentError fit_spatial_did(y[:, 1:1], zeros(N), W; vcov = :ols)
        @test_throws DimensionMismatch fit_spatial_did(y, zeros(N, T_per + 1), W; vcov = :ols)
        @test_throws ArgumentError fit_spatial_did(y, fill(0.5, N, T_per), W; vcov = :ols)
    end

    @testset "34e. staggered TWFE under heterogeneity, as documented" begin
        # The docstring warns that this estimator does not recover the
        # sample-weighted ATT when cohort effects differ, and quotes
        # numbers. These are those numbers: if the implementation
        # changes the warning has to change with it, which is the point
        # of pinning them.
        side = 10
        N = side^2
        T_per = 8
        coords = grid(side)
        W = make_rook_weights(side)
        μ = collect(range(-1.0, 1.0, length = N))
        λ = [0.0, 0.3, 0.1, -0.2, 0.4, 0.25, -0.1, 0.5]
        coh = Dict(3 => 1.0, 5 => 2.0, 7 => 3.0)

        function run_case(layout, profile)
            adopt = fill(0, N)
            for i in 1:N
                r, c = Int(coords[i, 1]), Int(coords[i, 2])
                if layout === :bands
                    c <= 2 ? (adopt[i] = 3) : c <= 4 ? (adopt[i] = 5) :
                        c <= 6 ? (adopt[i] = 7) : nothing
                else
                    k = mod(r + c, 4)
                    k == 0 ? (adopt[i] = 3) : k == 1 ? (adopt[i] = 5) :
                        k == 2 ? (adopt[i] = 7) : nothing
                end
            end
            Dp = zeros(N, T_per)
            for i in 1:N, t in 1:T_per
                adopt[i] != 0 && t >= adopt[i] && (Dp[i, t] = 1.0)
            end
            y = zeros(N, T_per)
            num, den, lo, hi = 0.0, 0.0, Inf, -Inf
            for i in 1:N, t in 1:T_per
                eff = 0.0
                if adopt[i] != 0 && t >= adopt[i]
                    eff = profile === :constant ? coh[adopt[i]] :
                          coh[adopt[i]] * (t - adopt[i] + 1)
                    num += eff; den += 1
                    lo = min(lo, eff); hi = max(hi, eff)
                end
                y[i, t] = μ[i] + λ[t] + eff
            end
            m = fit_spatial_did(y, Dp, W; vcov = :ols)
            return m.att_direct, num / den, lo, hi
        end

        # No noise, so every gap is bias rather than sampling error.
        est_bc, att_bc, _, _ = run_case(:bands, :constant)
        est_bd, att_bd, lo_bd, hi_bd = run_case(:bands, :dynamic)
        est_ic, att_ic, _, _ = run_case(:interleaved, :constant)
        est_id, att_id, _, _ = run_case(:interleaved, :dynamic)

        @test att_bc ≈ 1.6667 atol = 1e-3
        @test est_bc ≈ 2.7805 atol = 1e-3
        @test att_bd ≈ 4.1667 atol = 1e-3
        @test est_bd ≈ 1.8049 atol = 1e-3
        @test att_ic ≈ 1.6622 atol = 1e-3
        @test est_ic ≈ 2.0000 atol = 1e-3
        @test att_id ≈ 4.1554 atol = 1e-3
        @test est_id ≈ 3.4397 atol = 1e-3

        # The documented qualitative claims, which are what a reader
        # takes away: the bias is large, its sign is not predictable,
        # and spatially contiguous cohorts are the worst case.
        @test est_bc > att_bc            # too high with constant effects
        @test est_bd < att_bd            # too low with growing ones
        @test abs(est_bd - att_bd) / att_bd > 0.5
        @test abs(est_bd - att_bd) > abs(est_id - att_id)
        # Inside the hull in every case -- a weak guarantee, since the
        # hull is [1, 8] here, and the docstring says so.
        @test lo_bd <= est_bd <= hi_bd
        @test (hi_bd - lo_bd) >= 7.0
    end

    @testset "34f. covariance estimators and their coverage" begin
        side = 12
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        em = exposure(D, W)
        E = em.E[:, 1]
        γ, τ, δ = 0.7, 2.5, -1.25
        μ = collect(range(-1.0, 1.0, length = N))

        rng = MersenneTwister(4242)
        y = zeros(N, 2)
        y[:, 1] .= μ .+ randn(rng, N)
        y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ randn(rng, N)

        # Every estimator must produce a usable, symmetric, PSD
        # covariance and a positive standard error on each effect.
        m_ols = fit_spatial_did(y, D, W; vcov = :ols)
        m_clu = fit_spatial_did(y, D, W; vcov = :cluster)
        m_con = fit_spatial_did(y, D, W; vcov = :conley,
                                coords = coords, dist_cutoff = 2.5)
        for m in (m_ols, m_clu, m_con)
            V = vcov(m)
            @test maximum(abs.(V .- V')) < 1e-10
            @test all(diag(V) .>= 0)
            @test minimum(eigvals(Symmetric(V))) > -1e-10
            @test all(>(0), (m.att_direct_se, m.att_spill_treated_se,
                             m.att_spill_control_se, m.att_total_treated_se))
            # Point estimates do not depend on the covariance.
            @test m.att_direct ≈ m_ols.att_direct rtol = 1e-12
        end
        @test occursin("cluster", m_clu.vcov_name)
        @test occursin("Conley", m_con.vcov_name)
        @test occursin("OLS", m_ols.vcov_name)

        # Default is :cluster, on measured coverage rather than
        # convention -- see the fit_spatial_did docstring.
        m_def = fit_spatial_did(y, D, W)
        @test m_def.vcov ≈ m_clu.vcov rtol = 1e-12

        # Clustering on unit in a first-differenced cross-section is a
        # singleton-cluster design, so it coincides with the
        # heteroskedasticity-robust form. Checked directly, because the
        # docstring says so.
        Z = hcat(ones(N), D, E)
        r = m_clu.residuals
        bread = inv(Symmetric(Z' * Z))
        meat = zeros(3, 3)
        for i in 1:N
            zi = Z[i, :]
            meat .+= (zi * zi') .* r[i]^2
        end
        c_adj = N / (N - 1) * (N - 1) / (N - 3)
        @test m_clu.vcov ≈ bread * (c_adj .* meat) * bread rtol = 1e-10

        # :conley without what it needs must say what is missing.
        @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :conley)
        @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :conley,
                                                   coords = coords)
        @test_throws ArgumentError fit_spatial_did(y, D, W; vcov = :nonsense)
        # Driscoll-Kraay needs more than one period; a differenced
        # two-period design has one.
        @test_throws ArgumentError fit_spatial_did(y, D, W;
                                                   vcov = DriscollKraayVCOV())

        # A spec instance is accepted as well as a symbol.
        m_spec = fit_spatial_did(y, D, W;
                                 vcov = ConleySpatialHAC(coords, 2.5; max_time_lag = 0))
        @test m_spec.vcov ≈ m_con.vcov rtol = 1e-10

        # Coverage, measured. These are the figures the docstring
        # quotes; the reps are kept modest so the suite stays quick, so
        # the bands are wide enough to be stable but still fail if an
        # estimator breaks.
        ρ_err = 0.6
        A = sparse(1.0I, N, N) - ρ_err * W
        Afac = lu(A)
        z95 = 1.959963984540054
        nreps = 300
        cov_c = 0; cov_o = 0; cov_n15 = 0; cov_n60 = 0
        rng2 = MersenneTwister(9091)
        for _ in 1:nreps
            u0 = Afac \ randn(rng2, N)
            u1 = Afac \ randn(rng2, N)
            yy = zeros(N, 2)
            yy[:, 1] .= μ .+ u0
            yy[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ u1
            mc = fit_spatial_did(yy, D, W; vcov = :cluster)
            abs(mc.att_direct - τ) <= z95 * mc.att_direct_se && (cov_c += 1)
            mo = fit_spatial_did(yy, D, W; vcov = :ols)
            abs(mo.att_direct - τ) <= z95 * mo.att_direct_se && (cov_o += 1)
            m15 = fit_spatial_did(yy, D, W; vcov = :conley,
                                  coords = coords, dist_cutoff = 1.5)
            abs(m15.att_direct - τ) <= z95 * m15.att_direct_se && (cov_n15 += 1)
            m60 = fit_spatial_did(yy, D, W; vcov = :conley,
                                  coords = coords, dist_cutoff = 6.0)
            abs(m60.att_direct - τ) <= z95 * m60.att_direct_se && (cov_n60 += 1)
        end
        # Near nominal.
        @test 0.90 <= cov_c / nreps <= 0.99
        @test 0.90 <= cov_o / nreps <= 0.99
        @test 0.89 <= cov_n15 / nreps <= 0.99
        # The documented failure: a wide Conley cutoff under-covers
        # badly, and by more than the narrow one.
        @test cov_n60 / nreps < 0.92
        @test cov_n60 < cov_n15
    end

    @testset "34g. the identification guard actually fires" begin
        side = 8
        N = side^2
        coords = grid(side)
        W = make_rook_weights(side)

        # Treat a checkerboard: every control borders a treated unit
        # and exposure among controls is constant, so δ is absorbed by
        # the intercept. The design is rank deficient and must be
        # refused by name rather than silently fitted.
        D = zeros(N)
        for i in 1:N
            isodd(Int(coords[i, 1]) + Int(coords[i, 2])) && (D[i] = 1.0)
        end
        em = exposure(D, W)
        rep = check_interference(em)
        # On a rook lattice a checkerboard makes every neighbour of a
        # control treated, so exposure is 1 for controls and 0 for the
        # treated -- exactly collinear with D.
        @test rep.n_control_unexposed == 0
        @test all(em.E[D .== 0.0, 1] .≈ 1.0)

        μ = collect(range(-1.0, 1.0, length = N))
        y = zeros(N, 2)
        y[:, 1] .= μ
        y[:, 2] .= μ .+ 0.7 .+ 2.5 .* D
        err = nothing
        try
            fit_spatial_did(y, D, W; vcov = :ols)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("not identified", err.msg)
        @test occursin("fit_spatial_did", err.msg)

        # A design with no unexposed control but genuine variation is
        # still identified, and the report says what the contrast is.
        Wq = make_queen_weights(side)
        Dq = zeros(N)
        for i in 1:N
            r, c = coords[i, 1], coords[i, 2]
            (3 <= r <= 4 && 3 <= c <= 4) && (Dq[i] = 1.0)
        end
        repq = check_interference(exposure(Dq, Wq))
        @test repq.identified
    end

    @testset "34h. ring exposure reads a band, not a neighbourhood" begin
        side = 10
        N, coords, D = block_design(side, 4, 6)

        near = ring_exposure(D, coords, 0.0, 1.5)
        far = ring_exposure(D, coords, 1.5, 3.0)
        @test size(near.E) == (N, 1)
        @test !(near.E[:, 1] ≈ far.E[:, 1])

        # The near ring is rook-and-diagonal contiguity, so it must
        # agree with a queen weight matrix's exposure.
        Wq = make_queen_weights(side)
        @test near.E[:, 1] ≈ exposure(D, Wq).E[:, 1] rtol = 1e-12

        # A band that reaches nothing leaves exposure at zero, and the
        # interference report must call that unidentified rather than
        # returning a spillover built on no variation.
        none = ring_exposure(D, coords, 20.0, 25.0)
        @test all(iszero, none.E)
        @test !check_interference(none).identified

        @test_throws DimensionMismatch ring_exposure(zeros(N + 1), coords, 0.0, 1.5)
    end

    @testset "34i. vcov_cluster on its own terms" begin
        # The estimator underneath, checked against the formula rather
        # than through the DiD. A covariance is a second-order quantity,
        # which is where audit findings 1-7 lived.
        rng = MersenneTwister(31337)
        n, K, G = 120, 3, 12
        Z = hcat(ones(n), randn(rng, n, K - 1))
        r = randn(rng, n)
        groups = repeat(1:G, inner = div(n, G))

        spec = ClusterRobustVCOV(groups)
        V = vcov_cluster(Z, r, spec)

        bread = inv(Symmetric(Z' * Z))
        S = zeros(K, G)
        for i in 1:n
            S[:, groups[i]] .+= Z[i, :] .* r[i]
        end
        c_adj = G / (G - 1) * (n - 1) / (n - K)
        @test V ≈ bread * (c_adj .* (S * S')) * bread rtol = 1e-12
        @test maximum(abs.(V .- V')) < 1e-10
        @test minimum(eigvals(Symmetric(V))) > -1e-10

        # Without the correction the scaling is exactly c_adj smaller.
        V0 = vcov_cluster(Z, r, ClusterRobustVCOV(groups; dof_correction = false))
        @test V0 ≈ V ./ c_adj rtol = 1e-12

        # Singleton clusters reduce to the heteroskedasticity-robust
        # form, which is the trap the docstring names.
        Vsing = vcov_cluster(Z, r, ClusterRobustVCOV(1:n))
        meat = zeros(K, K)
        for i in 1:n
            zi = Z[i, :]
            meat .+= (zi * zi') .* r[i]^2
        end
        c_hc = n / (n - 1) * (n - 1) / (n - K)
        @test Vsing ≈ bread * (c_hc .* meat) * bread rtol = 1e-12

        # Labels need not be integers or contiguous.
        glab = [string("g", g) for g in groups]
        @test vcov_cluster(Z, r, ClusterRobustVCOV(glab)) ≈ V rtol = 1e-12

        # Group order must not matter: permuting rows permutes nothing.
        perm = randperm(MersenneTwister(5), n)
        Vp = vcov_cluster(Z[perm, :], r[perm], ClusterRobustVCOV(groups[perm]))
        @test Vp ≈ V rtol = 1e-10

        @test_throws DimensionMismatch vcov_cluster(Z, r[1:(n - 1)], spec)
        @test_throws DimensionMismatch vcov_cluster(Z, r, ClusterRobustVCOV(groups[1:(n - 1)]))
        @test_throws ArgumentError ClusterRobustVCOV(Int[])
        # One group, and a saturated design, both divide by zero under
        # the correction and are named rather than returning Inf.
        @test_throws ArgumentError vcov_cluster(Z, r, ClusterRobustVCOV(ones(Int, n)))
        @test_throws ArgumentError vcov_cluster(Z[1:K, :], r[1:K],
                                                ClusterRobustVCOV(1:K))
    end

    @testset "34j. covariates enter without disturbing the effects" begin
        side = 10
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        em = exposure(D, W)
        E = em.E[:, 1]
        γ, τ, δ = 0.7, 2.5, -1.25
        μ = collect(range(-1.0, 1.0, length = N))
        βx = [1.5, -0.75]
        rng = MersenneTwister(5)

        # Two-period: X is N x K x 2, and differencing it is part of
        # the transform rather than something the caller must do.
        X = randn(rng, N, 2, 2)
        y = zeros(N, 2)
        y[:, 1] .= μ .+ X[:, :, 1] * βx
        y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ X[:, :, 2] * βx

        m = fit_spatial_did(y, D, W; X = X, vcov = :ols)
        @test m.coefnames == ["(Intercept)", "D", "WD", "ΔX1", "ΔX2"]
        @test m.coef[2] ≈ τ rtol = 1e-11
        @test m.coef[3] ≈ δ rtol = 1e-11
        @test m.coef[4] ≈ βx[1] rtol = 1e-11
        @test m.coef[5] ≈ βx[2] rtol = 1e-11
        @test norm(m.residuals) < 1e-10
        # The effects are still read off the same slots, so adding
        # covariates must not shift which coefficient is which.
        @test m.att_direct ≈ τ rtol = 1e-11
        @test m.att_spill_control ≈ δ * mean(E[D .== 0.0]) rtol = 1e-11

        # A time-invariant covariate differences to exactly zero. It is
        # refused with the reason, rather than silently dropped or left
        # to surface as a rank deficiency the caller has to decode.
        err = nothing
        try
            fit_spatial_did(y, D, W; X = randn(rng, N, 2), vcov = :ols)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("time-invariant", err.msg)
        @test_throws ArgumentError fit_spatial_did(y, D, W;
                                                   X = randn(rng, N, 2, 3), vcov = :ols)

        # Staggered: X is N x K x T and enters the within transform.
        T_per = 5
        Wt = row_standardize(W)
        adopt = fill(0, N)
        for i in 1:N
            c = coords[i, 2]
            c <= 3 ? (adopt[i] = 2) : c <= 6 ? (adopt[i] = 4) : nothing
        end
        Dp = zeros(N, T_per)
        for i in 1:N, t in 1:T_per
            adopt[i] != 0 && t >= adopt[i] && (Dp[i, t] = 1.0)
        end
        Xs = randn(rng, N, 2, T_per)
        λ = [0.0, 0.2, -0.1, 0.3, 0.15]
        ys = zeros(N, T_per)
        for i in 1:N, t in 1:T_per
            E_it = dot(view(Wt, i, :), view(Dp, :, t))
            ys[i, t] = μ[i] + λ[t] + τ * Dp[i, t] + δ * E_it +
                       dot(view(Xs, i, :, t), βx)
        end
        ms = fit_spatial_did(ys, Dp, W; X = Xs, vcov = :ols)
        @test ms.coefnames == ["D", "WD", "X1", "X2"]
        @test ms.coef[1] ≈ τ rtol = 1e-11
        @test ms.coef[2] ≈ δ rtol = 1e-11
        @test ms.coef[3] ≈ βx[1] rtol = 1e-11
        @test ms.coef[4] ≈ βx[2] rtol = 1e-11
        @test norm(ms.residuals) < 1e-10

        @test_throws DimensionMismatch fit_spatial_did(ys, Dp, W;
                                                       X = randn(rng, N, 2, T_per + 1),
                                                       vcov = :ols)
        @test_throws ArgumentError fit_spatial_did(ys, Dp, W;
                                                   X = randn(rng, N, 2), vcov = :ols)
    end

    @testset "34k. two spillover channels (Chagas et al. eq. 7)" begin
        side = 12
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        Wt = row_standardize(W)
        E = Wt * D
        ET, EC = D .* E, (1 .- D) .* E
        γ, τ, δT, δC = 0.7, 2.5, -0.6, 1.3
        μ = collect(range(-1.0, 1.0, length = N))
        Ē_T, Ē_C = mean(E[D .== 1.0]), mean(E[D .== 0.0])

        # Noiseless: the generating values, to machine precision.
        y = hcat(μ, μ .+ γ .+ τ .* D .+ δT .* ET .+ δC .* EC)
        m = fit_spatial_did(y, D, W; vcov = :ols, channels = :split)
        @test m.coefnames == ["(Intercept)", "D", "D×WD", "(1-D)×WD"]
        @test m.coef ≈ [γ, τ, δT, δC] rtol = 1e-12
        @test m.att_direct ≈ τ rtol = 1e-12
        @test m.att_spill_treated ≈ δT * Ē_T rtol = 1e-12
        @test m.att_spill_control ≈ δC * Ē_C rtol = 1e-12
        @test m.att_total_treated == m.att_direct + m.att_spill_treated
        @test m.channels === :split
        @test norm(m.residuals) < 1e-10
        # The pooled fit on the same data is misspecified, and its
        # control spillover is not δ_C·Ē_C: this is what the split is for.
        mp = fit_spatial_did(y, D, W; vcov = :ols)
        @test mp.channels === :pooled
        @test abs(mp.att_spill_control - δC * Ē_C) > 0.2 * δC * Ē_C
        @test isnan(mp.spillover_equality.stat) && mp.spillover_equality.df == 0

        # With noise: the fit is OLS on [1, D, D∘E, (1−D)∘E], and the
        # effects are the stated linear forms in it.
        rng = MersenneTwister(34011)
        yn = hcat(μ .+ randn(rng, N), μ .+ γ .+ τ .* D .+ δT .* ET .+ δC .* EC .+ randn(rng, N))
        ms = fit_spatial_did(yn, D, W; vcov = :ols, channels = :split)
        Z = hcat(ones(N), D, ET, EC)
        Δy = yn[:, 2] .- yn[:, 1]
        θ = Z \ Δy
        r = Δy .- Z * θ
        s2 = dot(r, r) / (N - 4)
        @test ms.coef ≈ θ rtol = 1e-10
        @test ms.vcov ≈ s2 .* inv(Z' * Z) rtol = 1e-10
        @test ms.att_spill_treated ≈ θ[3] * Ē_T rtol = 1e-10
        @test ms.att_spill_control ≈ θ[4] * Ē_C rtol = 1e-10
        @test ms.att_spill_control_se ≈ Ē_C * sqrt(ms.vcov[4, 4]) rtol = 1e-10
        a = [0.0, 1.0, Ē_T, 0.0]
        @test ms.att_total_treated_se ≈ sqrt(dot(a, ms.vcov, a)) rtol = 1e-10

        # The Wald test. Under the homoskedastic vcov it equals the
        # restricted-against-unrestricted sum-of-squares statistic
        # exactly, the pooled fit being the restricted model.
        t = ms.spillover_equality
        rp = fit_spatial_did(yn, D, W; vcov = :ols).residuals
        @test t.df == 1
        @test t.stat ≈ (dot(rp, rp) - dot(r, r)) / s2 rtol = 1e-9
        @test t.p_value ≈ ccdf(Chisq(1), t.stat) rtol = 1e-9
        # Under another vcov it is (δ̂_T − δ̂_C)² over its variance.
        mc = fit_spatial_did(yn, D, W; channels = :split)
        V = vcov(mc)
        @test mc.spillover_equality.stat ≈
              (mc.coef[3] - mc.coef[4])^2 / (V[3, 3] + V[4, 4] - 2V[3, 4]) rtol = 1e-10
        @test occursin("split", sprint(show, mc))
        @test occursin("equal channels", sprint(show, mc))
        @test !occursin("equal channels", sprint(show, fit_spatial_did(yn, D, W)))

        # Order 2: one coefficient per order and channel, and a joint
        # test with two degrees of freedom.
        m2 = fit_spatial_did(yn, D, W; channels = :split, order = 2)
        @test m2.coefnames == ["(Intercept)", "D", "D×WD", "D×W2D", "(1-D)×WD", "(1-D)×W2D"]
        @test m2.spillover_equality.df == 2
        E2 = Wt * E
        @test m2.att_spill_control ≈ m2.coef[5] * Ē_C + m2.coef[6] * mean(E2[D .== 0.0]) rtol = 1e-10

        # Staggered, with treatment switching on and off so that every
        # unit is treated in some period, as in Chagas et al.'s data:
        # the regression identifies off timing, and the report counts
        # unit-periods.
        T_per = 6
        rngs = MersenneTwister(34012)
        Dp = zeros(N, T_per)
        for t in 2:T_per, i in 1:N
            Dp[i, t] = rand(rngs) < 0.5 ? 1.0 : 0.0
        end
        Dp[:, 2] .= 1.0                         # everyone treated at least once
        λ = randn(rngs, T_per)
        ys = [μ[i] + λ[t] + τ * Dp[i, t] +
              (Dp[i, t] * δT + (1 - Dp[i, t]) * δC) * dot(view(Wt, i, :), view(Dp, :, t))
              for i in 1:N, t in 1:T_per]
        mst = fit_spatial_did(ys, Dp, W; vcov = :ols, channels = :split)
        @test mst.design === :staggered
        @test mst.coefnames == ["D", "D×WD", "(1-D)×WD"]
        @test mst.coef ≈ [τ, δT, δC] rtol = 1e-11
        @test mst.interference.n_treated == count(isone, Dp)
        @test mst.interference.n_treated + mst.interference.n_control == N * T_per
        @test occursin("unit-periods", sprint(show, mst))
        # Pooled works on the same all-ever-treated design too.
        @test fit_spatial_did(ys, Dp, W; vcov = :ols).design === :staggered
        @test_throws ArgumentError fit_spatial_did(ys, zeros(N, T_per), W)
        @test_throws ArgumentError fit_spatial_did(ys, ones(N, T_per), W)

        # The treated channel needs exposure to vary among the treated.
        # Isolated treated units (no treated neighbour) leave D∘E = 0:
        # pooled fits, split is refused with the reason.
        Diso = zeros(N)
        for i in 1:N
            r_, c_ = Int(coords[i, 1]), Int(coords[i, 2])
            (mod(r_, 4) == 1 && mod(c_, 4) == 1) && (Diso[i] = 1.0)
        end
        @test check_interference(exposure(Diso, W)).exposure_var_treated == 0
        yi = hcat(μ .+ randn(rng, N), μ .+ γ .+ τ .* Diso .+ randn(rng, N))
        @test fit_spatial_did(yi, Diso, W; vcov = :ols).channels === :pooled
        err = nothing
        try
            fit_spatial_did(yi, Diso, W; vcov = :ols, channels = :split)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("channels = :split", err.msg)
        @test_throws ArgumentError fit_spatial_did(yn, D, W; channels = :bogus)

        # Size and power of the equality test, measured. Two-period,
        # heteroskedastic errors, default vcov; the null and alternative
        # share draws. Over 20 other seeds: size 0.053 to 0.088 on Julia
        # 1.13.4 and 0.050 to 0.098 on 1.10.13 (means 0.069 and 0.075:
        # HC1 is slightly oversized at N = 144); power 0.85 to 0.89 and
        # 0.79 to 0.88.
        nrep = 400
        rej0 = 0; rej1 = 0
        Dsc = (rr = MersenneTwister(77); [rand(rr) < 0.3 ? 1.0 : 0.0 for _ in 1:N])
        Esc = Wt * Dsc
        sdv = 1 .+ Dsc
        rngm = MersenneTwister(34013)
        for _ in 1:nrep
            u = sdv .* randn(rngm, N)
            base = μ .+ γ .+ τ .* Dsc .+ u
            y0 = hcat(μ, base .+ 1.0 .* Esc)                                    # δ_T = δ_C
            y1 = hcat(μ, base .+ (-2.0) .* Dsc .* Esc .+ 2.0 .* (1 .- Dsc) .* Esc) # δ_T ≠ δ_C
            rej0 += fit_spatial_did(y0, Dsc, W; channels = :split).spillover_equality.p_value < 0.05
            rej1 += fit_spatial_did(y1, Dsc, W; channels = :split).spillover_equality.p_value < 0.05
        end
        @test 0.03 <= rej0 / nrep <= 0.12
        @test rej1 / nrep >= 0.7
    end

    @testset "34l. spatial-error ML (errors = :spatial)" begin
        side = 12
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        Wt = row_standardize(W)
        E = Wt * D
        rng = MersenneTwister(34021)
        A = lu(sparse(1.0I, N, N) - 0.5 * W)
        y = hcat(A \ randn(rng, N),
                 0.7 .+ 2.5 .* D .- 0.6 .* D .* E .+ 1.3 .* (1 .- D) .* E .+ A \ randn(rng, N))

        # Two-period: the cross-section SEM on the differenced design,
        # exactly, for both channel settings.
        for (ch, Xd) in ((:split, hcat(D, D .* E, (1 .- D) .* E)), (:pooled, hcat(D, E)))
            m = fit_spatial_did(y, D, W; channels = ch, errors = :spatial)
            h = fit_spatial_error(Wt, Xd, y[:, 2] .- y[:, 1])
            p = size(Xd, 2) + 1
            @test m.coef == coef(h)[1:p]
            @test m.vcov == vcov(h)[1:p, 1:p]
            @test m.lambda == h.lambda && m.lambda_se == h.lambda_se
            @test m.errors === :spatial
            @test occursin("spatial error ML", m.vcov_name)
            Ē_C = mean(E[D .== 0.0])
            @test m.att_spill_control ≈ m.coef[end] * Ē_C rtol = 1e-12
        end
        ms = fit_spatial_did(y, D, W; channels = :split, errors = :spatial)
        @test occursin("spatial error (ML): λ", sprint(show, ms))
        mi = fit_spatial_did(y, D, W)
        @test mi.errors === :iid && isnan(mi.lambda) && isnan(mi.lambda_se)

        # Staggered: the two-way fixed-effects panel SEM on the stacked
        # levels of the same design.
        T_per = 5
        Dp = Float64.(rand(rng, N, T_per) .< 0.5)
        ys = [randn(rng) + 1.0 * Dp[i, t] + 0.4 * dot(view(Wt, i, :), view(Dp, :, t))
              for i in 1:N, t in 1:T_per]
        mst = fit_spatial_did(ys, Dp, W; channels = :split, errors = :spatial)
        Xs = zeros(N * T_per, 3); yv = zeros(N * T_per)
        for t in 1:T_per, i in 1:N
            r = (t - 1) * N + i
            e = dot(view(Wt, i, :), view(Dp, :, t))
            Xs[r, :] .= (Dp[i, t], Dp[i, t] * e, (1 - Dp[i, t]) * e)
            yv[r] = ys[i, t]
        end
        hs = fit_spatial_error(TwoWayFE(), Wt, Xs, yv, N, T_per)
        @test mst.coef == coef(hs)[1:3]
        @test mst.vcov == vcov(hs)[1:3, 1:3]
        @test mst.lambda == hs.lambda

        # What it refuses: a vcov it would ignore, an unknown error
        # model, and an unstandardised W.
        @test_throws ArgumentError fit_spatial_did(y, D, W; errors = :spatial, vcov = :ols)
        @test_throws ArgumentError fit_spatial_did(y, D, W; errors = :bogus)
        @test_throws ArgumentError fit_spatial_did(y, D, W; errors = :spatial,
                                                   standardize = false)

        # Coverage of the total effect on the treated, measured, under
        # spatially correlated errors and a contiguous treated block:
        # OLS with the default vcov under-covers here and the ML fit
        # does not. The two share draws. Over 20 other seeds: OLS 0.64
        # to 0.75 on Julia 1.13.4 and 0.64 to 0.74 on 1.10.13; ML 0.90 to
        # 0.95 on both (mean 0.93: ML is slightly short of nominal at
        # N = 144).
        nrep = 300
        cov_iid = 0; cov_sp = 0
        τ, δ = 1.0, 0.8
        truth = τ + δ * mean(E[D .== 1.0])
        A6 = lu(sparse(1.0I, N, N) - 0.6 * W)
        rngm = MersenneTwister(34022)
        for _ in 1:nrep
            yy = hcat(A6 \ randn(rngm, N), 0.5 .+ τ .* D .+ δ .* E .+ A6 \ randn(rngm, N))
            a = fit_spatial_did(yy, D, W; channels = :split)
            b = fit_spatial_did(yy, D, W; channels = :split, errors = :spatial)
            cov_iid += abs(a.att_total_treated - truth) <= 1.959963984540054 * a.att_total_treated_se
            cov_sp += abs(b.att_total_treated - truth) <= 1.959963984540054 * b.att_total_treated_se
        end
        @test 0.88 <= cov_sp / nrep <= 0.99
        @test cov_iid / nrep <= 0.80
    end

    @testset "34m. spatial-error GMM (errors = :spatial_gmm)" begin
        SEMm = SpatialEconometrics.SpatialErrorModel
        side = 12
        N, coords, D = block_design(side, 3, 6)
        W = make_rook_weights(side)
        Wt = row_standardize(W)
        E = Wt * D
        rng = MersenneTwister(34031)
        A = lu(sparse(1.0I, N, N) - 0.5 * W)
        sdv = 1 .+ D
        y = hcat(A \ (sdv .* randn(rng, N)),
                 0.7 .+ 2.5 .* D .- 0.6 .* D .* E .+ 1.3 .* (1 .- D) .* E .+ A \ (sdv .* randn(rng, N)))

        # Two-period: Kelejian & Prucha's cross-section on the
        # differenced design, the same fit as `fit_spatial_error`'s
        # `estimator = :gmm`, exactly, for both channel settings.
        for (ch, Xd) in ((:split, hcat(D, D .* E, (1 .- D) .* E)), (:pooled, hcat(D, E)))
            m = fit_spatial_did(y, D, W; channels = ch, errors = :spatial_gmm)
            h = fit_spatial_error(Wt, Xd, y[:, 2] .- y[:, 1]; estimator = :gmm)
            p = size(Xd, 2) + 1
            @test m.coef == coef(h)[1:p]
            @test m.vcov == vcov(h)[1:p, 1:p]
            @test m.lambda == h.lambda && m.lambda_se == h.lambda_se
            @test m.errors === :spatial_gmm
            @test occursin("spatial error GMM", m.vcov_name)
        end
        mg = fit_spatial_did(y, D, W; channels = :split, errors = :spatial_gmm)
        @test occursin("spatial error (GMM): λ", sprint(show, mg))

        # Staggered: the filter acts on the levels and the two-way
        # transform after it, so β̂ is OLS on demean((I − λW)·levels) at
        # the step-1 λ, which is checked by hand.
        T_per = 5
        Dp = Float64.(rand(rng, N, T_per) .< 0.5)
        ys = [randn(rng) * (1 + Dp[i, t]) + 1.0 * Dp[i, t] + 0.4 * dot(view(Wt, i, :), view(Dp, :, t))
              for i in 1:N, t in 1:T_per]
        mst = fit_spatial_did(ys, Dp, W; channels = :split, errors = :spatial_gmm)
        Xs = zeros(N * T_per, 3); yv = zeros(N * T_per)
        for t in 1:T_per, i in 1:N
            r = (t - 1) * N + i
            e = dot(view(Wt, i, :), view(Dp, :, t))
            Xs[r, :] .= (Dp[i, t], Dp[i, t] * e, (1 - Dp[i, t]) * e)
            yv[r] = ys[i, t]
        end
        o = SEMm._sem_gmm_het(Wt, Xs, yv, N, T_per; fe = true)
        @test mst.coef == o.beta && mst.vcov == o.vcov_beta && mst.lambda == o.lambda
        λ1 = o.lambda_initial
        lag(M) = SpatialEconometrics.SpatialUtils.panel_spatial_lag(Wt, M, N, T_per)
        dmz(M) = SpatialEconometrics.SpatialUtils.within_transform(M, N, T_per, TwoWayFE())
        Xf = dmz(Xs .- λ1 .* lag(Xs))
        yf = dmz(reshape(yv .- λ1 .* lag(reshape(yv, :, 1))[:, 1], :, 1))[:, 1]
        @test o.beta ≈ Xf \ yf rtol = 1e-10

        # What it refuses.
        @test_throws ArgumentError fit_spatial_did(y, D, W; errors = :spatial_gmm, vcov = :ols)
        @test_throws ArgumentError fit_spatial_did(y, D, W; errors = :spatial_gmm,
                                                   standardize = false)

        # Coverage of the total effect on the treated, measured, with a
        # contiguous treated block, spatially correlated errors and the
        # error sd doubled on the treated: ML under-covers and GMM
        # does not. The two share draws. Over 20 other seeds: ML 0.68 to
        # 0.76 on Julia 1.13.4 and on 1.10.13; GMM 0.88 to 0.95 and 0.87
        # to 0.95 (means 0.92: GMM is slightly short of nominal at
        # N = 144).
        nrep = 300
        cov_ml = 0; cov_gm = 0
        τ, δ = 1.0, 0.8
        truth = τ + δ * mean(E[D .== 1.0])
        A6 = lu(sparse(1.0I, N, N) - 0.6 * W)
        rngm = MersenneTwister(34032)
        for _ in 1:nrep
            yy = hcat(A6 \ (sdv .* randn(rngm, N)),
                      0.5 .+ τ .* D .+ δ .* E .+ A6 \ (sdv .* randn(rngm, N)))
            a = fit_spatial_did(yy, D, W; channels = :split, errors = :spatial)
            b = fit_spatial_did(yy, D, W; channels = :split, errors = :spatial_gmm)
            cov_ml += abs(a.att_total_treated - truth) <= 1.959963984540054 * a.att_total_treated_se
            cov_gm += abs(b.att_total_treated - truth) <= 1.959963984540054 * b.att_total_treated_se
        end
        @test 0.85 <= cov_gm / nrep <= 0.99
        @test cov_ml / nrep <= 0.82
    end
end
