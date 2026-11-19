@testset "23. Regressions found by the full sweep" begin
    _progress("23. Regressions found by the full sweep")
    W = make_rook_weights(10); N = size(W, 1)
    IN = sparse(1.0I, N, N)
    rng = MersenneTwister(606)
    X = randn(rng, N, 2)
    y = (IN - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))
    coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
    CSx = SpatialEconometrics.CrossSection
    SPx = SpatialEconometrics.StaticPanel

    # ── robust_coeftable must be callable ──
    # It lives in the umbrella module, which has both `using StatsAPI` and
    # `using StatsModels`. That makes the bare name `vcov` ambiguous, so it
    # resolved to nothing and every call raised UndefVarError.
    m_sar = fit_spatial(CSx.SAR(), W, X, y; varnames=["x1", "x2"])
    rt = robust_coeftable(m_sar, ConleySpatialHAC(coords, 3.0))
    @test !isempty(sprint(show, rt))
    @test length(rt.coef) == length(coef(m_sar))
    @test length(rt.se_robust) == length(coef(m_sar))
    @test all(rt.se_robust .>= 0.0)
    @test all(isfinite, rt.se_robust)
    # Robust SEs differ from the model's own, but not wildly.
    @test rt.se_baseline ≈ stderror(m_sar) rtol = 1e-10
    @test all(rt.se_robust .< 50 .* rt.se_baseline)
    # t-stats must be coef / robust se.
    @test rt.z_robust ≈ coef(m_sar) ./ rt.se_robust rtol = 1e-10
    # The cross-section path shares the misscaled bread described in
    # testset 7, so it gets the same exact scale check: scaling y by c
    # scales every coefficient's robust standard error by exactly c and
    # leaves rho's alone.
    m_sar_c = fit_spatial(CSx.SAR(), W, X, 3.0 .* y; varnames=["x1", "x2"])
    se_cs_1 = rt.se_robust
    se_cs_c = robust_coeftable(m_sar_c, ConleySpatialHAC(coords, 3.0)).se_robust
    @test se_cs_c[1:(end - 1)] ≈ 3.0 .* se_cs_1[1:(end - 1)] rtol = 1e-4
    @test se_cs_c[end] ≈ se_cs_1[end] rtol = 1e-4

    # Panel variants exercise the other robust_vcov methods.
    Lp = 7; Np = Lp * Lp; Tp = 4; NTp = Np * Tp
    Wp = make_rook_weights(Lp)
    Xp = randn(rng, NTp, 2)
    Slu = lu(sparse(1.0I, Np, Np) - 0.4 * Wp)
    mu = randn(rng, Np)
    yp = zeros(NTp)
    for t in 1:Tp
        idx = ((t-1)*Np+1):(t*Np)
        yp[idx] = Slu \ (mu .+ Xp[idx, :] * [1.5, -0.8] .+ 0.5 .* randn(rng, Np))
    end
    m_pan = fit_spatial_panel(SPx.SAR(), SPx.TwoWayFE(), Wp, Xp, yp, Np, Tp)
    coords_p = hcat(rand(rng, Np) .* 10, rand(rng, Np) .* 10)
    for spec in (DriscollKraayVCOV(), ConleySpatialHAC(coords_p, 3.0))
        rtp = robust_coeftable(m_pan, spec)
        @test !isempty(sprint(show, rtp))
        @test all(isfinite, rtp.se_robust)
        @test all(rtp.se_robust .>= 0.0)
    end

    # ── balance_gras must be differentiable ──
    # The dual path factorized with `ldlt(Symmetric(dense))`, which has no
    # method, so every ForwardDiff call through balance_gras threw.
    X0 = [10.0 -2.0 5.0; 3.0 12.0 -1.0; 6.0 4.0 9.0]
    u = [14.0, 15.0, 20.0]
    v = [18.0, 15.0, 16.0]
    v = v .* (sum(u) / sum(v))

    # GRAS pins row sums to u, so sum(X) == sum(u) and d sum(X)/du is exactly 1.
    g = ForwardDiff.gradient(uu -> sum(balance_gras(X0, uu, v)), u)
    @test all(isfinite, g)
    @test g ≈ ones(3) atol = 1e-8

    # The reverse-mode rule must agree with the primal and the same analytic gradient.
    yb, pb = ChainRulesCore.rrule(balance_gras, X0, u, v)
    @test yb ≈ balance_gras(X0, u, v) atol = 1e-10
    dvals = pb(ones(size(yb)))
    @test collect(dvals[3]) ≈ ones(3) atol = 1e-8

    # Gradient wrt X0 must match central differences.
    f(x) = sum(abs2, balance_gras(reshape(x, 3, 3), u, v))
    x0 = vec(X0)
    g_ad = ForwardDiff.gradient(f, x0)
    g_fd = similar(g_ad)
    for i in eachindex(x0)
        h = 1e-6
        xp2 = copy(x0); xp2[i] += h
        xm2 = copy(x0); xm2[i] -= h
        g_fd[i] = (f(xp2) - f(xm2)) / (2h)
    end
    @test maximum(abs.(g_ad .- g_fd) ./ max.(abs.(g_fd), 1e-8)) < 1e-5

    # ── power_analysis rates are real rates ──
    # Each count was divided by n_reps even when a diagnostic threw, so power
    # was understated and an all-failure run reported 0.0, which is
    # indistinguishable from a test having no power.
    pa = power_analysis(r -> SAR_DGP(r, [1.5, -0.8], 0.5), [6, 8], [0.0, 0.6];
                        n_reps = 30, seed = 3)
    for M in (pa.power_moran, pa.power_lm_lag, pa.power_lm_error)
        @test all(isfinite, M)                      # NaN only if nothing ran
        @test all(0.0 .<= M .<= 1.0)
        # Power must rise with rho, and the rho=0 column is the test's size.
        @test M[1, 1] < 0.35
        @test M[end, end] > M[end, 1]
    end

    # ── one chi-squared survival function, shared ──
    # There used to be three copies, which is how one of them came to be
    # wrong: a truncated lower-gamma series reported p = 0.9977 for an LR of
    # 499.5 on 3 df. They now resolve to a single definition in SpatialUtils,
    # so identity — not numerical agreement — is the invariant to pin.
    UTx = SpatialEconometrics.SpatialUtils
    MSx = SpatialEconometrics.SpatialModelSelection
    PDx = SpatialEconometrics.SpatialPanelDiagnostics
    RMx = SpatialEconometrics.SpatialRegimeModels
    SSx = SpatialEconometrics.SpatialShiftShare
    NWx = SpatialEconometrics.SpatialNetworks
    @test MSx._chi2_sf === UTx._chi2_sf
    @test PDx._chi2_sf === UTx._chi2_sf
    @test RMx._chi2_sf === UTx._chi2_sf
    # A fourth copy did get added, in SpatialShiftShare, and this test did
    # not catch it because it was spelled `_chisq_sf`. Identity is pinned
    # by module here rather than by name, so a renamed copy fails too.
    @test SSx._chi2_sf === UTx._chi2_sf
    @test NWx._chi2_sf === UTx._chi2_sf
    @test !isdefined(SSx, :_chisq_sf)
    # PD keeps the F / incomplete-beta helpers, so its log Gamma must be the
    # shared one too rather than a fourth private copy.
    @test PDx._loggamma === UTx._loggamma
    @test PDx._f_sf(2.0, 4, 8) ≈ 0.1875 rtol = 1e-9

    for df in (1, 2, 3, 6, 10, 50), x in (0.5, 2.0, 7.815, 25.0, 100.0)
        q = UTx._chi2_sf(x, df)
        @test isfinite(q)
        @test 0.0 <= q <= 1.0
    end
    # Standard critical values, and the degenerate cases.
    @test UTx._chi2_sf(3.841, 1) ≈ 0.05 atol = 5e-5
    @test UTx._chi2_sf(7.815, 3) ≈ 0.05 atol = 5e-5
    @test UTx._chi2_sf(0.0, 3) == 1.0      # nothing to reject
    @test UTx._chi2_sf(10.0, 0) == 1.0     # no restriction to reject
    # The case that exposed the bug: a large LR must stay decisive.
    @test UTx._chi2_sf(499.5, 3) < 1e-100
    # Exact closed form at even df: Q(x; 2k) = exp(-x/2) * sum_{j<k} (x/2)^j / j!
    for (x, df) in [(2.0, 2), (5.0, 4), (10.0, 6), (3.0, 10)]
        k = df ÷ 2; z = x / 2
        want = exp(-z) * sum(z^j / factorial(j) for j in 0:(k-1))
        @test UTx._chi2_sf(x, df) ≈ want rtol = 1e-7
    end

    # ── GAL / GWT readers must preserve the unit ordering ──
    # Both readers assigned matrix indices to ids on FIRST APPEARANCE, so a
    # neighbour mentioned before its own record landed at the wrong index and
    # the matrix came back silently permuted: on a 3x3 rook grid unit 1's
    # neighbours {2, 4} were read back as {2, 3}.
    mktempdir() do dir
        Wio = make_rook_weights(5)
        idv = collect(1:size(Wio, 1))

        galp = joinpath(dir, "rt.gal")
        write_gal(galp, Wio, idv)
        Wg, idg = read_gal(galp)
        @test size(Wg) == size(Wio)
        @test idg == string.(idv)
        # GAL stores adjacency only, so check the pattern as well as the values.
        @test Set(zip(findnz(Wg)[1], findnz(Wg)[2])) ==
              Set(zip(findnz(Wio)[1], findnz(Wio)[2]))
        @test maximum(abs.(Wg .- Wio)) < 1e-12

        gwtp = joinpath(dir, "rt.gwt")
        write_gwt(gwtp, Wio, idv)
        Ww, idw = read_gwt(gwtp)
        @test size(Ww) == size(Wio)
        @test idw == string.(idv)
        @test maximum(abs.(Ww .- Wio)) < 1e-12

        # Non-contiguous ids: GAL must map by record order, and GWT by the
        # ordering it is handed, not by order of appearance in the edge list.
        odd = [101, 7, 55, 3, 99, 42, 88, 12, 76]
        W3 = make_rook_weights(3)
        g3 = joinpath(dir, "odd.gal")
        write_gal(g3, W3, odd)
        W3r, id3 = read_gal(g3)
        @test id3 == string.(odd)
        @test maximum(abs.(W3r .- W3)) < 1e-12

        w3 = joinpath(dir, "odd.gwt")
        write_gwt(w3, W3, odd)
        W3w, _ = read_gwt(w3; ids = odd)
        @test maximum(abs.(W3w .- W3)) < 1e-12

        # A neighbour id with no record of its own is a malformed file, not
        # something to paper over with a silently invented index.
        bad = joinpath(dir, "bad.gal")
        write(bad, "0 2\n1 1\n2\n2 1\n9\n")
        @test_throws ErrorException read_gal(bad)
    end
end
