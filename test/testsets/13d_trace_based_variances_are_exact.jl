@testset "13d. Trace-based variances are exact and reproducible" begin
    _progress("13d. Trace-based variances are exact and reproducible")
    # tr(G), tr(G^2) and tr(G-transpose-G) were estimated by unseeded
    # Hutchinson probes off the global RNG: 250 for the power traces, 30
    # for the last one. So fitting the same model twice gave different
    # standard errors (0.5% apart on the rho SE at N = 81), the fit quietly
    # consumed global RNG state, and since spatial_effects reads the same
    # power traces, the reported impacts moved between runs as well. Of the
    # 21 assertions that touched a standard error, 20 checked only size.
    SU = SpatialEconometrics.SpatialUtils
    L = 8; N = L * L
    W = make_rook_weights(L); Wd = Matrix(W)
    rng = MersenneTwister(1_300_015)
    X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
    y = (I - 0.5 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+ randn(rng, N))

    # Power traces against dense powers of W.
    mu = SU.spatial_trace_powers(W; max_power = 12, max_exact_n = 10_000)
    @test mu[1] == 1.0
    Pm = Matrix{Float64}(I, N, N)
    for m in 1:12
        Pm = Pm * Wd
        @test isapprox(mu[m + 1], tr(Pm) / N; atol = 1e-12)
    end

    # The three G traces against a dense G = W(I - rho*W)^-1, in both the
    # plain and the constant-projected form the panel estimators need.
    for rho in (-0.6, 0.3, 0.8), pc in (false, true)
        G = Wd * inv(I - rho .* Wd)
        if pc
            Q = I - ones(N, N) ./ N
            G = Q * G * Q
        end
        g = SU.g_traces(W, rho; project_constant = pc, max_exact_n = 10_000)
        @test isapprox(g[1], tr(G); atol = 1e-10)
        @test isapprox(g[2], tr(G * G); atol = 1e-9)
        @test isapprox(g[3], sum(abs2, G); atol = 1e-9)
    end

    # Above the size gate the probe path runs instead. Seeded, it has to
    # repeat exactly, and it has to land near the exact value.
    g_ex = SU.g_traces(W, 0.5; max_exact_n = 10_000)
    g_p1 = SU.g_traces(W, 0.5; max_exact_n = 0, n_probes = 400,
                       rng = MersenneTwister(7))
    g_p2 = SU.g_traces(W, 0.5; max_exact_n = 0, n_probes = 400,
                       rng = MersenneTwister(7))
    @test g_p1 == g_p2
    @test isapprox(g_p1[3], g_ex[3]; rtol = 0.15)

    # The SAR variance against the expected information matrix, written out
    # densely with no probes anywhere. This is the first assertion in the
    # suite to check a standard error against a value rather than a size.
    m = fit_spatial(SAR(), W, X, y; method = ExactLU())
    b = coef(m)[1:3]; rho_h = m.rho; s2 = m.sigma2; P = 3
    G = Wd * inv(I - rho_h .* Wd); Gxb = G * (Xa * b)
    Ie = zeros(P + 2, P + 2)
    Ie[1:P, 1:P] = (transpose(Xa) * Xa) ./ s2
    Ie[1:P, P + 1] = (transpose(Xa) * Gxb) ./ s2
    Ie[P + 1, 1:P] = Ie[1:P, P + 1]
    Ie[P + 1, P + 1] = dot(Gxb, Gxb) / s2 + tr(transpose(G) * G) + tr(G * G)
    Ie[P + 1, P + 2] = tr(G) / s2
    Ie[P + 2, P + 1] = Ie[P + 1, P + 2]
    Ie[P + 2, P + 2] = N / (2 * s2^2)
    se_ref = sqrt.(diag(inv(Symmetric(Ie)))[1:(P + 1)])
    @test isapprox(stderror(m), se_ref; rtol = 1e-8)
    @test isapprox(m.rho_se, se_ref[P + 1]; rtol = 1e-8)

    # Same data, same call: bit-identical variance, traces and impacts.
    fits = [fit_spatial(SAR(), W, X, y; method = ExactLU()) for _ in 1:4]
    for f in fits
        @test vcov(f) == vcov(fits[1])
        @test f.rho_se == fits[1].rho_se
        @test f.trace_powers == fits[1].trace_powers
    end
    sdm = [fit_spatial(SDM(), W, X, y; method = ExactLU()) for _ in 1:3]
    effs = [spatial_effects(f) for f in sdm]
    for e in effs
        @test e.direct == effs[1].direct
        @test e.indirect == effs[1].indirect
        @test e.indirect_se == effs[1].indirect_se
    end

    # The panel estimators shared the defect, including the time-FE branch
    # that projects out the constant eigenvector.
    Np = 36; Tp = 6
    Wp = make_rook_weights(6)
    rng2 = MersenneTwister(1_300_016)
    Xp = randn(rng2, Np * Tp, 2)
    yp = Xp * [1.0, -0.5] .+ randn(rng2, Np * Tp)
    for fe in (IndivFE(), TwoWayFE())
        ps = [fit_spatial_panel(SAR(), fe, Wp, Xp, yp, Np, Tp) for _ in 1:3]
        for f in ps
            @test vcov(f) == vcov(ps[1])
            @test f.rho_se == ps[1].rho_se
        end
        @test all(isfinite, stderror(ps[1]))
    end
end
