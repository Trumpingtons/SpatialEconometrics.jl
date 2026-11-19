@testset "9. Spatial Diagnostics (Moran, LM, Geary)" begin
    _progress("9. Spatial Diagnostics (Moran, LM, Geary)")
    rng = MersenneTwister(909); W = make_rook_weights(15); N = size(W, 1)
    X = randn(rng, N, 2)
    y_sar = (I - 0.5 * W) \ (X * [1.5, -0.8] .+ 0.4 .* randn(rng, N))

    mr = moran_global(y_sar, W)
    @test mr.I_stat > 0.0
    @test mr.p_value < 0.05

    gc = geary_c(y_sar, W)
    @test gc.C_stat < 1.0  # positive autocorrelation → C < 1

    Xc = hcat(ones(N), X)
    lms = lm_tests(y_sar, Xc, W)
    @test lms.lag.stat > 0.0
    @test lms.lag.p_value < 0.05

    # LM-lag and LM-error are different statistics. They were once written
    # with the same body, so both reported the LM-error value under two
    # labels; a sign-and-threshold check passes on that. Pin both to their
    # closed forms instead.
    let qrX = qr(Xc), Wd = Matrix(W)
        b = qrX \ y_sar
        e = y_sar .- Xc * b
        s2 = dot(e, e) / N
        Tr = tr(Wd' * Wd + Wd * Wd)
        WXb = Wd * (Xc * b)
        MWXb = WXb .- Xc * (qrX \ WXb)
        Jrr = dot(MWXb, MWXb) / s2 + Tr
        d_rho = dot(e, Wd * y_sar) / s2
        d_lam = dot(e, Wd * e) / s2

        @test isapprox(lms.lag.stat, d_rho^2 / Jrr; rtol = 1e-8)
        @test isapprox(lms.error.stat, d_lam^2 / Tr; rtol = 1e-8)
        @test !isapprox(lms.lag.stat, lms.error.stat; rtol = 1e-6)

        # Anselin, Bera, Florax & Yoon (1996). The lag correction
        # SUBTRACTS the error score; an earlier version added it, which
        # made the robust lag test fire on pure SEM data.
        @test isapprox(lms.robust_lag.stat,
                       (d_rho - d_lam)^2 / (Jrr - Tr); rtol = 1e-8)
        @test isapprox(lms.robust_error.stat,
                       (d_lam - Tr * d_rho / Jrr)^2 / (Tr - Tr^2 / Jrr);
                       rtol = 1e-8)
    end

    # The battery has to separate the two processes, not merely reject.
    # On data generated with a pure error process the robust ERROR test
    # must be the one that fires, and vice versa.
    let L = 14, Wl = make_rook_weights(L), n = size(Wl, 1), Wf = Matrix(Wl)
        g = MersenneTwister(31337)
        Xd = hcat(randn(g, n), randn(g, n)); Xdc = hcat(ones(n), Xd)
        bd = [1.5, -0.8]

        y_sem = Xd * bd .+ (I - 0.5 * Wf) \ randn(g, n)
        t_sem = lm_tests(y_sem, Xdc, Wl)
        @test t_sem.robust_error.p_value < 0.05
        @test t_sem.robust_lag.p_value > 0.05

        y_lag = (I - 0.5 * Wf) \ (Xd * bd .+ randn(g, n))
        t_lag = lm_tests(y_lag, Xdc, Wl)
        @test t_lag.robust_lag.p_value < 0.05
        @test t_lag.robust_error.p_value > 0.05
    end


    lisa = moran_local(y_sar, W; n_perm=199, rng=MersenneTwister(9091))
    @test length(lisa.local_I) == N
    @test any(lisa.cluster_type .== :HH) || any(lisa.cluster_type .== :LL)
end
