@testset "5b. Dynamic GMM: Windmeijer Assembly and Diagnostics" begin
    _progress("5b. Dynamic GMM: Windmeijer Assembly and Diagnostics")
    DG = SpatialEconometrics.DynamicPanelSystemGMM

    # y_it = ρ y_{i,t-1} + x_it β + μ_i + ε_it, burnt in so the initial
    # condition is from the stationary distribution.
    function gen_dp(rng; N = 100, Tt = 6, ρ = 0.5, b = 1.0, burn = 25)
        Y = zeros(N, Tt); Xt = zeros(N, Tt, 1)
        μ = randn(rng, N)
        for i in 1:N
            x = randn(rng, Tt + burn)
            yv = μ[i] / (1 - ρ)
            for sidx in 1:burn
                yv = ρ * yv + b * x[sidx] + μ[i] + randn(rng)
            end
            for t in 1:Tt
                yv = ρ * yv + b * x[burn + t] + μ[i] + randn(rng)
                Y[i, t] = yv; Xt[i, t, 1] = x[burn + t]
            end
        end
        (Y, Xt)
    end

    # Rebuild the two-step pieces so the assembly can be checked.
    function gmm_pieces(variant, Y, Xt; max_lag = 4, collapsed = false)
        Nn = size(Xt, 1)
        Xl, yl, Zl, Hm, T_diff, Lq =
            DG.build_gmm_system(variant, Y, Xt, max_lag, collapsed)
        Pq = 1 + size(Xt, 3)
        S_ZX = zeros(Lq, Pq); S_Zy = zeros(Lq); O1 = zeros(Lq, Lq)
        for i in 1:Nn
            S_ZX .+= Zl[i]' * Xl[i]
            S_Zy .+= Zl[i]' * yl[i]
            O1 .+= Zl[i]' * Hm * Zl[i]
        end
        W1 = pinv(Symmetric(O1))
        G1i = inv(Symmetric(S_ZX' * W1 * S_ZX))
        t1 = G1i * (S_ZX' * (W1 * S_Zy))
        O2 = zeros(Lq, Lq)
        for i in 1:Nn
            z = Zl[i]' * (yl[i] .- Xl[i] * t1); O2 .+= z * z'
        end
        W2 = pinv(Symmetric(O2))
        V2 = inv(Symmetric(S_ZX' * W2 * S_ZX))
        t2 = V2 * (S_ZX' * (W2 * S_Zy))
        V1 = G1i * (S_ZX' * W1 * O2 * W1 * S_ZX) * G1i
        g2 = S_Zy .- S_ZX * t2
        uv = W2 * g2; Apre = -V2 * (S_ZX' * W2)
        psi = th -> begin
            wu = zeros(eltype(th), Lq)
            for i in 1:Nn
                z = Zl[i]' * (yl[i] .- Xl[i] * th)
                wu .+= z .* dot(z, uv)
            end
            Apre * wu
        end
        Dm = ForwardDiff.jacobian(psi, t1)
        (; V2, V1, D = Dm, t1, t2, L = Lq, P = Pq, g2, W2, T_diff, H = Hm)
    end

    rngd = MersenneTwister(2024)
    Yd, Xd = gen_dp(rngd)

    for variant in (DifferenceGMM(), SystemGMM())
        q = gmm_pieces(variant, Yd, Xd)
        m = fit_dynamic_gmm(variant, Yd, Xd)

        # point estimates and the uncorrected variance
        @test maximum(abs, coef(m) .- q.t2) < 1e-9
        @test maximum(abs, m.coef_onestep .- q.t1) < 1e-9
        @test maximum(abs, m.vcov_naive .- q.V2) < 1e-10
        @test m.n_instruments == q.L
        @test m.hansen_df == q.L - q.P
        @test isapprox(m.hansen_j, dot(q.g2, q.W2, q.g2); rtol = 1e-9)

        # Windmeijer (2005) eq. 4.4 / Roodman (2009) eq. 23:
        #   V_c = V_2 + D V_2 + (D V_2)' + D V_1 D'
        DV = q.D * q.V2
        V_correct = q.V2 .+ DV .+ transpose(DV) .+ (q.D * q.V1 * q.D')
        @test maximum(abs, vcov(m) .- V_correct) < 1e-10
        @test issymmetric(round.(vcov(m); digits = 12))

        # The discarded assembly put a V_1 inside the middle terms,
        # shrinking them by about two orders of magnitude and leaving the
        # corrected variance barely above the uncorrected one.
        V_old = q.V2 .+ (q.D * q.V1 * q.V2) .+ (q.V2 * q.V1 * q.D') .+
                (q.D * q.V1 * q.D')
        @test norm(q.D * q.V2) > 50 * norm(q.D * q.V1 * q.V2)
        for j in 1:q.P
            @test sqrt(V_old[j, j]) < sqrt(V_correct[j, j])
            # and the correction must actually inflate the naive variance
            @test sqrt(V_correct[j, j]) > sqrt(q.V2[j, j])
        end
    end

    # ---- the instrument matrix -------------------------------------
    let Tt = 6, N2 = 8, ml = 4
        rg2 = MersenneTwister(5)
        Y2, X2 = gen_dp(rg2; N = N2, Tt = Tt)
        T_diff = Tt - 2

        Xl, yl, Zl, Hm, td, Lq =
            DG.build_gmm_system(DifferenceGMM(), Y2, X2, ml, false)
        @test td == T_diff
        # one column per (period, lag) pair, plus the exogenous regressor
        @test Lq == sum(min(t - 2, ml) for t in 3:Tt) + 1
        @test size(Zl[1]) == (T_diff, Lq)

        # the differenced equation and its regressors
        i = 3
        for eq in 1:T_diff
            t = eq + 2
            @test isapprox(yl[i][eq], Y2[i, t] - Y2[i, t-1]; rtol = 1e-12)
            @test isapprox(Xl[i][eq, 1], Y2[i, t-1] - Y2[i, t-2]; rtol = 1e-12)
            @test isapprox(Xl[i][eq, 2], X2[i, t, 1] - X2[i, t-1, 1];
                           rtol = 1e-12)
        end
        # equation t is instrumented by the levels y_{t-2} … y_1
        @test isapprox(Zl[i][1, 1], Y2[i, 1]; rtol = 1e-12)
        @test isapprox(Zl[i][2, 2], Y2[i, 2]; rtol = 1e-12)
        @test isapprox(Zl[i][2, 3], Y2[i, 1]; rtol = 1e-12)

        # H is the covariance of Δε up to scale: 2 on the diagonal, -1 off
        @test size(Hm) == (T_diff, T_diff)
        @test all(diag(Hm) .== 2)
        @test all(diag(Hm, 1) .== -1) && all(diag(Hm, -1) .== -1)
        @test Hm == Hm'

        # collapsing holds the count down: one column per lag, not per pair
        _, _, Zc, _, _, Lc =
            DG.build_gmm_system(DifferenceGMM(), Y2, X2, ml, true)
        @test Lc == min(ml, Tt - 2) + 1
        @test Lc < Lq
        # column s of the collapsed block holds lag s in every equation
        for eq in 1:T_diff
            t = eq + 2
            @test isapprox(Zc[i][eq, 1], Y2[i, t-2]; rtol = 1e-12)
        end

        # System GMM stacks level equations under the differenced ones,
        # instrumented by Δy_{t-1}, and gives them an identity H block.
        Xs, ys, Zs, Hs, _, Ls =
            DG.build_gmm_system(SystemGMM(), Y2, X2, ml, false)
        @test size(Zs[1], 1) == 2 * T_diff
        @test Ls == Lq + T_diff
        for eq in 1:T_diff
            row = T_diff + eq; t = eq + 2
            @test isapprox(ys[i][row], Y2[i, t]; rtol = 1e-12)
            @test isapprox(Xs[i][row, 1], Y2[i, t-1]; rtol = 1e-12)
            @test isapprox(Zs[i][row, Lq - 1 + eq],
                           Y2[i, t-1] - Y2[i, t-2]; rtol = 1e-12)
        end
        @test Hs[(T_diff+1):end, (T_diff+1):end] == I(T_diff)
        @test all(Hs[1:T_diff, (T_diff+1):end] .== 0)
    end

    # ---- the diagnostics are calibrated ----------------------------
    # Δε is MA(1) by construction, so AR(1) in the differenced residuals
    # must reject hard while AR(2) must behave like a standard normal.
    # The corrected variance must also cover better than the naive one,
    # and the gain is largest at small N, which is the point of it.
    function dp_diagnostics(variant, N, R, seed)
        rg = MersenneTwister(seed)
        a1 = Float64[]; a2 = Float64[]
        hit_c = 0; hit_n = 0; jrej = 0; njj = 0
        for _ in 1:R
            Yr, Xr = gen_dp(rg; N = N)
            mm = fit_dynamic_gmm(variant, Yr, Xr)
            isfinite(mm.ar1_z) && push!(a1, mm.ar1_z)
            isfinite(mm.ar2_z) && push!(a2, mm.ar2_z)
            abs(coef(mm)[1] - 0.5) <= 1.96 * stderror(mm)[1] && (hit_c += 1)
            abs(coef(mm)[1] - 0.5) <= 1.96 * sqrt(mm.vcov_naive[1, 1]) &&
                (hit_n += 1)
            if mm.hansen_df > 0
                njj += 1
                ccdf(Chisq(mm.hansen_df), mm.hansen_j) < 0.05 && (jrej += 1)
            end
        end
        (mean(a1), mean(a2), std(a2), hit_c / R, hit_n / R, jrej / max(njj, 1))
    end

    m1, m2, s2, cov_c, cov_n, jr = dp_diagnostics(SystemGMM(), 50, 80, 4242)
    @test m1 < -2.5                 # AR(1) must reject: Δε is MA(1)
    @test abs(m2) < 0.45            # AR(2) centred
    @test 0.75 < s2 < 1.35          # AR(2) scaled like a standard normal
    @test jr < 0.15                 # Hansen J not wildly oversized
    # the Windmeijer correction earns its keep at N = 50
    @test cov_c > cov_n + 0.05
    @test cov_c > 0.85
end
