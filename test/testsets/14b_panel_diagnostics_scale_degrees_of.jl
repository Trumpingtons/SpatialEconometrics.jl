@testset "14b. Panel Diagnostics: Scale, Degrees of Freedom, Degeneracy" begin
    _progress("14b. Panel Diagnostics: Scale, Degrees of Freedom, Degeneracy")
    rng = MersenneTwister(1414)
    L = 6; N = L * L; T_per = 5; NT = N * T_per
    W = make_rook_weights(L)
    X = hcat(ones(NT), randn(rng, NT, 2))
    β = [1.0, 2.0, -1.0]
    y = X * β .+ randn(rng, NT)

    A = kron(sparse(1.0I, T_per, T_per), W)
    # τ = tr(A²) + tr(A′A), the variance of the spatial error score
    τ = T_per * (sum(abs2, W) + sum(W .* transpose(W)))
    β_ols = X \ y; e = y .- X * β_ols; σ² = dot(e, e) / NT
    M = I - X * ((X' * X) \ X')
    d_ρ = dot(e, A * y) / σ²
    d_λ = dot(e, A * e) / σ²
    Aŷ = A * (X * β_ols)
    D = dot(Aŷ, M * Aŷ) / σ² + τ

    plm = panel_lm_tests(y, X, W, N, T_per)

    # Each component must be the squared score over its own variance.
    @test isapprox(plm.error_stat, d_λ^2 / τ; rtol = 1e-10)
    @test isapprox(plm.lag_stat, d_ρ^2 / D; rtol = 1e-10)

    # The discarded denominator used the lagged *observed* y and dropped
    # the trace term, which makes it smaller and the statistic larger.
    J_old = dot(A * y, M * (A * y)) / σ²
    @test J_old < D
    @test d_ρ^2 / J_old > plm.lag_stat

    # The joint test is the quadratic form in the inverse information, not
    # the sum of the marginals: d_ρ and d_λ are correlated, Cov = τ.
    LM_sp = (τ * d_ρ^2 - 2 * τ * d_ρ * d_λ + D * d_λ^2) / (τ * (D - τ))
    @test isapprox(plm.joint_stat, LM_sp + plm.re_stat; rtol = 1e-10)
    @test plm.joint_df == 3
    @test !isapprox(plm.joint_stat,
                    plm.lag_stat + plm.error_stat + plm.re_stat; rtol = 1e-6)

    # ---- scale under the null ----------------------------------------
    # Each component is χ²(1) and the joint χ²(3), so the means are 1 and
    # 3. The Monte-Carlo standard error of a χ²(1) mean at R = 300 is
    # 0.082, and of a χ²(3) mean 0.141.
    function lm_null_means(seed, R)
        rg = MersenneTwister(seed)
        acc = zeros(5)
        for _ in 1:R
            Xr = hcat(ones(NT), randn(rg, NT, 2))
            yr = Xr * β .+ randn(rg, NT)
            t = panel_lm_tests(yr, Xr, W, N, T_per)
            br = Xr \ yr; er = yr .- Xr * br; s2 = dot(er, er) / NT
            Mr = I - Xr * ((Xr' * Xr) \ Xr')
            Ay = A * yr
            old = (dot(er, Ay) / s2)^2 / (dot(Ay, Mr * Ay) / s2)
            acc .+= (t.lag_stat, t.error_stat, t.re_stat, t.joint_stat, old)
        end
        acc ./ R
    end
    m_lag, m_err, m_re, m_joint, m_old = lm_null_means(9090, 300)
    @test 0.78 < m_lag < 1.25
    @test 0.78 < m_err < 1.25
    @test 0.78 < m_re < 1.25
    @test 2.4 < m_joint < 3.6
    # the discarded lag statistic sits clearly above its corrected form
    @test m_old > m_lag + 0.08

    # ---- Chow degrees of freedom -------------------------------------
    # The sub-period fits estimate the K slopes *and* a spatial lag
    # coefficient, so the break imposes K + 1 restrictions. Counting only
    # K roughly doubles the size of the test.
    K = size(X, 2)
    ct = spatial_chow_test(y, X, W, N, T_per, 3)
    @test ct.df == K + 1

    function chow_rej(seed, R; ρ = 0.4, brk = 3, break_size = 0.0)
        rg = MersenneTwister(seed)
        S_lu = lu(sparse(1.0I, N, N) - ρ * W)
        n_rej = 0
        for _ in 1:R
            Xr = hcat(ones(NT), randn(rg, NT, 2))
            yr = zeros(NT)
            for t in 1:T_per
                idx = ((t-1)*N+1):(t*N)
                bt = t <= brk ? β : β .+ [0.0, break_size, 0.0]
                yr[idx] = S_lu \ (Xr[idx, :] * bt .+ randn(rg, N))
            end
            spatial_chow_test(yr, Xr, W, N, T_per, brk).p_value < 0.05 &&
                (n_rej += 1)
        end
        n_rej / R
    end
    # correctly sized with no break, and powerful against one
    @test chow_rej(515, 300) < 0.10
    @test chow_rej(616, 150; break_size = 0.6) > 0.60

    # ---- Hausman: no fabricated statistic ----------------------------
    nm = ["x1", "x2", "ρ"]
    b_fe = [1.00, 2.00, 0.40]
    b_re = [1.05, 1.96, 0.42]
    d = b_fe .- b_re
    V_re = [0.0100 0.0010 0.0005
            0.0010 0.0120 0.0008
            0.0005 0.0008 0.0090]

    # A genuine efficiency ordering: the difference is positive definite
    # and the statistic is the textbook form on all three directions.
    V_fe = V_re + 0.004 * Matrix(1.0I, 3, 3)
    h = spatial_hausman_test(HausMock(nm, b_fe, V_fe),
                             HausMock(nm, b_re, V_re))
    @test h.df == 3
    @test isapprox(h.stat, dot(d, (V_fe - V_re) \ d); rtol = 1e-8)
    @test 0.0 <= h.p_value <= 1.0

    # Reversed arguments: nothing positive to invert, so the test says so
    # rather than returning a number.
    h_rev = spatial_hausman_test(HausMock(nm, b_re, V_re),
                                 HausMock(nm, b_fe, V_fe))
    @test h_rev.df == 0
    @test isnan(h_rev.stat)
    @test isnan(h_rev.p_value)

    # One clearly positive direction, the other two a near-cancellation of
    # indeterminate sign -- the common case for two panel fits that differ
    # only in their fixed effects. The old code shifted every eigenvalue up
    # to force definiteness, which left the smallest at the shift itself,
    # so the inverse carried an eigenvalue of 1e6 and the statistic was
    # whatever that noise direction happened to be.
    Q = Matrix(qr(randn(MersenneTwister(77), 3, 3)).Q)
    V_part = Q * Diagonal([0.004, -1.0e-5, -2.0e-5]) * Q'
    h_part = spatial_hausman_test(HausMock(nm, b_fe, V_re + V_part),
                                  HausMock(nm, b_re, V_re))
    @test h_part.df == 1
    ev = eigen(Symmetric(V_part))
    j = argmax(ev.values)
    z = dot(ev.vectors[:, j], d)
    @test isapprox(h_part.stat, z^2 / ev.values[j]; rtol = 1e-6)
    shift = -minimum(ev.values) + 1.0e-6
    stat_ridge = dot(d, Symmetric(V_part + shift * I) \ d)
    @test stat_ridge > 50 * h_part.stat

    # ---- panel_moran: the pooled row is internally consistent ----
    # pooled.I_stat is an average over T_per periods, so the variance
    # beside it must be the variance of that average. It carried the
    # single-period variance, too large by a factor of T_per.
    u = randn(MersenneTwister(2323), NT)
    pm = panel_moran(u, W, N, T_per)
    one_period = moran_global(u[1:N], W)
    @test length(pm.period_results) == T_per
    @test pm.T_periods == T_per
    @test isapprox(pm.pooled.I_stat,
                   mean(r.I_stat for r in pm.period_results); rtol = 1e-12)
    @test isapprox(pm.pooled.E_I, one_period.E_I; rtol = 1e-12)
    @test isapprox(pm.pooled.Var_I, one_period.Var_I / T_per; rtol = 1e-12)
    # the combined z is Stouffer's, not I_stat standardised by Var_I
    @test isapprox(pm.pooled.z_stat,
                   mean(r.z_stat for r in pm.period_results) * sqrt(T_per);
                   rtol = 1e-12)

    # The variance of the average must track the spread of the average,
    # not the spread of a single period. Over 400 independent draws the
    # two differ by a factor of about T_per.
    function moran_spread(seed, R)
        rg = MersenneTwister(seed)
        avg = zeros(R); first_p = zeros(R)
        for r in 1:R
            q = panel_moran(randn(rg, NT), W, N, T_per)
            avg[r] = q.pooled.I_stat
            first_p[r] = q.period_results[1].I_stat
        end
        (var(avg), var(first_p))
    end
    v_avg, v_one = moran_spread(2424, 400)
    @test 2.5 < v_one / v_avg < 8.0
    @test 0.5 < pm.pooled.Var_I / v_avg < 2.0
    @test pm.pooled.Var_I < one_period.Var_I
end
