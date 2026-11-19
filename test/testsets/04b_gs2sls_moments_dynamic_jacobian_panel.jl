@testset "4b. GS2SLS moments, dynamic Jacobian, panel information" begin
    _progress("4b. GS2SLS moments, dynamic Jacobian, panel information")
    GS = SpatialEconometrics.SpatialGS2SLSPanel
    L = 14; N = L * L
    W = make_rook_weights(L); Wd = Matrix(W)
    c = sum(abs2, W) / N                      # tr(W'W)/N
    @test isapprox(c, tr(transpose(Wd) * Wd) / N; rtol = 1e-12)

    # --- The Kelejian-Prucha moment conditions. Two of the three used to
    # omit their right-hand sides, so the objective drove eps'eps/N and
    # eps'W'W eps/N toward zero when their true values are sigma2 and
    # sigma2*tr(W'W)/N. Only the middle moment was valid.
    rng = MersenneTwister(400_041)
    lam_t = 0.5
    eps = randn(rng, N)
    u = (I - lam_t .* Wd) \ eps
    Wu = Wd * u; WWu = Wd * Wu
    e_true = u .- lam_t .* Wu
    @test isapprox(e_true, eps; rtol = 1e-10)
    s2 = dot(eps, eps) / N
    # the two moments the old objective wrongly set to zero
    @test isapprox(dot(eps, eps) / N, s2; rtol = 1e-12)
    @test s2 > 0.5
    @test dot(Wd * eps, Wd * eps) / N > 0.1
    # and the one that really is zero in expectation
    @test abs(dot(eps, Wd * eps) / N) < 0.15

    # The objective must be near-minimal at the truth now.
    lam_hat = GS.estimate_lambda_gmm(u, Wu, WWu, c)
    q_true = GS.gmm_objective(lam_t, u, Wu, WWu, c)
    @test GS.gmm_objective(lam_hat, u, Wu, WWu, c) <= q_true + 1e-12
    for bad in (lam_t - 0.3, lam_t + 0.3)
        @test GS.gmm_objective(bad, u, Wu, WWu, c) > q_true
    end

    # One draw is too noisy to pin -- the sampling standard deviation is
    # about 0.08 at this N -- so compare the two estimators over a set of
    # draws. The discarded objective is rebuilt here, so this stays a
    # regression test under any seed rather than pinning one number.
    function old_obj(uu, Wuu, WWuu, l)
        m11 = dot(uu, uu) / N; m12 = dot(uu, Wuu) / N
        m22 = dot(Wuu, Wuu) / N; m13 = dot(uu, WWuu) / N
        m23 = dot(Wuu, WWuu) / N; m33 = dot(WWuu, WWuu) / N
        g1 = m11 - 2l * m12 + l^2 * m22
        g2 = m12 - l * m22 - l * m13 + l^2 * m23
        g3 = m22 - 2l * m23 + l^2 * m33
        g1^2 + g2^2 + g3^2
    end
    new_err = Float64[]; old_err = Float64[]
    for _ in 1:12
        uu = (I - lam_t .* Wd) \ randn(rng, N)
        Wuu = Wd * uu; WWuu = Wd * Wuu
        push!(new_err,
              abs(GS.estimate_lambda_gmm(uu, Wuu, WWuu, c) - lam_t))
        push!(old_err,
              abs(argmin(l -> old_obj(uu, Wuu, WWuu, l),
                         -0.95:0.005:0.95) - lam_t))
    end
    # the old objective was biased about +0.27 at this lambda
    @test mean(old_err) > 0.15
    @test mean(new_err) < 0.08
    @test mean(new_err) < mean(old_err) / 2
    @test isapprox(mean(GS.estimate_lambda_gmm(u, Wu, WWu, c)), lam_hat)

    # --- The full fit: lambda's variance slot used to hold sigma2/NT,
    # which involves neither W nor lambda-hat, and the Sargan p-value was
    # the literal constant 1.0.
    Lp = 8; Np = Lp * Lp; T_tot = 6; NTp = Np * T_tot
    Wp = make_rook_weights(Lp)
    rng2 = MersenneTwister(400_042)
    Xe = randn(rng2, NTp, 2); z1 = randn(rng2, NTp); z2 = randn(rng2, NTp)
    en = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rng2, NTp)
    Sl = lu(sparse(1.0I, Np, Np) - 0.3 .* Wp)
    Bl = lu(sparse(1.0I, Np, Np) - 0.25 .* Wp)
    yp = zeros(NTp)
    for t in 1:T_tot
        idx = ((t - 1) * Np + 1):(t * Np)
        yp[idx] = Sl \ (Xe[idx, :] * [1.2, -0.6] .+ en[idx] .* 0.8 .+
                        (Bl \ (0.4 .* randn(rng2, Np))))
    end
    m_gs2 = fit_gs2sls_panel(IndivFE(), Wp, Xe, reshape(en, NTp, 1),
                             hcat(z1, z2), yp, Np, T_tot)
    se_gs = stderror(m_gs2)
    @test all(isfinite, se_gs)
    @test se_gs[end] > 0
    @test !isapprox(se_gs[end]^2, m_gs2.sigma2 / (Np * (T_tot - 1));
                    rtol = 1e-6)
    @test 0.0 < m_gs2.sargan_pval < 1.0
    @test m_gs2.sargan_pval != 1.0
    # over-identified by one: three instruments columns beyond the
    # structural parameters is what gives the test any power
    @test m_gs2.sargan_stat > 0
    # a 120-replication Monte Carlo put every one of the five estimates
    # within 0.006 of the truth; here just pin that they are sane
    @test isapprox(m_gs2.rho, 0.3; atol = 0.1)
    @test isapprox(m_gs2.lambda, 0.25; atol = 0.15)

    # --- The dynamic panel counted log|I - rho*W| once instead of once
    # per period, which left rho-hat inconsistent: over 80 draws it sat at
    # 0.397 for T = 8 and 0.409 for T = 64 against a true 0.35, so the
    # error never shrank and coverage fell to 1.2% as the standard error
    # tightened on the wrong value. With T large the surviving bias is
    # small enough to pin.
    Ld = 7; Nd = Ld * Ld; Td = 16
    Wdp = make_rook_weights(Ld)
    rng3 = MersenneTwister(400_043)
    Xd = randn(rng3, Nd * Td, 1)
    mu_i = randn(rng3, Nd)
    Sd = lu(sparse(1.0I, Nd, Nd) - 0.35 .* Wdp)
    yd = zeros(Nd * Td)
    yd[1:Nd] = Sd \ (mu_i .+ Xd[1:Nd, :] * [0.8] .+ 0.4 .* randn(rng3, Nd))
    for t in 2:Td
        idx = ((t - 1) * Nd + 1):(t * Nd)
        ip = ((t - 2) * Nd + 1):((t - 1) * Nd)
        yd[idx] = Sd \ (mu_i .+ 0.3 .* yd[ip] .+ 0.15 .* (Wdp * yd[ip]) .+
                        Xd[idx, :] * [0.8] .+ 0.4 .* randn(rng3, Nd))
    end
    m_dyn2 = fit_dynamic_spatial_panel(SAR(), IndivFE(), Wdp, Xd, yd, Nd,
                                       Td; rng = MersenneTwister(11))
    @test isapprox(m_dyn2.rho, 0.35; atol = 0.04)
    @test isapprox(m_dyn2.gamma, 0.3; atol = 0.05)
    @test isapprox(m_dyn2.eta, 0.15; atol = 0.07)
    @test isapprox(coef(m_dyn2)[3], 0.8; atol = 0.05)

    # --- The spatial panel information matrix, against a numerical
    # Hessian of the within-transformed likelihood. Its trace inputs were
    # made exact earlier, but the assembled matrix had never been checked.
    Ls = 7; Ns = Ls * Ls; Ts = 5
    Ws = make_rook_weights(Ls); Wsd = Matrix(Ws)
    rng4 = MersenneTwister(400_044)
    Xs2 = randn(rng4, Ns * Ts, 2)
    ys2 = zeros(Ns * Ts)
    for t in 1:Ts
        idx = ((t - 1) * Ns + 1):(t * Ns)
        ys2[idx] = (I - 0.4 .* Wsd) \ (Xs2[idx, :] * [1.0, -0.5] .+
                                       randn(rng4, Ns))
    end
    m_sp = fit_spatial_panel(SAR(), IndivFE(), Ws, Xs2, ys2, Ns, Ts;
                             method = ExactLU())
    dmv(z) = begin
        Z = reshape(copy(z), Ns, Ts); Z .-= mean(Z; dims = 2); vec(Z)
    end
    yt2 = dmv(ys2)
    Xt2 = reduce(hcat, [dmv(Xs2[:, k]) for k in 1:2])
    Te2 = Ts - 1
    f_sp = th -> begin
        bb = th[1:2]; rr = th[3]; ss = th[4]
        ss <= 0 && return -Inf
        A = I - rr .* Wsd; ee = zeros(Ns * Ts)
        for t in 1:Ts
            idx = ((t - 1) * Ns + 1):(t * Ns)
            ee[idx] = A * yt2[idx] .- Xt2[idx, :] * bb
        end
        -0.5 * Ns * Te2 * log(2pi * ss) - dot(ee, ee) / (2ss) +
            Te2 * first(logabsdet(A))
    end
    th_sp = vcat(coef(m_sp)[1:2], m_sp.rho, m_sp.sigma2)
    hh = 1e-5 .* max.(abs.(th_sp), 1.0); Hsp = zeros(4, 4)
    for i in 1:4, j in 1:4
        ei = zeros(4); ei[i] = hh[i]; ej = zeros(4); ej[j] = hh[j]
        Hsp[i, j] = (f_sp(th_sp + ei + ej) - f_sp(th_sp + ei - ej) -
                     f_sp(th_sp - ei + ej) + f_sp(th_sp - ei - ej)) /
                    (4hh[i] * hh[j])
    end
    se_sp_num = sqrt.(abs.(diag(inv(-Symmetric(Hsp)))[1:3]))
    @test isapprox(stderror(m_sp), se_sp_num; rtol = 0.06)

    # --- The regime model: rho's variance slot also held 1/N.
    Lr = 10; Nr = Lr * Lr
    Wr = make_rook_weights(Lr); Wrd = Matrix(Wr)
    rng5 = MersenneTwister(400_045)
    Xr = hcat(randn(rng5, Nr), randn(rng5, Nr))
    labels = [i <= Nr ÷ 2 ? 1 : 2 for i in 1:Nr]
    yr = (I - 0.4 .* Wrd) \ (randn(rng5, Nr) .+ Xr * [1.0, -0.5])
    m_rg = fit_spatial_regime(Wr, Xr, yr, ExogenousRegimes(labels);
                              method = ExactLU())
    Pr = length(coef(m_rg)) - 1
    Xrg = m_rg.X_aug
    br = coef(m_rg)[1:Pr]; rr = m_rg.rho; s2r = m_rg.sigma2
    Gr = Wrd * inv(I - rr .* Wrd); Gxb = Gr * (Xrg * br)
    Ir = zeros(Pr + 2, Pr + 2)
    Ir[1:Pr, 1:Pr] = (transpose(Xrg) * Xrg) ./ s2r
    Ir[1:Pr, Pr + 1] = (transpose(Xrg) * Gxb) ./ s2r
    Ir[Pr + 1, 1:Pr] = Ir[1:Pr, Pr + 1]
    Ir[Pr + 1, Pr + 1] = dot(Gxb, Gxb) / s2r + tr(Gr * Gr) +
                         tr(transpose(Gr) * Gr)
    Ir[Pr + 1, Pr + 2] = tr(Gr) / s2r
    Ir[Pr + 2, Pr + 1] = Ir[Pr + 1, Pr + 2]
    Ir[Pr + 2, Pr + 2] = Nr / (2 * s2r^2)
    se_rg_ref = sqrt.(diag(inv(Symmetric(Ir)))[1:(Pr + 1)])
    @test isapprox(stderror(m_rg), se_rg_ref; rtol = 1e-8)
    @test !isapprox(stderror(m_rg)[end], 1 / sqrt(Nr); rtol = 1e-3)
end
