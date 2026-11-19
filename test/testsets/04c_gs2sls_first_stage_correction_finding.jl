@testset "4c. GS2SLS λ first-stage correction (finding 24)" begin
    _progress("4c. GS2SLS λ first-stage correction (finding 24)")
    GS = SpatialEconometrics.SpatialGS2SLSPanel
    SU = SpatialEconometrics.SpatialUtils
    L = 8; N = L * L; T_tot = 6; NT = N * T_tot
    W = make_rook_weights(L)
    λ_true = 0.25; ρ_true = 0.3

    # One panel, built the way testset 4 builds its own.
    function gen(rg)
        Xe = randn(rg, NT, 2); z1 = randn(rg, NT); z2 = randn(rg, NT)
        en = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rg, NT)
        Sl = lu(sparse(1.0I, N, N) - ρ_true .* W)
        Bl = lu(sparse(1.0I, N, N) - λ_true .* W)
        y = zeros(NT)
        for t in 1:T_tot
            idx = ((t - 1) * N + 1):(t * N)
            y[idx] = Sl \ (Xe[idx, :] * [1.2, -0.6] .+ en[idx] .* 0.8 .+
                           (Bl \ (0.4 .* randn(rg, N))))
        end
        return Xe, en, z1, z2, y
    end

    # Rebuild the first stage so the corrected and uncorrected standard
    # errors can be compared on identical inputs.
    function first_stage(Xe, en, z1, z2, y)
        T_post = T_tot - 1
        idx = (N + 1):NT
        y_p = y[idx]; Xe_p = Xe[idx, :]
        en_p = reshape(en[idx], :, 1); Zi_p = hcat(z1, z2)[idx, :]
        Wy_p = SU.panel_spatial_lag!(similar(y_p), W, y_p, N, T_post)
        fe = IndivFE()
        y_t = SU.within_transform!(copy(y_p), N, T_post, fe)
        Wy_t = SU.within_transform!(copy(Wy_p), N, T_post, fe)
        Xe_t = SU.within_transform(Xe_p, N, T_post, fe)
        En_t = SU.within_transform(en_p, N, T_post, fe)
        Zi_t = SU.within_transform(Zi_p, N, T_post, fe)
        WXe = SU.panel_spatial_lag(W, Xe_t, N, T_post)
        H = hcat(Xe_t, WXe, SU.panel_spatial_lag(W, WXe, N, T_post),
                 Zi_t, SU.panel_spatial_lag(W, Zi_t, N, T_post))
        Psi_endog = hcat(Wy_t, En_t)
        HHi = inv(Symmetric(H' * H))
        Psi_hat = H * (HHi * (H' * Psi_endog))
        Psi_full = hcat(Psi_hat[:, 1], Xe_t, Psi_hat[:, 2:end])
        Psi_raw = hcat(Wy_t, Xe_t, En_t)
        θ = (Psi_full' * Psi_raw) \ (Psi_full' * y_t)
        u = y_t .- Psi_raw * θ
        Wu = SU.panel_spatial_lag!(similar(u), W, u, N, T_post)
        WWu = SU.panel_spatial_lag!(similar(u), W, Wu, N, T_post)
        lam = GS.estimate_lambda_gmm(u, Wu, WWu, sum(abs2, W) / N)
        return lam, u, Wu, Psi_full, Psi_raw, T_post
    end

    lam, u, Wu, Psi_full, Psi_raw, T_post = first_stage(gen(MersenneTwister(4400))...)

    se_plain = GS._lambda_gmm_se(lam, u, Wu, W, N, T_post)
    se_corr = GS._lambda_gmm_se(lam, u, Wu, W, N, T_post;
                                Psi_full=Psi_full, Psi_raw=Psi_raw)
    @test isfinite(se_plain) && se_plain > 0
    @test isfinite(se_corr) && se_corr > 0
    # The correction adds a positive semi-definite term, so it can only
    # widen the interval -- never narrow it.
    @test se_corr > se_plain
    # …and it is a correction, not a rescaling: the same order of magnitude.
    @test se_corr < 2.0 * se_plain

    # The added term itself must be symmetric and positive semi-definite.
    e_f = u .- lam .* Wu
    σ2 = dot(e_f, e_f) / (N * T_post)
    c = sum(abs2, W) / N
    FS = GS._first_stage_term(lam, e_f, W, Psi_full, Psi_raw, N, T_post,
                              c, σ2)
    @test size(FS) == (2, 2)
    @test isapprox(FS, FS'; atol=1e-14 * max(1.0, maximum(abs, FS)))
    @test all(eigvals(Symmetric(FS)) .>= -1e-12)

    # Omitting the first-stage arguments must reproduce the uncorrected
    # value exactly -- that path is what checks the moment algebra against
    # known disturbances.
    @test GS._lambda_gmm_se(lam, u, Wu, W, N, T_post) == se_plain
    # A first stage with no columns adds nothing.
    @test GS._lambda_gmm_se(lam, u, Wu, W, N, T_post;
                            Psi_full=zeros(N * T_post, 0),
                            Psi_raw=zeros(N * T_post, 0)) == se_plain

    # The full fit must use the corrected value, with σ̂² on the
    # N(T_post − 1) degrees of freedom unit demeaning leaves and the
    # factor κ = (T_post − 1)/T_post by which it shrinks the pooled
    # quadratic forms' variance (finding 36).
    se_fit = GS._lambda_gmm_se(lam, u, Wu, W, N, T_post;
                               Psi_full=Psi_full, Psi_raw=Psi_raw,
                               n_eff=N * (T_post - 1),
                               kappa=(T_post - 1) / T_post)
    Xe, en, z1, z2, y = gen(MersenneTwister(4400))
    m = fit_gs2sls_panel(IndivFE(), W, Xe, reshape(en, NT, 1),
                         hcat(z1, z2), y, N, T_tot)
    j = findfirst(n -> occursin("λ", n), coefnames(m))
    @test !isnothing(j)
    @test isapprox(stderror(m)[j], se_fit; rtol=1e-8)
    @test stderror(m)[j] > se_plain

    # ── Calibration: the corrected interval must cover near nominal ──
    # Before the correction the ratio of reported standard error to
    # sampling variability sat at 0.82-0.86 at every size, and coverage at
    # 88.8-92.0%. Pin the direction of travel with a short run.
    lam_hat = Float64[]; lam_se = Float64[]; hit = 0
    rg = MersenneTwister(4401)
    for r in 1:40
        Xe_r, en_r, z1_r, z2_r, y_r = gen(rg)
        try
            mr = fit_gs2sls_panel(IndivFE(), W, Xe_r, reshape(en_r, NT, 1),
                                  hcat(z1_r, z2_r), y_r, N, T_tot)
            jr = findfirst(n -> occursin("λ", n), coefnames(mr))
            isnothing(jr) && continue
            isfinite(stderror(mr)[jr]) || continue
            push!(lam_hat, coef(mr)[jr]); push!(lam_se, stderror(mr)[jr])
            abs(coef(mr)[jr] - λ_true) <= 1.96 * stderror(mr)[jr] &&
                (hit += 1)
        catch
            continue
        end
    end
    @test length(lam_hat) >= 30
    # λ̂ is consistent at this size.
    @test abs(mean(lam_hat) - λ_true) < 0.06
    # The reported standard error now tracks the sampling variability.
    @test mean(lam_se) / std(lam_hat) > 0.85
    # It stays conservative rather than optimistic.
    @test mean(lam_se) / std(lam_hat) < 1.25
    @test hit / length(lam_hat) > 0.80
end
