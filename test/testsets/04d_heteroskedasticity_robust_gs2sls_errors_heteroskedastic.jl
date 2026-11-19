@testset "4d. heteroskedasticity-robust GS2SLS (errors = :heteroskedastic)" begin
    _progress("4d. heteroskedasticity-robust GS2SLS (errors = :heteroskedastic)")
    GS = SpatialEconometrics.SpatialGS2SLSPanel
    SU = SpatialEconometrics.SpatialUtils

    # A panel after the unit demeaning the fit applies, stacked N rows
    # per period: N = 196 rook cells, T = 4, ρ = 0.5, λ = 0.4, one
    # endogenous regressor, error sd 2 where (W′W)ᵢᵢ is above its median
    # and skewed (centred χ²₃) innovations.
    side = 14; N = side^2; Tp = 4; n = N * Tp
    ρ0, λ0 = 0.5, 0.4
    W = row_standardize(make_rook_weights(side))
    rg = MersenneTwister(4510)
    WtW = W' * W; dd = repeat(Vector(diag(WtW)), Tp)
    A1 = WtW - spdiagm(0 => diag(WtW)); S1 = A1 + A1'; S2 = W + W'
    σ = repeat([d > median(diag(WtW)) ? 2.0 : 1.0 for d in diag(WtW)], Tp)
    dm(M) = SU.within_transform(Matrix(M), N, Tp, IndivFE())
    lagM(M) = SU.panel_spatial_lag(W, Matrix(M), N, Tp)
    X = dm(randn(rg, n, 2)); Ziv = dm(randn(rg, n, 2)); μ = randn(rg, N)
    H = hcat(X, lagM(X), lagM(lagM(X)), Ziv, lagM(Ziv))
    Sl = lu(sparse(1.0I, N, N) - ρ0 .* W); Bl = lu(sparse(1.0I, N, N) - λ0 .* W)
    Q = GS._independent_columns(H .- λ0 .* lagM(H))
    function draw(rg)
        v = randn(rg, n)
        en = Ziv * [0.6, -0.4] .+ 0.5 .* v
        ε = σ .* ((rand(rg, Chisq(3), n) .- 3) ./ sqrt(6) .* 0.8 .+ 0.6 .* v)
        y = zeros(n)
        for t in 1:Tp
            id = ((t - 1) * N + 1):(t * N)
            y[id] = Sl \ (μ .+ X[id, :] * [1.0, -1.0] .+ 0.8 .* en[id] .+ (Bl \ ε[id]))
        end
        y = dm(reshape(y, :, 1))[:, 1]; en = dm(reshape(en, :, 1))[:, 1]
        Z = hcat(lagM(reshape(y, :, 1))[:, 1], X, en)
        return y, Z
    end
    # The GS2SLS fit at the true λ, its residuals and Ψ̂ there.
    function at_truth(y, Z)
        WZ = lagM(Z)
        θ, B, Ẑ, ε̂ = GS._gs2sls_filtered(y .- λ0 .* Z[:, 1], Z .- λ0 .* WZ, Q)
        û = y .- Z * θ; Wû = lagM(reshape(û, :, 1))[:, 1]
        g, G = SU._gm_moments(û, W, N, Tp, dd)
        Ψ, a, e, α = GS._gs2sls_psi(λ0, û, Wû, Z, WZ, Ẑ, B, S1, S2, N, Tp)
        return (θ = θ, B = B, Ẑ = Ẑ, Zs = Z .- λ0 .* WZ, û = û, g = g, G = G,
                Ψ = Ψ, a = a, e = e, α = α, Ψq = SU._gm_psi(e, S1, S2, N, Tp))
    end

    # αᵣ is the derivative in θ of the sample moment
    # n⁻¹e(θ)′Aᵣe(θ), e(θ) = (I − λW)(y − Zθ), which is quadratic in θ,
    # so a central difference is exact up to rounding.
    y, Z = draw(rg)
    o = at_truth(y, Z)
    mθ(θ) = (gg = SU._gm_moments(y .- Z * θ, W, N, Tp, dd); SU._gm_m(gg[1], gg[2], λ0))
    h = 1e-5
    Dθ = reduce(hcat, [(mθ(o.θ .+ h .* I[1:4, k]) .- mθ(o.θ .- h .* I[1:4, k])) ./ (2h)
                       for k in 1:4])'
    @test Dθ ≈ o.α rtol = 1e-6
    # and aᵣ′v/n = αᵣ′(BẐ′v): the linear term is α′ times θ̂'s error.
    vv = randn(rg, n)
    @test o.a' * vv ./ n ≈ o.α' * (o.B * (o.Ẑ' * vv)) rtol = 1e-10

    # The null distribution, simulated: at the true λ, the covariance of
    # √n times the moments, from GS2SLS residuals, against the mean of
    # Ψ̂ over the same draws. Power: Ψ̂ without its linear part, which is
    # what a fit that treats the residuals as the errors would use, is
    # far off. Ψ̂ overstates by a few per cent at this size (an O(n^-½)
    # term it omits), which is conservative. Over 20 seeds (Julia 1.13):
    # |z| at most 2.36 for the means and 3.28 for Ψ̂, and z at least 11.9
    # for the quadratic-only Ψ; on 1.10, 2.51, 2.96 and 15.5.
    nsim = 1500
    M = zeros(nsim, 2); Ψh = zeros(nsim, 3); Ψq = zeros(nsim, 3)
    for r in 1:nsim
        yr, Zr = draw(rg)
        orr = at_truth(yr, Zr)
        M[r, :] .= SU._gm_m(orr.g, orr.G, λ0)
        Ψh[r, :] .= (orr.Ψ[1, 1], orr.Ψ[2, 2], orr.Ψ[1, 2])
        Ψq[r, :] .= (orr.Ψq[1, 1], orr.Ψq[2, 2], orr.Ψq[1, 2])
    end
    zmean = mean(M, dims = 1) ./ (std(M, dims = 1) ./ sqrt(nsim))
    @test all(abs.(zmean) .< 4)
    for (k, (i, j)) in enumerate(((1, 1), (2, 2), (1, 2)))
        x = n .* M[:, i] .* M[:, j]
        d = x .- Ψh[:, k]
        @test abs(mean(d)) < 4 * std(d) / sqrt(nsim)
    end
    dq = n .* M[:, 2] .^ 2 .- Ψq[:, 2]
    @test mean(dq) / (std(dq) / sqrt(nsim)) > 8

    # The fit, by hand. The panel route on levels: unit effects, T = 6
    # (one period is dropped, as the homoskedastic path does).
    Tt = 6; NT = N * Tt
    rg2 = MersenneTwister(4520)
    Xl = randn(rg2, NT, 2); Zl = randn(rg2, NT, 2); vl = randn(rg2, NT)
    enl = Zl * [0.6, -0.4] .+ 0.5 .* vl
    sdl = repeat(σ[1:N], Tt)
    yl = zeros(NT); μl = randn(rg2, N)
    for t in 1:Tt
        id = ((t - 1) * N + 1):(t * N)
        ε = sdl[id] .* (0.8 .* randn(rg2, N) .+ 0.6 .* vl[id])
        yl[id] = Sl \ (μl .+ Xl[id, :] * [1.0, -1.0] .+ 0.8 .* enl[id] .+ (Bl \ ε))
    end
    m = fit_gs2sls_panel(IndivFE(), W, Xl, reshape(enl, :, 1), Zl, yl, N, Tt;
                         errors = :heteroskedastic)
    Tq = Tt - 1; idx = (N + 1):NT
    wt(M) = SU.within_transform(Matrix(M), N, Tq, IndivFE())
    lq(M) = SU.panel_spatial_lag(W, Matrix(M), N, Tq)
    yt = wt(reshape(yl[idx], :, 1))[:, 1]
    Wyt = wt(lq(reshape(yl[idx], :, 1)))[:, 1]
    Xt = wt(Xl[idx, :]); Et = wt(reshape(enl[idx], :, 1)); Zt = wt(Zl[idx, :])
    Ht = hcat(Xt, lq(Xt), lq(lq(Xt)), Zt, lq(Zt))
    oh = GS._gs2sls_het(W, yt, Wyt, Xt, Et, Ht, N, Tq)
    @test coef(m) == vcat(oh.theta, oh.lambda)
    @test vcov(m) == oh.vcov
    @test m.errors === :heteroskedastic && m.sargan_stat == oh.hansen_j
    # σ̂² on the degrees of freedom unit demeaning leaves, as the default's.
    @test m.sigma2 ≈ dot(oh.residuals, oh.residuals) / (N * (Tt - 2)) rtol = 1e-12
    nq = N * Tq
    WtWq = repeat(Vector(diag(WtW)), Tq)
    Zf = hcat(Wyt, Xt, Et); WZf = lq(Zf)
    objective(u, λ, Υ) = (gg = SU._gm_moments(u, W, N, Tq, WtWq);
                          mm = SU._gm_m(gg[1], gg[2], λ); dot(mm, Υ * mm))
    # Step 1: λ₀ minimises the unweighted objective at the 2SLS residuals.
    Ẑ1 = Ht * (Ht \ Zf)
    u1 = yt .- Zf * ((Ẑ1' * Zf) \ (Ẑ1' * yt))
    I2 = Matrix(1.0I, 2, 2)
    l0 = oh.lambda_initial
    @test objective(u1, l0, I2) <= min(objective(u1, l0 - 1e-3, I2), objective(u1, l0 + 1e-3, I2))
    # Step 2: θ̂ is 2SLS of the model filtered at λ₀, instrumented by the
    # filtered H.
    Zs0 = Zf .- l0 .* WZf; Q0 = Ht .- l0 .* lq(Ht)
    Ẑ0 = Q0 * (Q0 \ Zs0)
    θ0 = (Ẑ0' * Zs0) \ (Ẑ0' * (yt .- l0 .* Wyt))
    @test oh.theta ≈ θ0 rtol = 1e-10
    # Step 3: λ̂ minimises the objective weighted by Ψ̂⁻¹ at step 3's
    # unweighted estimate.
    û = yt .- Zf * oh.theta; Wû = lq(reshape(û, :, 1))[:, 1]
    B0 = inv(Ẑ0' * Zs0)
    l1 = SU._gm_solve(SU._gm_moments(û, W, N, Tq, WtWq)..., I2, (-0.995, 0.995))
    Ψ1 = GS._gs2sls_psi(l1, û, Wû, Zf, WZf, Ẑ0, B0, S1, S2, N, Tq)[1]
    Υ = inv(Ψ1); lh = oh.lambda
    @test objective(û, lh, Υ) <= min(objective(û, lh - 1e-3, Υ), objective(û, lh + 1e-3, Υ))
    # The covariance: the cluster sandwich for θ̂, (J′Ψ̂⁻¹J)⁻¹/n for λ̂ and
    # the cross block from the linear part, all by unit.
    Ψ̂, ah, eh, _ = GS._gs2sls_psi(lh, û, Wû, Zf, WZf, Ẑ0, B0, S1, S2, N, Tq)
    @test oh.Psi ≈ Ψ̂ rtol = 1e-10
    Jv = oh.J
    εs = (yt .- l0 .* Wyt) .- Zs0 * oh.theta
    rws(i) = [i + (t - 1) * N for t in 1:Tq]
    sc = [Ẑ0[rws(i), :]' * εs[rws(i)] for i in 1:N]
    lin = [ah[rws(i), :]' * eh[rws(i)] for i in 1:N]
    cvec = inv(dot(Jv, Ψ̂ \ Jv)) .* (Ψ̂ \ Jv)
    Vθ = N / (N - 1) .* (B0 * sum(s * s' for s in sc) * B0)
    Cθλ = N / (N - 1) .* (B0 * sum(sc[i] .* dot(lin[i], cvec) for i in 1:N)) ./ nq
    @test oh.vcov[1:4, 1:4] ≈ Vθ rtol = 1e-8
    @test oh.vcov[1:4, 5] ≈ Cθλ rtol = 1e-8
    @test oh.vcov[5, 5] ≈ inv(dot(Jv, Ψ̂ \ Jv)) / nq rtol = 1e-10
    # Hansen's J at the two-step GMM fit of the filtered model.
    Sq = sum((Q0[rws(i), :]' * εs[rws(i)]) * (Q0[rws(i), :]' * εs[rws(i)])' for i in 1:N)
    QZ = Q0' * Zs0; Qy = Q0' * (yt .- l0 .* Wyt)
    θg = (QZ' * (Sq \ QZ)) \ (QZ' * (Sq \ Qy))
    @test m.sargan_stat ≈ dot(Qy .- QZ * θg, Sq \ (Qy .- QZ * θg)) rtol = 1e-8
    @test m.sargan_pval ≈ SU._chi2_sf(m.sargan_stat, size(Q0, 2) - 4) rtol = 1e-10

    # The default path, corrected (finding 36): its coefficient
    # covariance is the cluster sandwich by unit, and Sargan's σ̂² is on
    # the N(T_post − 1) degrees of freedom unit demeaning leaves.
    md = fit_gs2sls_panel(IndivFE(), W, Xl, reshape(enl, :, 1), Zl, yl, N, Tt)
    Wu1 = lq(reshape(u1, :, 1))[:, 1]; WWu1 = lq(reshape(Wu1, :, 1))[:, 1]
    ld = GS.estimate_lambda_gmm(u1, Wu1, WWu1, sum(abs2, W) / N)
    @test md.lambda ≈ ld rtol = 1e-10
    Zsd = Zf .- ld .* WZf; Hsd = Ht .- ld .* lq(Ht)
    Ẑd = Hsd * ((Hsd' * Hsd) \ (Hsd' * Zsd))
    Pd = hcat(Ẑd[:, 1], Zsd[:, 2:3], Ẑd[:, 4])
    θd = (Pd' * Zsd) \ (Pd' * (yt .- ld .* Wyt))
    @test coef(md)[1:4] ≈ θd rtol = 1e-10
    ud = (yt .- ld .* Wyt) .- Zsd * θd
    Ld = inv(Pd' * Zsd)
    Md = sum((Pd[rws(i), :]' * ud[rws(i)]) * (Pd[rws(i), :]' * ud[rws(i)])' for i in 1:N)
    @test vcov(md)[1:4, 1:4] ≈ N / (N - 1) .* (Ld * Md * Ld') rtol = 1e-8
    σd = dot(ud, ud) / (N * (Tq - 1))
    @test md.sigma2 ≈ σd rtol = 1e-12
    @test md.sargan_stat ≈ dot(ud, Hsd * ((Hsd' * Hsd) \ (Hsd' * ud))) / σd rtol = 1e-8

    s = sprint(show, m)
    @test occursin("Robust to heteroskedasticity (Kelejian & Prucha 2010)", s)
    @test occursin("Hansen J", s) && !occursin("Sargan", s)
    @test occursin("Sargan", sprint(show, fit_gs2sls_panel(IndivFE(), W, Xl,
                                reshape(enl, :, 1), Zl, yl, N, Tt)))
    @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), W, Xl, reshape(enl, :, 1),
                                                Zl, yl, N, Tt; errors = :bogus)
    Wd = copy(W); Wd[1, 1] = 0.1
    @test_throws ArgumentError fit_gs2sls_panel(IndivFE(), Wd, Xl, reshape(enl, :, 1),
                                                Zl, yl, N, Tt; errors = :heteroskedastic)

    # Instruments: a full-rank H is returned as it is; with an intercept
    # and a row-standardised W, WX repeats the intercept and the repeat is
    # dropped, so a cross-section (NoFE, T = 2) with an intercept fits.
    @test GS._independent_columns(Ht) === Ht
    Hc = hcat(ones(n), X[:, 1], W * ones(N) |> v -> repeat(v, Tp), X[:, 2])
    Hr = GS._independent_columns(Hc)
    @test size(Hr, 2) == 3 && rank(Hr) == 3
    Xc = hcat(ones(2N), Xl[1:2N, :])
    for e in (:homoskedastic, :heteroskedastic)
        mc = fit_gs2sls_panel(NoFE(), W, Xc, reshape(enl[1:2N], :, 1), Zl[1:2N, :],
                              yl[1:2N] .+ 1.0, N, 2; errors = e)
        @test all(isfinite, coef(mc)) && all(isfinite, stderror(mc))
        @test occursin("FE: None", sprint(show, mc))
    end

    # Power: λ's interval under heteroskedasticity. The design is
    # Arraiz et al.'s kind: an irregular map, a distance band, and error
    # sd proportional to the number of neighbours. The homoskedastic
    # path's moments assume equal variances, so its λ̂ is biased (0.28
    # for 0.4) and its interval under-covers; the robust path's covers.
    # 300 replications. Over 20 seeds: homoskedastic 0.743-0.823 on Julia
    # 1.13 and 0.723-0.847 on 1.10, robust 0.907-0.973 and 0.913-0.980;
    # at this seed 0.777 and 0.943 (1.13), 0.830 and 0.950 (1.10).
    # Coverage of λ's interval for the homoskedastic and robust GS2SLS paths on
    # an irregular map with error sd ∝ neighbours (testset 4d's power check).
    function gs2sls_het_coverage(seed, nrep)
        rg = MersenneTwister(1)
        pts = rand(rg, 300, 2); pts[1:75, :] .= 0.6 .+ 0.4 .* pts[1:75, :]
        Wb = distance_weights(pts, 0.12; decay = :power, alpha = 0.0, row_standardize = false)
        pts = pts[vec(sum(Wb .!= 0, dims = 2)) .> 0, :]
        Wb = distance_weights(pts, 0.12; decay = :power, alpha = 0.0, row_standardize = false)
        nb = vec(sum(Wb .!= 0, dims = 2)); W = row_standardize(Wb); N = size(W, 1)
        sd = nb ./ mean(nb); Tt = 6; NT = N * Tt
        Sl = lu(sparse(1.0I, N, N) - 0.3 .* W); Bl = lu(sparse(1.0I, N, N) - 0.4 .* W)
        rg = MersenneTwister(seed); ch = 0; cr = 0
        for _ in 1:nrep
            X = randn(rg, NT, 2); Z = randn(rg, NT, 2); v = randn(rg, NT)
            en = Z * [0.5, -0.3] .+ 0.3 .* v
            y = zeros(NT)
            for t in 1:Tt
                id = ((t - 1) * N + 1):(t * N)
                ε = sd .* (0.6 .* randn(rg, N) .+ 0.5 .* v[id])
                y[id] = Sl \ (X[id, :] * [1.2, -0.6] .+ 0.8 .* en[id] .+ (Bl \ ε))
            end
            a = fit_gs2sls_panel(IndivFE(), W, X, reshape(en, :, 1), Z, y, N, Tt)
            b = fit_gs2sls_panel(IndivFE(), W, X, reshape(en, :, 1), Z, y, N, Tt;
                                 errors = :heteroskedastic)
            ch += abs(a.lambda - 0.4) <= 1.959963984540054 * stderror(a)[end]
            cr += abs(b.lambda - 0.4) <= 1.959963984540054 * stderror(b)[end]
        end
        return ch / nrep, cr / nrep
    end
    cov_h, cov_r = gs2sls_het_coverage(4530, 300)
    @test 0.885 <= cov_r <= 0.995
    @test cov_h <= 0.875
end
