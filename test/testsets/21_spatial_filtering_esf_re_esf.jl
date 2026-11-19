@testset "21. Spatial Filtering (ESF / RE-ESF)" begin
    _progress("21. Spatial Filtering (ESF / RE-ESF)")
    W = make_rook_weights(11); N = size(W, 1)
    IN = sparse(1.0I, N, N)
    S0 = sum(W)

    # ── Eigenvector basis identities (all exact) ──
    se = spatial_eigenvectors(W)
    @test se.N == N
    @test length(se.values) == N - 1           # complement of the constant
    @test size(se.vectors) == (N, N - 1)
    @test issorted(se.values; rev=true)

    # Each eigenvector's own Moran coefficient equals (N/S0)*lambda.
    emp = [moran_coefficient(se.vectors[:, i], W) for i in 1:length(se.values)]
    @test maximum(abs.(emp .- se.moran)) < 1e-8
    # Orthonormal columns.
    @test maximum(abs.(transpose(se.vectors) * se.vectors - I)) < 1e-8
    # Orthogonal to the constant vector, so the intercept is never absorbed.
    @test maximum(abs.(vec(sum(se.vectors, dims=1)))) < 1e-8
    # Centred spectrum sums to tr(M*Wsym*M) = -S0/N.
    @test sum(se.values) ≈ -S0 / N atol = 1e-8

    # Screening keeps a strict subset, ordered by Moran coefficient.
    se_scr = spatial_eigenvectors(W; min_moran_ratio=0.25)
    @test length(se_scr.values) < length(se.values)
    @test minimum(se_scr.moran) >= 0.25 * maximum(se.moran) - 1e-8

    # ── Gaussian ESF on SAR-generated data ──
    rng = MersenneTwister(5)
    X = randn(rng, N, 2)
    y = (IN - 0.6 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))
    m = fit_esf(X, y, W; family=:gaussian, max_eigenvectors=15, varnames=["x1", "x2"])
    @test nobs(m) == N
    @test coefnames(m)[1] == "(Intercept)"
    @test length(coef(m)) == 3 + length(m.selected)
    @test m.n_covariates == 2
    # Filtering must leave less residual autocorrelation than OLS.
    @test abs(m.moran_final) < abs(m.moran_initial)
    @test !isempty(m.selected)
    s = selected_eigenvectors(m)
    @test s.indices == m.selected
    @test length(s.moran) == length(s.indices)
    @test allunique(s.indices)
    @test length(residuals(m)) == N
    @test fitted(m) ≈ predict(m)

    # ── Poisson ESF: slope recovery ──
    function rpois(rng, lam)
        L = exp(-lam); k = 0; p = 1.0
        while true
            p *= rand(rng)
            p <= L && return Float64(k)
            k += 1
            k > 10_000 && return Float64(k)
        end
    end
    rng = MersenneTwister(31)
    Xc = randn(rng, N, 2)
    sig = 1.2 * se_scr.vectors[:, 2] * sqrt(N) * 0.4 +
          0.8 * se_scr.vectors[:, 5] * sqrt(N) * 0.4
    eta = 0.6 .+ Xc * [0.5, -0.35] .+ sig
    yp = [rpois(rng, exp(e)) for e in eta]
    mp = fit_esf(Xc, yp, W; family=:poisson, max_eigenvectors=8,
                 eigen_basis=se_scr, varnames=["x1", "x2"])
    @test coef(mp)[2] ≈ 0.5 atol = 0.15
    @test coef(mp)[3] ≈ -0.35 atol = 0.15
    @test all(>(0.0), fitted(mp))

    # ── Logit ESF: valid probabilities ──
    yb = Float64[(1.0 / (1.0 + exp(-e))) > rand(rng) ? 1.0 : 0.0 for e in eta]
    ml = fit_esf(Xc, yb, W; family=:logit, max_eigenvectors=8,
                 eigen_basis=se_scr, varnames=["x1", "x2"])
    @test all(p -> 0.0 <= p <= 1.0, fitted(ml))
    @test isfinite(loglikelihood(ml))

    # ── Negative binomial ESF: dispersion is estimated and positive ──
    yn = Float64[]
    for e in eta
        g = sum(-log(rand(rng)) for _ in 1:2) * 0.5   # mean-1 gamma, shape 2
        push!(yn, rpois(rng, exp(e) * g))
    end
    mn = fit_esf(Xc, yn, W; family=:negbin, max_eigenvectors=8,
                 eigen_basis=se_scr, varnames=["x1", "x2"])
    @test mn.alpha > 0.0
    @test var(yn) / mean(yn) > 1.0                    # genuinely overdispersed
    @test abs(mn.moran_final) < abs(mn.moran_initial)

    # ── RE-ESF on correctly specified data ──
    pos = findall(>(1e-10), se_scr.values)
    E = se_scr.vectors[:, pos]; lam = se_scr.values[pos]; L = length(pos)
    rng = MersenneTwister(3)
    Xr = randn(rng, N, 2)
    gam = sqrt(0.8) .* sqrt.(lam) .* randn(rng, L)
    yr = 1.0 .+ Xr * [1.5, -0.8] .+ E * gam .+ 0.5 .* randn(rng, N)
    mr = fit_re_esf(Xr, yr, W; varnames=["x1", "x2"])
    @test length(coef(mr)) == 3
    @test coef(mr)[1] ≈ 1.0 atol = 0.25
    @test coef(mr)[2] ≈ 1.5 atol = 0.25
    @test coef(mr)[3] ≈ -0.8 atol = 0.25
    @test mr.sigma2 ≈ 0.25 atol = 0.15
    @test mr.theta > 0.0
    @test mr.tau2 ≈ mr.theta * mr.sigma2
    @test length(mr.gamma) == L

    # Effective df sits strictly between the fixed-effect count and N.
    @test 3.0 < mr.edf < Float64(N)

    # The conditional residual Moran is expected to be NEGATIVE: removing Eγ̂
    # strips the positive-MC directions. The benchmark is the complement's
    # expectation, not zero.
    nrest = N - 1 - L
    ref_closed = (N / S0) * ((-S0 / N) - sum(lam)) / nrest
    @test mr.moran_reference ≈ ref_closed atol = 1e-10
    # Closed form must equal the mean MC of the non-retained eigenvectors.
    keep = sortperm(se.values; rev=true)[1:L]
    rest = setdiff(1:length(se.values), keep)
    @test mr.moran_reference ≈ mean(se.moran[rest]) atol = 1e-8
    @test mr.moran_reference < 0.0
    # A healthy fit lands at or above that reference, not near zero.
    @test mr.moran_final > mr.moran_reference - 0.15
    @test mr.moran_final < 0.0
    # The random effects really did absorb the positive-MC content.
    share_before = norm(transpose(E) * (yr - hcat(ones(N), Xr) * (hcat(ones(N), Xr) \ yr)))
    share_after = norm(transpose(E) * residuals(mr))
    @test share_after < share_before

    # V₀⁻¹ = I - E D E' is a genuine inverse of I + θEΛE'.
    d = (mr.theta .* lam) ./ (1.0 .+ mr.theta .* lam)
    V0 = I + mr.theta * (E * (lam .* transpose(E)))
    V0inv = I - E * (d .* transpose(E))
    @test maximum(abs.(V0 * V0inv - I)) < 1e-8

    # ── Input validation ──
    @test_throws AssertionError fit_esf(X, y[1:end-1], W)
    @test_throws AssertionError fit_re_esf(X, y[1:end-1], W)
end
