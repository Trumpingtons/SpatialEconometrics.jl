@testset "21b. Spatial Filtering: Basis Identities and ESF Variance" begin
    _progress("21b. Spatial Filtering: Basis Identities and ESF Variance")
    W = make_rook_weights(8); N = size(W, 1); S0 = sum(W)
    Wd = Matrix(W); Wsym = 0.5 .* (Wd .+ Wd')
    Mc = I - ones(N, N) / N
    Om = Mc * Wsym * Mc

    # ---- the eigenvector basis -----------------------------------
    se = spatial_eigenvectors(W)
    E = se.vectors
    @test size(E) == (N, N - 1)          # the constant is excluded
    @test issorted(se.values; rev = true)
    @test maximum(abs, E' * E - I) < 1e-12
    @test maximum(abs, vec(sum(E, dims = 1))) < 1e-12

    # Each column is an eigenvector of Ω = M W_sym M, and the full
    # spectrum of Ω is these eigenvalues plus one zero for the constant.
    @test maximum(norm(Om * E[:, i] - se.values[i] * E[:, i])
                  for i in 1:(N-1)) < 1e-10
    @test maximum(abs, sort(eigvals(Symmetric(Om))) .-
                       sort(vcat(0.0, se.values))) < 1e-10

    # MC_i = (N/S₀)λ_i, and must agree with the package's own statistic.
    @test maximum(abs, se.moran .- (N / S0) .* se.values) < 1e-14
    @test maximum(abs, [moran_coefficient(E[:, i], W) for i in 1:(N-1)] .-
                       se.moran) < 1e-10

    # tr(M W_sym M) = -S₀/N, because W has a zero diagonal and 1'W1 = S₀.
    # So the Moran coefficients over the complete basis sum to exactly -1
    # and average to -1/(N-1), which is E[I]. This is the identity
    # fit_re_esf's reference Moran coefficient is built on.
    @test isapprox(sum(se.values), -S0 / N; atol = 1e-10)
    @test isapprox(sum(se.moran), -1.0; atol = 1e-10)
    @test isapprox(mean(se.moran), -1 / (N - 1); atol = 1e-12)

    # min_moran_ratio screens the pool; n_candidates truncates it.
    se_scr = spatial_eigenvectors(W; min_moran_ratio = 0.25)
    @test all(se_scr.moran ./ maximum(se.moran) .>= 0.25 - 1e-12)
    @test size(spatial_eigenvectors(W; n_candidates = 7).vectors, 2) == 7

    # ---- the Gaussian ESF covariance -----------------------------
    rng = MersenneTwister(4242)
    X = randn(rng, N, 2); β = [1.0, 2.0, -1.0]
    y = (I - 0.6 * Wd) \ (β[1] .+ X * β[2:3] .+ 0.5 .* randn(rng, N))
    m = fit_esf(X, y, W)
    Xf = m.X_aug; P = length(coef(m))
    @test P == 1 + 2 + length(m.selected)
    @test dof_residual(m) == N - P

    # A Gaussian ESF fit is OLS on [1 X E_selected], so the covariance is
    # s²(X'X)⁻¹ with the *unbiased* s². Dividing by N understates every
    # standard error by exactly sqrt((N - P)/N).
    r = y .- Xf * (Xf \ y)
    @test maximum(abs, coef(m) .- (Xf \ y)) < 1e-10
    V_unb = (dot(r, r) / (N - P)) .* inv(Symmetric(Xf' * Xf))
    @test maximum(abs, stderror(m) ./ sqrt.(diag(V_unb)) .- 1) < 1e-10
    # the discarded version, rebuilt, is short by that factor
    V_ml = (dot(r, r) / N) .* inv(Symmetric(Xf' * Xf))
    @test isapprox(sqrt(diag(V_ml)[2] / diag(V_unb)[2]), sqrt((N - P) / N);
                   rtol = 1e-10)
    @test sqrt.(diag(V_ml))[2] < stderror(m)[2]
    # sigma2 stays the ML value, because the log-likelihood uses it
    @test isapprox(m.sigma2, dot(r, r) / N; rtol = 1e-10)

    # ---- RE-ESF against dense algebra ----------------------------
    mr = fit_re_esf(X, y, W)
    Er = mr.E; λr = mr.lambda; θ = mr.theta; σ2 = mr.sigma2
    Xa = mr.X_aug; Pr = size(Xa, 2); Lr = length(λr)
    V0 = I + θ .* (Er * Diagonal(λr) * Er')
    d = (θ .* λr) ./ (1 .+ θ .* λr)

    # Woodbury: (I + θEΛE')⁻¹ = I - E diag(θλ/(1+θλ)) E', since E'E = I.
    @test maximum(abs, (I - Er * Diagonal(d) * Er') - inv(V0)) < 1e-12
    @test isapprox(sum(log.(1 .+ θ .* λr)), logdet(V0); rtol = 1e-12)

    β_gls = (Xa' * (V0 \ Xa)) \ (Xa' * (V0 \ y))
    @test maximum(abs, coef(mr) .- β_gls) < 1e-10
    @test maximum(abs, vcov(mr) .- σ2 .* inv(Xa' * (V0 \ Xa))) < 1e-10

    r_gls = y .- Xa * β_gls
    rss = dot(r_gls, V0 \ r_gls)
    # unbiased, unlike the ESF branch above before its fix
    @test isapprox(σ2, rss / (N - Pr); rtol = 1e-10)
    @test !isapprox(σ2, rss / N; rtol = 1e-6)
    @test isapprox(mr.tau2, θ * σ2; rtol = 1e-12)

    # BLUP γ̂ = θΛE'V₀⁻¹r collapses to d .* (E'r).
    @test maximum(abs, mr.gamma .-
                  θ .* (λr .* (Er' * (V0 \ r_gls)))) < 1e-10
    @test maximum(abs, fitted(mr) .- (Xa * coef(mr) .+ Er * mr.gamma)) < 1e-10

    # The reference Moran coefficient averages the non-retained
    # directions, using Σλ over the complete basis = -S₀/N.
    @test isapprox(mr.moran_reference,
                   (N / S0) * ((-S0 / N) - sum(λr)) / (N - 1 - Lr);
                   rtol = 1e-10)
    @test isapprox(loglikelihood(mr),
                   -0.5 * (N * log(2π * σ2) + logdet(V0) + rss / σ2);
                   rtol = 1e-10)

    # REML: Brent must not be beaten by a grid over the same bracket.
    function reml_dense(lt)
        t = exp(lt); Vt = I + t .* (Er * Diagonal(λr) * Er')
        Vi = inv(Vt); XtX = Xa' * Vi * Xa; b = XtX \ (Xa' * Vi * y)
        rr = dot(y .- Xa * b, Vi * (y .- Xa * b))
        -0.5 * (logdet(Vt) + (N - Pr) * log(rr / (N - Pr)) + logdet(XtX))
    end
    @test reml_dense(log(θ)) >=
          maximum(reml_dense(g) for g in range(-12, 8; length = 801)) - 1e-8

    # ---- RE-ESF recovers its own variance components --------------
    function reesf_recovery(seed, R; tau2 = 0.5, sig = 0.5)
        eb = spatial_eigenvectors(W; min_moran_ratio = 0.25)
        ps = findall(>(1e-10), eb.values)
        Eb = eb.vectors[:, ps]; lb = eb.values[ps]
        rg = MersenneTwister(seed)
        t2 = zeros(R); s2 = zeros(R); cov_hit = 0
        for i in 1:R
            Xr = randn(rg, N, 2)
            g = sqrt(tau2) .* (sqrt.(lb) .* randn(rg, length(lb)))
            yr = β[1] .+ Xr * β[2:3] .+ Eb * g .+ sig .* randn(rg, N)
            mm = fit_re_esf(Xr, yr, W; eigen_basis = eb)
            t2[i] = mm.tau2; s2[i] = mm.sigma2
            abs(coef(mm)[2] - β[2]) <= 1.96 * stderror(mm)[2] &&
                (cov_hit += 1)
        end
        (mean(t2), mean(s2), cov_hit / R)
    end
    mt2, ms2, cvg = reesf_recovery(99, 120)
    @test 0.38 < mt2 < 0.65        # true 0.5
    @test 0.21 < ms2 < 0.30        # true 0.25
    @test cvg > 0.88               # nominal 0.95
end
