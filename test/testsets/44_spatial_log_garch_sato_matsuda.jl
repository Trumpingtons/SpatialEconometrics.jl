@testset "44. Spatial log-GARCH (Sato & Matsuda 2021) and covariate-distance W" begin
    _progress("44. Spatial log-GARCH (Sato & Matsuda 2021) and covariate-distance W")
    SG = SpatialEconometrics.SpatialGARCH
    lqcov = SpatialEconometrics.SpatialUtils._lq_cov
    # The design shared by 44a–44d: rook 7×7, one covariate.
    Lg = 7; ng = Lg^2; Wg = make_rook_weights(Lg); Wgd = Matrix(Wg)
    zg = randn(MersenneTwister(44_001), ng)
    Xg = hcat(ones(ng), zg)
    λ0, ρ0, α0, δ0 = 0.4, 0.2, 0.5, 1.0; a0 = λ0 + ρ0
    Rλ0 = I - λ0 * Wgd; Ra0 = I - a0 * Wgd
    gdata(Y) = SG._GData(Y, Wg * Y, Xg, SparseMatrixCSC{Float64, Int}(Wg),
                         ComplexF64.(eigvals(Wgd)), ng, 2)
    Ysim(U) = Ra0 \ (α0 .+ δ0 .* zg .+ Rλ0 * U)
    Yg = Ysim(log.(randn(MersenneTwister(44_002), ng) .^ 2))
    dg = gdata(Yg)
    # The full quasi log-likelihood (Sato & Matsuda eq. 8), written densely.
    function ll_dense(ψ)
        β = ψ[1:2]; s2 = ψ[3]; Rl = I - ψ[4] * Wgd; Ra = I - ψ[5] * Wgd
        V = Rl \ (Ra * Yg - Xg * β)
        return -ng / 2 * log(2π * s2) - dot(V, V) / (2s2) + logabsdet(Ra)[1] - logabsdet(Rl)[1]
    end
    ψt = [0.3, 0.8, 4.0, 0.35, 0.55]

    @testset "44a. the likelihood against a dense computation" begin
        P = SG._pieces(dg, ψt[4], ψt[5]; β = ψt[1:2], σ2 = ψt[3])
        @test SG._loglik(dg, ψt[4], ψt[5], P) ≈ ll_dense(ψt) atol = 1e-10
        # The profiled (β, σ²) solve their first-order conditions at fixed θ.
        Pc = SG._pieces(dg, 0.35, 0.55)
        g, _ = SG._score_hessian(dg, 0.35, 0.55, Pc)
        @test maximum(abs, g[1:3]) < 1e-10
        @test SG._loglik(dg, 0.35, 0.55, Pc) ≈ ll_dense(vcat(Pc.β, Pc.σ2, 0.35, 0.55)) atol = 1e-10
        for x in (-0.7, 0.3, 0.9)
            @test SG._logdetR(dg, x) ≈ logabsdet(I - x * Wgd)[1] atol = 1e-10
            @test SG._trG(dg, x) ≈ tr(Wgd / (I - x * Wgd)) atol = 1e-10
        end
        # No upper bound as λ → 1: with W1 = 1 the intercept absorbs what
        # R(λ)⁻¹ inflates, so σ̂²(λ) converges while −log|R(λ)| grows by
        # log 10 for each decade of 1 − λ.
        ℓk = [SG._concentrated(dg, 1 - 10.0^-k, a0) for k in 6:8]
        @test ℓk[2][1] - ℓk[1][1] ≈ log(10) atol = 1e-3
        @test ℓk[3][1] - ℓk[2][1] ≈ log(10) atol = 1e-3
        @test ℓk[3][2].σ2 ≈ ℓk[1][2].σ2 rtol = 1e-4
    end

    @testset "44b. analytic score and Hessian against automatic differentiation" begin
        for ψ in (ψt, [-0.2, 1.3, 2.5, -0.3, 0.7], [0.0, 0.0, 6.0, 0.8, -0.1])
            P = SG._pieces(dg, ψ[4], ψ[5]; β = ψ[1:2], σ2 = ψ[3])
            g, H = SG._score_hessian(dg, ψ[4], ψ[5], P)
            @test g ≈ ForwardDiff.gradient(ll_dense, ψ) atol = 1e-9
            @test H ≈ ForwardDiff.hessian(ll_dense, ψ) atol = 1e-8
        end
    end

    @testset "44c. the score covariance: forms, and the moment formula by brute force" begin
        # At the true ψ the score is a'V + V'AV − σ²tr A in V = log ε² − c.
        c = -1.2703628454614782; s2 = π^2 / 2
        β0 = [α0 + (1 - λ0) * c, δ0]
        forms = SG._lq_forms(dg, λ0, a0, β0, s2)
        for seed in (1, 2)
            U = log.(randn(MersenneTwister(seed), ng) .^ 2); V = U .- c
            d = gdata(Ysim(U))
            g, _ = SG._score_hessian(d, λ0, a0, SG._pieces(d, λ0, a0; β = β0, σ2 = s2))
            Slq = [(A === nothing ? 0.0 : dot(V, A * V) - s2 * tr(A)) + dot(av, V) for (A, av) in forms]
            @test Slq ≈ g atol = 1e-9
        end
        # _lq_cov against E[S_i S_j] summed over every index quadruple,
        # with unequal variances and third and fourth moments.
        rng = MersenneTwister(44_003); m = 6
        sv = 0.5 .+ rand(rng, m); m3 = randn(rng, m); m4 = 3 .* sv .^ 2 .+ 2 .* rand(rng, m)
        function mom4(i, j, k, l)               # E[v_i v_j v_k v_l]
            i == j == k == l && return m4[i]
            s = 0.0
            (i == j && k == l) && (s += sv[i] * sv[k])
            (i == k && j == l) && (s += sv[i] * sv[j])
            (i == l && j == k) && (s += sv[i] * sv[j])
            return s
        end
        mom3(i, j, k) = (i == j == k) ? m3[i] : 0.0
        Fm = Any[(Matrix(Symmetric(randn(rng, m, m))), randn(rng, m)) for _ in 1:2]
        push!(Fm, (nothing, randn(rng, m)))
        brute = zeros(3, 3)
        for p in 1:3, q in 1:3
            A, a = Fm[p]; B, b = Fm[q]
            A0 = A === nothing ? zeros(m, m) : A; B0 = B === nothing ? zeros(m, m) : B
            e = 0.0
            for i in 1:m, j in 1:m, k in 1:m, l in 1:m
                e += A0[i, j] * B0[k, l] * mom4(i, j, k, l)
            end
            for i in 1:m, k in 1:m, l in 1:m
                e += (a[i] * B0[k, l] + b[i] * A0[k, l]) * mom3(i, k, l)
            end
            e += sum(a .* b .* sv)
            e -= sum(diag(A0) .* sv) * sum(diag(B0) .* sv)
            brute[p, q] = e
        end
        @test lqcov(Fm, sv, m3, m4) ≈ brute rtol = 1e-12
    end

    @testset "44d. null test: the exact score covariance against simulation" begin
        # ε² = G/k with G ~ Gamma(k), so log G has cumulants ψ⁽ʲ⁾(k);
        # k = 1/2 is normal ε. Log-normal ε makes V itself normal.
        ζ3 = 1.2020569031595942
        laws = [
            (:normal,  rng -> log.(randn(rng, ng) .^ 2),
             -1.2703628454614782, π^2 / 2, -14ζ3, π^4 + 3 * (π^2 / 2)^2),
            (:gamma2,  rng -> log.((-log.(rand(rng, ng)) .- log.(rand(rng, ng))) ./ 2),
             0.42278433509846713 - log(2), π^2 / 6 - 1, 2 - 2ζ3,
             π^4 / 15 - 6 + 3 * (π^2 / 6 - 1)^2),
            (:lognorm, rng -> 2 .* randn(rng, ng) .- 2, -2.0, 4.0, 0.0, 48.0),
        ]
        R = 4000
        for (name, drawU, c, s2, μ3, μ4) in laws
            β0 = [α0 + (1 - λ0) * c, δ0]
            forms = SG._lq_forms(dg, λ0, a0, β0, s2)
            Ω = lqcov(forms, s2, μ3, μ4)
            rng = MersenneTwister(44_100 + length(String(name)))
            S = zeros(R, 5); Ωhat = zeros(5, 5)
            for r in 1:R
                U = drawU(rng); d = gdata(Ysim(U))
                S[r, :] = SG._score_hessian(d, λ0, a0, SG._pieces(d, λ0, a0; β = β0, σ2 = s2))[1]
                V = U .- c
                Ωhat .+= lqcov(forms, s2, mean(V .^ 3), mean(V .^ 4)) ./ R
            end
            C = cov(S); mS = vec(mean(S, dims = 1))
            se = [std((S[:, i] .- mS[i]) .* (S[:, j] .- mS[j])) / sqrt(R) for i in 1:5, j in 1:5]
            @test maximum(abs.(C .- Ω) ./ se) < 4
            @test maximum(abs.(Ωhat .- Ω) ./ se) < 4
            # Power: the Gaussian information, right only for normal V,
            # fails the same test by a wide margin when ε is normal.
            Ωg = lqcov(forms, s2, 0.0, 3s2^2)
            name === :normal && @test maximum(abs.(C .- Ωg) ./ se) > 15
            name === :lognorm && @test Ωg ≈ Ω
        end
    end

    # A larger fit for 44e–44f.
    Lf = 20; nf = Lf^2; Wf = make_rook_weights(Lf); Wfd = Matrix(Wf)
    rngf = MersenneTwister(44_010); zf = randn(rngf, nf)
    λf, ρf = 0.45, 0.45
    Uf = log.(randn(rngf, nf) .^ 2)
    Yf = (I - (λf + ρf) * Wfd) \ (0.5 .+ zf .+ (I - λf * Wfd) * Uf)
    yf = sqrt.(exp.(Yf)) .* sign.(randn(rngf, nf))
    mf = fit_spatial_log_garch(yf, Wf; Z = zf)

    @testset "44e. the fit: recovery, step 2, the sandwich, invariance" begin
        @test mf.gradient_norm < 1e-8
        @test all(abs.(coef(mf) .- [1.0, λf, ρf]) .< 4 .* stderror(mf))
        @test coefnames(mf) == ["z1", "λ", "ρ"]
        # Step 2 (eq. 11) and log ĥ, recomputed densely.
        Rl = I - mf.lambda * Wfd; a = mf.lambda + mf.rho
        C = Rl \ ((I - a * Wfd) * Yf .- zf .* mf.delta[1])
        @test mf.alpha ≈ (1 - mf.lambda) * log(mean(exp.(C))) rtol = 1e-10
        @test abs(mf.alpha - 0.5) < 0.3
        @test mf.log_volatility ≈ Rl \ (mf.rho .* (Wfd * Yf) .+ mf.alpha .+ zf .* mf.delta[1]) rtol = 1e-10
        # The sandwich, rebuilt from its pieces.
        d = SG._GData(Yf, Wf * Yf, hcat(ones(nf), zf), SparseMatrixCSC{Float64, Int}(Wf),
                      ComplexF64.(eigvals(Wfd)), nf, 2)
        P = SG._pieces(d, mf.lambda, a)
        _, H = SG._score_hessian(d, mf.lambda, a, P)
        Ω = SG._score_cov(d, mf.lambda, a, P.β, P.σ2, mean(P.V .^ 3), mean(P.V .^ 4))
        J = Matrix(1.0I, 5, 5); J[5, 4] = -1.0
        Vs = J * (H \ Ω / H) * J'
        @test mf.vcov ≈ Vs[[2, 4, 5], [2, 4, 5]] rtol = 1e-8
        # The signs of y and the order of the units do not matter.
        @test coef(fit_spatial_log_garch(abs.(yf), Wf; Z = zf)) ≈ coef(mf) atol = 1e-12
        perm = randperm(MersenneTwister(44_011), nf)
        mp = fit_spatial_log_garch(yf[perm], Wf[perm, perm]; Z = zf[perm])
        @test coef(mp) ≈ coef(mf) atol = 1e-6
        @test stderror(mp) ≈ stderror(mf) rtol = 1e-5
        @test mp.alpha ≈ mf.alpha atol = 1e-6
    end

    @testset "44f. inputs are checked" begin
        y0 = copy(yf); y0[3] = 0.0
        @test_throws ArgumentError fit_spatial_log_garch(y0, Wf; Z = zf)
        @test_throws ArgumentError fit_spatial_log_garch(yf, Wf; Z = hcat(zf, 2 .* zf))
        @test_throws ArgumentError fit_spatial_log_garch(yf, Wf; Z = hcat(ones(nf), zf))
        @test_throws ArgumentError fit_spatial_log_garch(yf, sparse(1.0 .* (Wfd .> 0)); Z = zf)
        @test_throws DimensionMismatch fit_spatial_log_garch(yf[1:end-1], Wf; Z = zf)
        # A solver that cannot meet tol throws rather than return a non-root.
        @test_throws ErrorException fit_spatial_log_garch(yf, Wf; Z = zf, tol = 1e-30, maxiter = 2)
        # A search that runs to the edge of `bounds` says so.
        eb = try fit_spatial_log_garch(yf, Wf; Z = zf, bounds = (-0.99, 0.3)); nothing catch e; e end
        @test eb isa ErrorException && occursin("edge of `bounds`", eb.msg)
    end

    @testset "44g. covariate_weights" begin
        rng = MersenneTwister(44_020)
        Xc = hcat(randn(rng, 60), 100 .* randn(rng, 60))
        @test covariate_weights(Xc, 5; standardize = false) == knn_weights(Xc, 5; binary = true)
        Xs = (Xc .- mean(Xc, dims = 1)) ./ std(Xc, dims = 1)
        @test covariate_weights(Xc, 5) == knn_weights(Xs, 5; binary = true)
        # Standardising makes each covariate's units irrelevant.
        @test covariate_weights(Xc .* [3.0 0.01], 5) == covariate_weights(Xc, 5)
        @test all(sum(covariate_weights(Xc, 5), dims = 2) .≈ 1)
        @test_throws ArgumentError covariate_weights(hcat(Xc, ones(60)), 5)
    end
end
