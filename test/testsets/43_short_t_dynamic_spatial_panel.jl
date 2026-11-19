@testset "43. Short-T dynamic spatial panel with common shocks (Bai & Chen 2026)" begin
    _progress("43. Short-T dynamic spatial panel with common shocks (Bai & Chen 2026)")
    CS = SpatialEconometrics.CommonShockDynamicPanel
    function cs_knn(rng, N, k)
        P = rand(rng, N, 2); W = zeros(N, N)
        for i in 1:N
            dd = [sum(abs2, P[i, :] .- P[j, :]) for j in 1:N]; dd[i] = Inf
            for j in partialsortperm(dd, 1:k); W[i, j] = 1 / k; end
        end
        return W
    end
    # _CSData from N×(T+1) outcomes, per-regressor N×T matrices and controls.
    function cs_data(Yall, Xk, Zs, W)
        ev = eigvals(W)
        return CS._CSData(size(Yall, 1), size(Yall, 2) - 1, length(Xk), 2, size(Zs, 2),
                          Yall[:, 2:end], Yall[:, 1:(end - 1)], W * Yall[:, 2:end], Xk, Zs, W,
                          real.(ev), imag.(ev))
    end
    # Bai & Chen's baseline design (§6.1.1): λ ~ N(0, I₂), y₀ = 1'λ + ν.
    function cs_base(rng, W, N, T; ρ = 0.5, φ = 0.25, β = [0.8, -0.3])
        Λ = randn(rng, N, 2); y0 = Λ * ones(2) .+ randn(rng, N)
        B = lu(I - ρ * W); ys = [y0]; Xs = [zeros(N, 2)]
        for t in 1:T
            Xt = randn(rng, N, 2)
            push!(ys, B \ (φ .* ys[end] .+ Xt * β .+ 0.3 * sin(2π * t / T) .+ Λ * randn(rng, 2) .+
                           sqrt(0.5 + t / (5T)) .* randn(rng, N)))
            push!(Xs, Xt)
        end
        return reduce(vcat, ys), reduce(vcat, Xs)
    end

    rng = MersenneTwister(43_001)
    N, T = 7, 4
    W = cs_knn(rng, N, 3)
    Yall = randn(rng, N, T + 1)
    Xk = [randn(rng, N, T) for _ in 1:2]
    Zs = CS._standardise(randn(rng, N, 3), ["a", "b", "c"])
    d = cs_data(Yall, Xk, Zs, W)
    α = CS._pack(0.3, 0.4, [0.5, -0.2], randn(rng, T), randn(rng, 2, 3),
                 vcat(Matrix(1.0I, 2, 2), randn(rng, T - 2, 2)), [1.0 0.3; 0.3 0.8],
                 [0.7, 0.9, 1.1, 0.6])

    @testset "43a. the likelihood equals the stacked NT likelihood" begin
        # Built from scratch: C(θ)y = h + e over all NT entries, the
        # Jacobian from log|det C|, and the NT×NT covariance I_N⊗Σ_u.
        ρ, φ, β, δ, A, F, Ση, σ2 = CS._unpack(d, α)
        RT = Matrix(1.0I, T, T); for t in 2:T; RT[t, t - 1] = -φ; end
        C = kron(I(N), RT) - ρ * kron(W, I(T))
        e1 = [1.0; zeros(T - 1)]
        h = reduce(vcat, [Xk[1][i, :] .* β[1] .+ Xk[2][i, :] .* β[2] .+ φ * Yall[i, 1] .* e1 .+
                          δ .+ F * A * Zs[i, :] for i in 1:N])
        e = C * vec(Yall[:, 2:end]') - h
        Σf = kron(I(N), F * Ση * F' + Diagonal(σ2))
        ℓ = logabsdet(C)[1] - logdet(Σf) / 2 - dot(e, Σf \ e) / 2
        @test CS._loglik(d, α) ≈ ℓ rtol = 1e-12
    end

    @testset "43b. the analytic score and Hessian equal automatic differentiation" begin
        gad = ForwardDiff.gradient(a -> CS._loglik(d, a), α)
        @test maximum(abs, CS._score(d, α) .- gad) < 1e-10 * maximum(abs, gad)
        Had = -ForwardDiff.hessian(a -> CS._loglik(d, a), α)
        @test maximum(abs, CS._neg_hessian(d, α) .- Had) < 1e-10 * maximum(abs, Had)
    end

    @testset "43c. the rearranged scores (Appendix D), against explicit blocks" begin
        ρ, φ, β, δ, A, F, Ση, σ2 = CS._unpack(d, α)
        RT = Matrix(1.0I, T, T); for t in 2:T; RT[t, t - 1] = -φ; end
        G = inv(kron(I(N), RT) - ρ * kron(W, I(T)))
        LT = zeros(T, T); for t in 2:T; LT[t, t - 1] = 1; end
        WG = kron(W, I(T)) * G; LG = kron(I(N), LT) * G
        blk(A, i, j) = A[((i - 1) * T + 1):(i * T), ((j - 1) * T + 1):(j * T)]
        E = CS._E(d, CS._U(d, ρ, φ, β), δ, A, F)
        M = inv(F * Ση * F' + Diagonal(σ2))
        Pr = [E[i, :]' * M * blk(WG, i, j) * E[j, :] for i in 1:N, j in 1:N]
        Pf = [E[i, :]' * M * blk(LG, i, j) * E[j, :] for i in 1:N, j in 1:N]
        Pρ, Pφ, WB1 = CS._pair_mats(d, α)
        @test maximum(abs, Pρ .- Pr) < 1e-12 * maximum(abs, Pr)
        @test maximum(abs, Pφ .- Pf) < 1e-12 * maximum(abs, Pf)
        # Σ_i tr(M𝒲_iiΣ_u) is the Jacobian term T tr(B⁻¹W), unit by unit.
        Σu = inv(M)
        @test [tr(M * blk(WG, i, i) * Σu) for i in 1:N] ≈ T .* diag(WB1) atol = 1e-12
        mg = CS._martingale(d, α)
        @test vec(sum(mg.g; dims = 1)) ≈ CS._score(d, α) rtol = 1e-12
        Ωv = zeros(2, 2)
        for i in 1:N, j in 1:(i - 1)
            v = [Pr[i, j] + Pr[j, i], Pf[i, j] + Pf[j, i]]
            Ωv .+= v * v'
        end
        @test mg.Ωv[1:2, 1:2] ≈ Ωv rtol = 1e-12
        @test all(iszero, mg.Ωv[3:end, :])
        @test mg.g[:, 3:end] == mg.gdag[:, 3:end]
    end

    @testset "43d. Ω̂_M does not depend on the order of the units; g_i does" begin
        perm = [3, 7, 1, 5, 2, 6, 4]
        dp = cs_data(Yall[perm, :], [x[perm, :] for x in Xk], Zs[perm, :], W[perm, perm])
        m1 = CS._martingale(d, α); m2 = CS._martingale(dp, α)
        Ω1 = m1.Ωdag .+ m1.Ωv; Ω2 = m2.Ωdag .+ m2.Ωv
        @test Ω2 ≈ Ω1 rtol = 1e-11
        @test maximum(abs, m2.g[:, 1] .- m1.g[perm, 1]) > 1e-3 * maximum(abs, m1.g[:, 1])
    end

    @testset "43e. E[Ω̂_M(α₀)] = Var(Σg_i(α₀)) under skewed errors" begin
        # Conditional on C_N (y₀, X, W and the controls held fixed), the
        # g_i(α₀) are martingale differences and (52)–(53) make Ω̂_M(α₀)
        # unbiased for Var(Σg_i) in finite samples. Paired z-tests on
        # S S' − Ω̂_M. The raw outer product Σ s_i s_i' misses the
        # cross-unit terms of the ρ score, so it must fail the same test.
        r2 = MersenneTwister(43_005)
        Nn, Tn = 30, 4
        Wn = cs_knn(r2, Nn, 4)
        Xn = [randn(r2, Nn, Tn) for _ in 1:2]
        y0 = randn(r2, Nn)
        Zn = CS._standardise(hcat(y0, randn(r2, Nn), Wn * y0), ["a", "b", "c"])
        ρ0, φ0, β0 = 0.4, 0.3, [0.8, -0.3]
        δ0 = [0.1, -0.2, 0.3, 0.0]; A0 = [0.5 0.2 0.3; -0.2 0.4 0.1]
        F0 = [1.0 0; 0 1; 0.6 -0.4; 0.2 0.9]; Ση0 = [0.8 0.2; 0.2 0.5]; σ20 = [0.6, 0.8, 1.0, 1.2]
        α0 = CS._pack(ρ0, φ0, β0, δ0, A0, F0, Ση0, σ20)
        chi(k) = (sum(randn(r2, k, 3) .^ 2; dims = 2)[:, 1] .- 3) ./ sqrt(6)
        Lη = cholesky(Ση0).L; B = lu(I - ρ0 * Wn)
        R = 8000; p = length(α0)
        D = zeros(R, p, p); Dn = zeros(R)
        for rep in 1:R
            η = reduce(hcat, [chi(Nn) for _ in 1:2]) * Lη'
            U = δ0' .+ (Zn * A0' .+ η) * F0' .+ reduce(hcat, [sqrt(σ20[t]) .* chi(Nn) for t in 1:Tn])
            ys = [y0]
            for t in 1:Tn
                push!(ys, B \ (φ0 .* ys[end] .+ hcat(Xn[1][:, t], Xn[2][:, t]) * β0 .+ U[:, t]))
            end
            dn = cs_data(reduce(hcat, ys), Xn, Zn, Wn)
            mg = CS._martingale(dn, α0)
            S = vec(sum(mg.g; dims = 1))
            D[rep, :, :] = S * S' .- (mg.Ωdag .+ mg.Ωv)
            su = CS._unit_scores(dn, α0)[:, 1] .+ Tn * CS._dlogdetB(dn, ρ0) / Nn
            Dn[rep] = S[1]^2 - sum(abs2, su)
        end
        z = dropdims(mean(D; dims = 1) ./ (std(D; dims = 1) ./ sqrt(R)); dims = 1)
        # 325 distinct entries; the largest |z| is recorded in docs/AUDIT.md.
        @test maximum(abs, z) < 4.5
        zn = mean(Dn) / (std(Dn) / sqrt(R))
        @test zn > 5
    end

    r3 = MersenneTwister(43_007)
    Nf, Tf = 600, 5
    Wf = cs_knn(r3, Nf, 8)
    yf, Xf = cs_base(r3, Wf, Nf, Tf)
    fit = fit_common_shock_dynamic_panel(yf, Xf, Wf, Nf, Tf + 1; r = 2, L = 0, varnames = ["a", "b"])

    @testset "43f. the fit solves the first-order conditions" begin
        @test fit.converged
        @test fit.score_norm < 1e-8
        @test coefnames(fit) == ["rho", "phi", "a", "b"]
        @test fit.F[1:2, :] == Matrix(1.0I, 2, 2)
        @test isposdef(fit.Sigma_eta) && all(>(0), fit.sigma2)
        θ0 = [0.5, 0.25, 0.8, -0.3]
        @test all(abs.(coef(fit) .- θ0) .< 4 .* stderror(fit))
        # The pair factor N/(N − p) only enlarges the variance.
        @test all(diag(fit.vcov) .>= diag(fit.vcov_uncorrected))
        @test fit.vcov ≈ fit.vcov_alpha[1:4, 1:4]
        s = sprint(show, fit)
        @test occursin("Bai & Chen", s) && occursin("L = 0", s)
    end

    @testset "43g. the θ block of the sandwich is the profiled covariance (28)" begin
        dd = let Yf = reshape(yf, Nf, Tf + 1)
            Xk2 = [reshape(Xf[:, k], Nf, Tf + 1)[:, 2:end] for k in 1:2]
            C, cn = CS._serl_controls(reduce(hcat, [vec(mean(x; dims = 2)) for x in Xk2]),
                                      ["a", "b"], Yf[:, 1], Wf, 0)
            cs_data(Yf, Xk2, CS._standardise(C, cn), Wf)
        end
        H = CS._neg_hessian(dd, fit.alpha)
        mg = CS._martingale(dd, fit.alpha)
        p = length(fit.alpha)
        Ω = mg.Ωdag .+ (Nf / (Nf - p)) .* mg.Ωv
        θ = 1:4; ϕ = 5:p
        Hθϕ = H[θ, θ] - H[θ, ϕ] * (H[ϕ, ϕ] \ H[ϕ, θ])
        P = H[θ, ϕ] / H[ϕ, ϕ]
        J = hcat(Matrix(1.0I, 4, 4), -P)
        Vp = Hθϕ \ (J * Ω * J') / Hθϕ'
        @test fit.vcov ≈ (Vp .+ Vp') ./ 2 rtol = 1e-8
    end

    @testset "43h. relabelling the units leaves the fit unchanged" begin
        perm = randperm(MersenneTwister(43_008), Nf)
        ip = vcat([perm .+ (t * Nf) for t in 0:Tf]...)
        fp = fit_common_shock_dynamic_panel(yf[ip], Xf[ip, :], Wf[perm, perm], Nf, Tf + 1; r = 2, L = 0)
        # Both fits stop at |∂ℓ/∂α|/N < 1e-8, so they agree to that
        # tolerance, not to rounding.
        @test coef(fp) ≈ coef(fit) atol = 1e-6
        @test stderror(fp) ≈ stderror(fit) rtol = 1e-5
    end

    @testset "43i. inputs are checked" begin
        @test_throws ArgumentError fit_common_shock_dynamic_panel(yf, Xf, Wf, Nf, Tf + 1; r = 5)
        @test_throws ArgumentError fit_common_shock_dynamic_panel(yf, Xf, Wf, Nf, Tf + 1; r = 2, L = -1)
        @test_throws DimensionMismatch fit_common_shock_dynamic_panel(yf[1:end-1], Xf, Wf, Nf, Tf + 1; r = 2)
        @test_throws ArgumentError fit_common_shock_dynamic_panel(yf, Xf, Wf, Nf, Tf + 1; r = 2,
                                                                 controls = hcat(ones(Nf), randn(Nf)))
        Zdup = randn(MersenneTwister(1), Nf)
        @test_throws ArgumentError fit_common_shock_dynamic_panel(yf, Xf, Wf, Nf, Tf + 1; r = 2,
                                                                 controls = hcat(Zdup, 2 .* Zdup))
    end
end
