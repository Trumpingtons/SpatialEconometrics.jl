@testset "42. Short-T CRE spatial dynamic panel (Li & Yang 2021)" begin
    _progress("42. Short-T CRE spatial dynamic panel (Li & Yang 2021)")
    CM = SpatialEconometrics.CREDynamicSpatialPanel
    # Queen W1 = W3 on an L×L grid; W2 group interaction in blocks of L.
    function cre_W(L)
        n = L^2
        W1 = Matrix(make_queen_weights(L))
        W2 = zeros(n, n)
        for b in 0:(L - 1), i in 1:L, j in 1:L
            i == j && continue
            W2[b*L+i, b*L+j] = 1 / (L - 1)
        end
        return W1, W2
    end
    # The linear maps of the full shock vector u = (ε; v_{−m+1}; …; v_T),
    # with y_{−m} = 0: Y = c_Y + L_Y u, Y₋₁ = c_Y₋₁ + L_Y₋₁ u, e = L_e u.
    function cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, m)
        n = size(W1, 1)
        B1inv = inv(I - λ1 * W1); 𝓑 = B1inv * (ρ * I + λ2 * W2); B3inv = inv(I - λ3 * W3)
        ns = n + n * (T + m)
        vblk(t) = n + (t + m - 1) * n + 1:n + (t + m) * n
        Ly = Dict{Int, Matrix{Float64}}(-m => zeros(n, ns))
        for t in (-m + 1):T
            Lt = 𝓑 * Ly[t - 1]
            Lt[:, 1:n] .+= B1inv
            Lt[:, vblk(t)] .+= B1inv * B3inv
            Ly[t] = Lt
        end
        LY = reduce(vcat, [Ly[t] for t in 1:T]); LY1 = reduce(vcat, [Ly[t - 1] for t in 1:T])
        Le = zeros(n * T, ns)
        for t in 1:T
            r = (t - 1) * n + 1:t * n
            Le[r, 1:n] .= Matrix(I, n, n); Le[r, vblk(t)] .= B3inv
        end
        return LY, LY1, Le
    end
    # Every AQS component at ψ₀ as (A, a) with S = u'Au + a'u + const.
    function cre_forms(X, Oi, σ2, W1, W2, W3, λ3, T, LY, LY1, Le, cY, cY1)
        n = size(W1, 1); nT = n * T; ns = size(Le, 2)
        IT = Matrix(I, T, T)
        sym(A) = (A + A') / 2
        forms = Tuple{Matrix{Float64}, Vector{Float64}}[]
        for j in 1:size(X, 2)
            push!(forms, (zeros(ns, ns), Le' * (Oi * X[:, j]) ./ σ2))
        end
        push!(forms, (sym(Le' * Oi * Le) ./ (2σ2^2), zeros(ns)))
        J = kron(ones(T, T), Matrix(I, n, n))
        push!(forms, (sym(Le' * Oi * J * Oi * Le) ./ (2σ2), zeros(ns)))
        bilin(Mw, L, c) = (sym(Le' * Oi * Mw * L) ./ σ2, Le' * (Oi * (Mw * c)) ./ σ2)
        push!(forms, bilin(Matrix(I, nT, nT), LY1, cY1))
        push!(forms, bilin(kron(IT, W1), LY, cY))
        push!(forms, bilin(kron(IT, W2), LY1, cY1))
        B3 = I - λ3 * W3; Bi = inv(B3' * B3)
        G = kron(IT, Bi * (B3' * W3 + W3' * B3) * Bi)
        push!(forms, (sym(Le' * Oi * G * Oi * Le) ./ (2σ2), zeros(ns)))
        return forms
    end
    # Cov of linear-quadratic forms in independent shocks with variances
    # s2 and moments μ3, μ4 (A symmetric).
    function cre_exact_cov(forms, s2, μ3, μ4)
        p = length(forms); V = zeros(p, p); Σ = Diagonal(s2)
        for i in 1:p, j in 1:p
            A, a = forms[i]; B, b = forms[j]
            V[i, j] = 2tr(A * Σ * B * Σ) + sum(diag(A) .* diag(B) .* (μ4 .- 3 .* s2 .^ 2)) +
                      sum(diag(A) .* b .* μ3) + sum(diag(B) .* a .* μ3) + dot(a, Σ * b)
        end
        return V
    end
    # A fixed design (β = γ = π = 1) and its deterministic path.
    function cre_design(rng, n, T, m, W1, W2, ρ, λ1, λ2)
        Xs = [randn(rng, n) for t in -m:T]; Xbar = sum(Xs[m+1:end]) ./ (T + 1)
        z = Float64.(rand(rng, n) .< 0.5); α = randn(rng, T + m) .+ 1
        B1inv = inv(I - λ1 * W1); 𝓑 = B1inv * (ρ * I + λ2 * W2)
        yd = zeros(n); ys = Vector{Vector{Float64}}()
        for (j, t) in enumerate(-m+1:T)
            yd = 𝓑 * yd .+ B1inv * (Xs[j+1] .+ z .+ Xbar .+ α[j])
            t >= 0 && push!(ys, yd)
        end
        m == 0 && (ys = vcat([zeros(n)], ys))       # y_0 = y_{−m} = 0
        αT = α[m+1:m+T]
        ψ0(σv2, φ, λ3) = vcat(αT[end], αT[1:end-1] .- αT[end], 1.0, 1.0, 1.0, σv2, φ, ρ, λ1, λ2, λ3)
        return reshape(reduce(vcat, Xs[m+1:end]), :, 1), reshape(z, :, 1), ys, ψ0
    end
    # The data for shocks u: y_0..y_T stacked.
    cre_y(ys, LY, LY1, u, n) = vcat(ys[1] .+ LY1[1:n, :] * u, reduce(vcat, ys[2:end]) .+ LY * u)

    ρ, λ1, λ2, λ3, σv2, φ = 0.3, 0.2, 0.2, 0.2, 1.0, 1.0

    @testset "42a. Lemma 2.1 and the AQS traces, exactly" begin
        L = 3; n = L^2; T, m = 3, 2
        W1, W2 = cre_W(L); W3 = W1
        X, Z, ys, _ = cre_design(MersenneTwister(42_001), n, T, m, W1, W2, ρ, λ1, λ2)
        LY, LY1, Le = cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, m)
        d, _, _ = CM._cre_data(cre_y(ys, LY, LY1, zeros(size(Le, 2)), n), X, Z, W1, W2, W3,
                               n, T + 1, m, true, ["x"], ["z"])
        M = CM._mats(d, [φ, ρ, λ1, λ2, λ3])
        Cs, Ds = CM._cd_blocks(M, T, m)
        blk(t) = (t - 1) * n + 1:t * n
        C = zeros(n * T, n * T); D = zeros(n * T, n * T); C1 = zeros(n * T, n * T); D1 = zeros(n * T, n * T)
        for t in 1:T, s in 1:T
            C[blk(t), blk(s)] = Cs[t + 1]; C1[blk(t), blk(s)] = Cs[t]
            t - s >= 0 && (D[blk(t), blk(s)] = Ds[t - s + 1])
            t - 1 - s >= 0 && (D1[blk(t), blk(s)] = Ds[t - s])
        end
        Σu = Diagonal(vcat(fill(φ * σv2, n), fill(σv2, n * (T + m))))
        # E(Ye') and E(Y₋₁e') by the explicit maps, against σ²(φC + D).
        @test LY * Σu * Le' ≈ σv2 .* (φ .* C .+ D) atol = 1e-12
        @test LY1 * Σu * Le' ≈ σv2 .* (φ .* C1 .+ D1) atol = 1e-12
        IT = Matrix(I, T, T)
        Ω = φ .* kron(ones(T, T), Matrix(I, n, n)) .+ kron(IT, M.Binv3)
        Ωi = inv(Ω)
        full = [tr((φ .* C1 .+ D1) * Ωi), tr((φ .* C .+ D) * Ωi * kron(IT, W1)),
                tr((φ .* C1 .+ D1) * Ωi * kron(IT, W2))]
        @test CM._aqs_traces(d, M, φ) ≈ full rtol = 1e-12
        @test CM._aqs_traces_blocks(d, M, φ) ≈ full rtol = 1e-12
        # The closed forms of Ω⁻¹ and log|Ω|.
        E = randn(MersenneTwister(42_002), n, T)
        @test vec(CM._oinv(M, E)) ≈ Ω \ vec(E) rtol = 1e-12
        @test M.logdetΩ ≈ logdet(Ω) rtol = 1e-12
    end

    @testset "42b. m = 0: the AQS is the conditional score" begin
        L = 3; n = L^2; T = 3
        W1, W2 = cre_W(L); W3 = W1
        X, Z, ys, _ = cre_design(MersenneTwister(42_003), n, T, 0, W1, W2, ρ, λ1, λ2)
        y = vcat(ys...) .+ 0.3 .* randn(MersenneTwister(42_004), n * (T + 1))
        d, _, _ = CM._cre_data(y, X, Z, W1, W2, W3, n, T + 1, 0, true, ["x"], ["z"])
        ψ = vcat(randn(MersenneTwister(42_005), CM._k(d)), 1.2, 0.8, 0.3, 0.2, 0.15, 0.25)
        g = ForwardDiff.gradient(p -> CM._loglik(d, p), ψ)
        @test CM._cqs(d, ψ) ≈ g rtol = 1e-10
        @test CM._aqs(d, ψ) ≈ CM._cqs(d, ψ) atol = 1e-10 * norm(g)
        # With m > 0 they differ, in the ρ and λ1, λ2 components only.
        dm, _, _ = CM._cre_data(y, X, Z, W1, W2, W3, n, T + 1, 4, true, ["x"], ["z"])
        Δ = CM._aqs(dm, ψ) .- CM._cqs(dm, ψ)
        i = CM._iρ(dm)
        @test all(abs.(Δ[1:i-1]) .< 1e-12) && abs(Δ[end]) < 1e-12
        @test all(abs.(Δ[i:i+2]) .> 1e-3)
        # The central-difference Jacobian used for Σ̂, against ForwardDiff.
        Jf = ForwardDiff.jacobian(p -> CM._aqs(dm, p), ψ)
        @test CM._jac_central(p -> CM._aqs(dm, p), ψ) ≈ Jf rtol = 1e-6
    end

    # 42c: the corrected plug-in Γ̂ at ψ₀ with the true moments is unbiased
    # for the exact Var(S*(ψ₀)) over all the shocks, the pre-sample ones
    # included; the moment estimators at ψ₀ are unbiased; the AQS has mean
    # zero and the conditional score does not. A seeded Monte Carlo, judged
    # by its own standard errors.
    @testset "42c. Γ̂ is unbiased for the exact Var(S*), $errs errors" for errs in (:normal, :chi2)
        L = 4; n = L^2; T, m = 3, 2
        W1, W2 = cre_W(L); W3 = W1
        rng = MersenneTwister(errs === :normal ? 42_006 : 42_007)
        X, Z, ys, ψ0f = cre_design(rng, n, T, m, W1, W2, ρ, λ1, λ2)
        LY, LY1, Le = cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, m)
        d, _, _ = CM._cre_data(cre_y(ys, LY, LY1, zeros(size(Le, 2)), n), X, Z, W1, W2, W3,
                               n, T + 1, m, true, ["x"], ["z"])
        ψ0 = ψ0f(σv2, φ, λ3)
        M = CM._mats(d, ψ0[end-4:end])
        Oi = inv(φ .* kron(ones(T, T), Matrix(I, n, n)) .+ kron(Matrix(I, T, T), M.Binv3))
        forms = cre_forms(d.X, Oi, σv2, W1, W2, W3, λ3, T, LY, LY1, Le,
                          reduce(vcat, ys[2:end]), reduce(vcat, ys[1:end-1]))
        s2 = vcat(fill(φ * σv2, n), fill(σv2, n * (T + m)))
        sk, ku = errs === :normal ? (0.0, 3.0) : (2sqrt(2), 15.0)
        μ3 = sk .* s2 .^ 1.5; μ4 = ku .* s2 .^ 2
        Vex = cre_exact_cov(forms, s2, μ3, μ4)
        tm = (μ3ε = μ3[1], μ4ε = μ4[1], μ3v = μ3[end], μ4v = μ4[end])
        draw(k) = errs === :normal ? randn(rng, k) : (randn(rng, k) .^ 2 .- 1) ./ sqrt(2)
        R = 2000; p = length(ψ0)
        G = zeros(R, p, p); G0 = zeros(R, p, p); Mo = zeros(R, 4); S = zeros(R, p); Sc = zeros(R, p)
        for r in 1:R
            u = sqrt.(s2) .* draw(length(s2))
            dr, _, _ = CM._cre_data(cre_y(ys, LY, LY1, u, n), X, Z, W1, W2, W3, n, T + 1, m, true, ["x"], ["z"])
            G[r, :, :] = CM._plugin_gamma(dr, ψ0, tm)
            G0[r, :, :] = CM._plugin_gamma(dr, ψ0, tm; correct = false)
            mo = CM._shock_moments(dr, ψ0); Mo[r, :] = [mo.μ3ε, mo.μ4ε, mo.μ3v, mo.μ4v]
            S[r, :] = CM._aqs(dr, ψ0); Sc[r, :] = CM._cqs(dr, ψ0)
        end
        zof(A, target) = (dropdims(mean(A; dims = 1); dims = 1) .- target) ./
                         (dropdims(std(A; dims = 1); dims = 1) ./ sqrt(R))
        # Entries whose Γ̂ does not vary over draws (the β block: β's score
        # is linear, so its conditional variance is fixed) have no Monte
        # Carlo error, so they are compared exactly instead of by z.
        scale = sqrt.(diag(Vex) * diag(Vex)')
        fixed = dropdims(std(G; dims = 1); dims = 1) .< 1e-9 .* scale
        @test any(fixed) && !all(fixed)
        G1 = G[1, :, :]
        @test maximum(abs.(G1[fixed] .- Vex[fixed]) ./ scale[fixed]) < 1e-8
        @test maximum(abs.(zof(G, Vex)[.!fixed])) < 4.0
        relG = abs.(dropdims(mean(G; dims = 1); dims = 1) .- Vex) ./ scale
        @test maximum(relG) < 0.01
        # Without the correction for ε̂ the plug-in is biased: the
        # correction moves the mean of Γ̂ by much more than the bias that
        # remains with it. Paired on the same draws, so nearly noise-free.
        relD = abs.(dropdims(mean(G0 .- G; dims = 1); dims = 1)) ./ scale
        @test maximum(relD) > 3 * maximum(relG)
        @test maximum(abs.(zof(Mo, [tm...]))) < 4.0
        @test maximum(abs.(zof(S, zeros(p)))) < 4.0
        @test abs(zof(Sc, zeros(p))[CM._iρ(d)]) > 6       # the CQS's ρ component is biased
        # The exact reference itself: E S = 0, so E(SᵢSⱼ) = Vex, each entry
        # judged by its own Monte Carlo SE (a fixed relative tolerance is
        # not, with χ² errors the products are heavy-tailed).
        SS = reshape(S, R, p, 1) .* reshape(S, R, 1, p)
        @test maximum(abs.(zof(SS, Vex))) < 4.5
    end

    @testset "42d. the T = 2 branch of the μ3v estimator" begin
        L = 4; n = L^2; T, m = 2, 2
        W1, W2 = cre_W(L); W3 = W1
        rng = MersenneTwister(42_008)
        X, Z, ys, ψ0f = cre_design(rng, n, T, m, W1, W2, ρ, λ1, λ2)
        LY, LY1, Le = cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, m)
        ψ0 = ψ0f(σv2, φ, λ3)
        ns = size(Le, 2); R = 3000
        Mo = zeros(R, 4)
        for r in 1:R
            u = (randn(rng, ns) .^ 2 .- 1) ./ sqrt(2)
            dr, _, _ = CM._cre_data(cre_y(ys, LY, LY1, u, n), X, Z, W1, W2, W3, n, T + 1, m, true, ["x"], ["z"])
            mo = CM._shock_moments(dr, ψ0); Mo[r, :] = [mo.μ3ε, mo.μ4ε, mo.μ3v, mo.μ4v]
        end
        z = (vec(mean(Mo; dims = 1)) .- [2sqrt(2), 15, 2sqrt(2), 15]) ./ (vec(std(Mo; dims = 1)) ./ sqrt(R))
        @test maximum(abs.(z)) < 4.0
    end

    @testset "42e. φ = ∞ is a false root of S_φ, not of φ·S_φ" begin
        L = 3; n = L^2; T, m = 3, 2
        W1, W2 = cre_W(L); W3 = W1
        X, Z, ys, _ = cre_design(MersenneTwister(42_009), n, T, m, W1, W2, ρ, λ1, λ2)
        y = vcat(ys...) .+ randn(MersenneTwister(42_010), n * (T + 1))
        d, _, _ = CM._cre_data(y, X, Z, W1, W2, W3, n, T + 1, m, true, ["x"], ["z"])
        for φbig in (1e6, 1e8)
            s = CM._aqs_c(d, [φbig, ρ, λ1, λ2, λ3])
            @test abs(s[1]) < 1e-4 * n
            @test φbig * s[1] ≈ -n / 2 rtol = 1e-3
        end
    end

    @testset "42f. the fit" begin
        L = 10; n = L^2; T, m = 3, 10
        W1, W2 = cre_W(L); W3 = W1
        rng = MersenneTwister(42_011)
        X, Z, ys, _ = cre_design(rng, n, T, m, W1, W2, ρ, λ1, λ2)
        # X_0..X_T as the fit expects: cre_design returns X_0..X_T already.
        LY, LY1, Le = cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, m)
        y = cre_y(ys, LY, LY1, randn(rng, size(Le, 2)), n)
        f = fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = m, W2 = W2, W3 = W3,
                                          varnames = ["x"], zvarnames = ["z"], compute_cqmle = true)
        @test f.converged && f.score_norm < 1e-10
        nm = coefnames(f)
        @test nm[end-6:end] == ["mean_x", "sigma2_v", "phi", "rho", "lambda1", "lambda2", "lambda3"]
        b = Dict(zip(nm, coef(f)))
        # The root the fit reaches from its IV start is the one the solver
        # reaches from the true δ. (A fixed band around the truth is not a
        # test here: at n = 100 the SE of ρ̂ is about .06, and on Julia
        # 1.10's stream this draw puts ρ̂ 2.1 SEs from .3, mean_x 3.4.)
        dm = CM._cre_data(y, X, Z, W1, W2, W3, n, T + 1, m, true, ["x"], ["z"])[1]
        δt, okt, _, _ = CM._solve_aqs(dm, [φ, ρ, λ1, λ2, λ3])
        @test okt && maximum(abs.(δt .- coef(f)[end-4:end])) < 1e-6
        @test abs(b["rho"] - ρ) < 4 * stderror(f)[findfirst(==("rho"), nm)]
        @test issymmetric(vcov(f)) && isposdef(vcov(f))
        @test stderror(f) ≈ sqrt.(diag(vcov(f)))
        @test vcov(f) != f.vcov_hessian
        @test length(f.cqmle) == length(coef(f)) && f.cqmle != coef(f)
        @test nobs(f) == n * T
        @test length(residuals(f)) == n * T
        s = sprint(show, f)
        @test occursin("process start m = 10", s) && occursin("sandwich", s)
        @test !occursin("time_1", s)
        # Without W2 and W3 the equations for λ2, λ3 are dropped.
        f2 = fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = m, W2 = nothing, W3 = nothing)
        @test coefnames(f2)[end-2:end] == ["phi", "rho", "lambda1"]
        @test f2.converged
        # m = 0: the M-estimator is the conditional QMLE.
        X0, Z0, ys0, _ = cre_design(rng, n, T, 0, W1, W2, ρ, λ1, λ2)
        LY0, LY10, Le0 = cre_full_maps(W1, W2, W3, ρ, λ1, λ2, λ3, T, 0)
        y0 = cre_y(ys0, LY0, LY10, randn(rng, size(Le0, 2)), n)
        f0 = fit_cre_dynamic_spatial_panel(y0, X0, Z0, W1, n, T + 1; m = 0, W2 = W2, W3 = W3,
                                           compute_cqmle = true)
        @test coef(f0) ≈ f0.cqmle rtol = 1e-6
    end

    @testset "42g. argument checks" begin
        L = 3; n = L^2; T = 3
        W1, W2 = cre_W(L)
        X, Z, ys, _ = cre_design(MersenneTwister(42_012), n, T, 2, W1, W2, ρ, λ1, λ2)
        y = vcat(ys...)
        @test_throws UndefKeywordError fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1)
        @test_throws ArgumentError fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = -1)
        @test_throws ArgumentError fit_cre_dynamic_spatial_panel(y[1:2n], X[1:2n, :], Z, W1, n, 2; m = 1)
        @test_throws DimensionMismatch fit_cre_dynamic_spatial_panel(y[1:end-1], X, Z, W1, n, T + 1; m = 1)
        # A time-invariant column of X duplicates its own mean.
        Xc = repeat(randn(MersenneTwister(42_013), n), T + 1)
        @test_throws ArgumentError fit_cre_dynamic_spatial_panel(y, hcat(X, Xc), Z, W1, n, T + 1; m = 1)
    end
end
