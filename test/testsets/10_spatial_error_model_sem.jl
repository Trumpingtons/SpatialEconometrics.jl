@testset "10. Spatial Error Model (SEM)" begin
    _progress("10. Spatial Error Model (SEM)")
    rng = MersenneTwister(1010); W = make_rook_weights(15); N = size(W, 1)
    X = randn(rng, N, 2); β_true = [1.0, 2.0, -1.0]; λ_true = 0.55
    u = (I - λ_true * W) \ (0.5 .* randn(rng, N))
    y = β_true[1] .+ X * β_true[2:3] .+ u

    m_sem = fit_spatial_error(W, X, y)
    @test isapprox(m_sem.lambda, λ_true; atol=0.12)
    @test length(coef(m_sem)) == 4  # intercept + 2β + λ
    @test coefnames(m_sem) == ["(Intercept)", "x1", "x2", "λ"]

    # Panel SEM
    T_per = 6; NT = N * T_per
    X_pan = randn(rng, NT, 2); μ_i = randn(rng, N)
    y_pan = zeros(NT)
    B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        u_t = B_lu \ (0.4 .* randn(rng, N))
        y_pan[idx] = μ_i .+ X_pan[idx, :] * [1.5, -0.7] .+ u_t
    end
    m_sem_pan = fit_spatial_error(IndivFE(), W, X_pan, y_pan, N, T_per)
    @test isapprox(m_sem_pan.lambda, λ_true; atol=0.15)
    @test coefnames(m_sem_pan) == ["x1", "x2", "λ"]

    @testset "10a. heteroskedasticity-robust GM (estimator = :gmm)" begin
        SEMm = SpatialEconometrics.SpatialErrorModel
        side = 12; Ng = side^2
        Wg = row_standardize(make_rook_weights(side))
        rngg = MersenneTwister(1011)
        WtW = Wg' * Wg
        ddg = Vector(diag(WtW))
        A1 = WtW - spdiagm(0 => diag(WtW))
        S1 = A1 + A1'; S2 = Wg + Wg'

        # The sample moments: g − G[λ, λ²] is n⁻¹ε(λ)′Aᵣε(λ) with
        # ε(λ) = u − λWu, for every λ, in a cross-section and over periods.
        for Tg in (1, 3)
            u = randn(rngg, Ng * Tg)
            g, G = SEMm._gm_moments(u, Wg, Ng, Tg, repeat(ddg, Tg))
            for λ in (-0.7, 0.2, 0.9)
                q1 = 0.0; q2 = 0.0
                for t in 1:Tg
                    e = u[(t - 1) * Ng + 1:t * Ng] .- λ .* (Wg * u[(t - 1) * Ng + 1:t * Ng])
                    q1 += dot(e, A1 * e); q2 += dot(e, Wg * e)
                end
                @test SEMm._gm_m(g, G, λ) ≈ [q1, q2] ./ (Ng * Tg) atol = 1e-13
            end
        end

        # Ψ against the exact covariance of the quadratic forms. With
        # zero-diagonal Sᵣ and independent ε of variances σᵢ²,
        # Cov(½ε′Sᵣε, ½ε′Sₛε) = ½ tr(SᵣΣSₛΣ), whatever the third and
        # fourth moments. `_gm_psi` estimates σᵢ²σⱼ² by eᵢ²eⱼ², so fed
        # e = σ it must return that covariance over n exactly.
        # The variance follows dᵢ = (W′W)ᵢᵢ, as Arraiz et al.'s follows
        # the neighbour count: the 1998 moment's mean is Σᵢ σᵢ²(dᵢ − c),
        # so this is a design it is biased in.
        σg = [d > median(ddg) ? 3.0 : 1.0 for d in ddg]
        Σg = Diagonal(σg .^ 2)
        Sd1, Sd2 = Matrix(S1), Matrix(S2)
        Ψexact = [tr(Sd1 * Σg * Sd1 * Σg) tr(Sd1 * Σg * Sd2 * Σg);
                  tr(Sd2 * Σg * Sd1 * Σg) tr(Sd2 * Σg * Sd2 * Σg)] ./ (2Ng)
        @test SEMm._gm_psi(σg, S1, S2, Ng, 1) ≈ Ψexact rtol = 1e-12

        # The null distribution, simulated: at the true λ the moments
        # are n⁻¹ε′Aᵣε. Heteroskedastic, skewed ε (centred χ²₃). Their
        # mean must be zero and the covariance of √n times them Ψexact.
        # Power: the Kelejian–Prucha (1998) moment ε′(W′W −
        # (tr W′W/n)I)ε/n, which assumes equal variances, has a mean
        # away from zero here.
        nsim = 4000
        M = zeros(nsim, 2); m98 = zeros(nsim)
        c98 = tr(WtW) / Ng
        for r in 1:nsim
            z = (rand(rngg, Chisq(3), Ng) .- 3) ./ sqrt(6)
            e = σg .* z
            M[r, 1] = dot(e, A1 * e) / Ng; M[r, 2] = dot(e, Wg * e) / Ng
            m98[r] = (dot(Wg * e, Wg * e) - c98 * dot(e, e)) / Ng
        end
        zmean = mean(M, dims = 1) ./ (std(M, dims = 1) ./ sqrt(nsim))
        @test all(abs.(zmean) .< 4)
        C = cov(sqrt(Ng) .* M)
        for (a, b) in ((1, 1), (2, 2), (1, 2))
            x = Ng .* (M[:, a] .- mean(M[:, a])) .* (M[:, b] .- mean(M[:, b]))
            @test abs(C[a, b] - Ψexact[a, b]) < 4 * std(x) / sqrt(nsim)
        end
        @test abs(mean(m98)) / (std(m98) / sqrt(nsim)) > 8

        # Over periods, after the unit demeaning a fixed-effects fit
        # applies: Cov(eᵢₜ, eᵢₛ) = σᵢ²(δₜₛ − 1/T), so the pooled moment's
        # covariance is (2nT)⁻¹ Σᵢⱼ S²ᵢⱼ σᵢ²σⱼ² (T − 1), which is what
        # `_gm_psi`'s (eᵢ·eⱼ)² estimates. Power: summing per-period
        # covariances, as if periods were independent, gives
        # T(1 − 1/T)² in place of T − 1, which is a factor (T−1)/T off.
        Tg = 4; nsimp = 3000
        Mp = zeros(nsimp, 2)
        ddp = repeat(ddg, Tg)
        for r in 1:nsimp
            e = vec(σg .* randn(rngg, Ng, Tg))
            Ep = reshape(e, Ng, Tg); Ep .-= mean(Ep, dims = 2)
            ev = vec(Ep)
            g, G = SEMm._gm_moments(ev, Wg, Ng, Tg, ddp)
            Mp[r, :] .= g
        end
        Ψp = Ψexact .* (Tg - 1) ./ Tg
        Ψnaive = Ψexact .* (Tg * (1 - 1 / Tg)^2) ./ Tg
        Cp = cov(sqrt(Ng * Tg) .* Mp)
        for (a, b) in ((1, 1), (2, 2))
            x = (Ng * Tg) .* (Mp[:, a] .- mean(Mp[:, a])) .^ 2
            se = std(x) / sqrt(nsimp)
            @test abs(Cp[a, b] - Ψp[a, b]) < 4se
            @test abs(Cp[a, b] - Ψnaive[a, b]) > 4se
        end

        # The estimator: its steps, by hand. β̂ is OLS on the data
        # filtered at the step-1 λ, and λ̂ minimises the step-2
        # objective.
        Xg = hcat(ones(Ng), randn(rngg, Ng, 2))
        Ag = lu(sparse(1.0I, Ng, Ng) - 0.5 * Wg)
        yg = Xg * [1.0, 2.0, -1.0] .+ Ag \ (σg .* randn(rngg, Ng))
        o = SEMm._sem_gmm_het(Wg, Xg, yg, Ng, 1)
        λ1 = o.lambda_initial
        @test o.beta ≈ (Xg .- λ1 .* (Wg * Xg)) \ (yg .- λ1 .* (Wg * yg)) rtol = 1e-12
        û = yg .- Xg * o.beta
        g2, G2 = SEMm._gm_moments(û, Wg, Ng, 1, ddg)
        Ψ2 = SEMm._gm_psi(û .- λ1 .* (Wg * û), S1, S2, Ng, 1)
        f(λ) = (mm = SEMm._gm_m(g2, G2, λ); dot(mm, Ψ2 \ mm))
        @test f(o.lambda) <= min(f(o.lambda - 1e-3), f(o.lambda + 1e-3))
        # HC3: (X′X)⁻¹ X′ diag(ẽ²) X (X′X)⁻¹ with ẽᵢ = eᵢ/(1 − hᵢᵢ),
        # times (n−1)/n.
        Xs = o.X_filtered; B = inv(Xs' * Xs)
        h = vec(sum((Xs * B) .* Xs, dims = 2))
        V3 = B * (Xs' * ((o.residuals ./ (1 .- h)) .^ 2 .* Xs)) * B .* ((Ng - 1) / Ng)
        @test o.vcov_beta ≈ V3 rtol = 1e-10
        @test o.lambda_se ≈ sqrt(inv(dot(o.J, o.Psi \ o.J)) / Ng) rtol = 1e-12

        # The public route returns the same fit.
        mg = fit_spatial_error(Wg, Xg[:, 2:3], yg; estimator = :gmm)
        @test coef(mg) == vcat(o.beta, o.lambda)
        @test vcov(mg)[1:3, 1:3] == o.vcov_beta && vcov(mg)[4, 4] == o.lambda_se^2
        @test isnan(mg.loglik)
        @test occursin("GMM, robust to heteroskedasticity", sprint(show, mg))
        @test !occursin("Log-lik", sprint(show, mg))
        @test_throws ArgumentError fit_spatial_error(Wg, Xg[:, 2:3], yg; estimator = :bogus)
        Wd = copy(Wg); Wd[1, 1] = 0.1
        @test_throws ArgumentError fit_spatial_error(Wd, Xg[:, 2:3], yg; estimator = :gmm)
    end
end
