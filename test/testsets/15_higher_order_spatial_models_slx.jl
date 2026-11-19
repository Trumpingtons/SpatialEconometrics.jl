@testset "15. Higher-Order Spatial Models (SLX, SAC, SDEM)" begin
    _progress("15. Higher-Order Spatial Models (SLX, SAC, SDEM)")
    rng = MersenneTwister(1500); W = make_rook_weights(15); N = size(W, 1)
    X = randn(rng, N, 2)

    # SLX: y = Xβ + WXθ + ε
    θ_true = [0.6, -0.3]
    y_slx = fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * θ_true .+ 0.4 .* randn(rng, N)
    m_slx = fit_slx(W, X, y_slx; varnames=["x1", "x2"])
    @test length(coef(m_slx)) == 5  # intercept + 2β + 2θ
    @test coefnames(m_slx) == ["(Intercept)", "x1", "x2", "W_x1", "W_x2"]
    @test length(residuals(m_slx)) == N
    @test isfinite(loglikelihood(m_slx))

    # SAC: y = ρW₁y + Xβ + u, u = λW₂u + ε
    ρ_true = 0.4; λ_true = 0.3
    IN = sparse(1.0I, N, N)
    ε_sac = 0.5 .* randn(rng, N)
    u_sac = (IN - λ_true * W) \ ε_sac
    y_sac = (IN - ρ_true * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ u_sac)
    m_sac = fit_sac(W, W, X, y_sac; varnames=["x1", "x2"])
    @test length(coef(m_sac)) == 5  # intercept + 2β + ρ + λ
    @test coefnames(m_sac) == ["(Intercept)", "x1", "x2", "ρ", "λ"]
    @test isfinite(loglikelihood(m_sac))
    @test nobs(m_sac) == N
    @test isapprox(m_sac.rho, ρ_true; atol=0.20)

    # SDEM: y = Xβ + WXθ + u, u = λWu + ε
    λ_sdem = 0.45
    u_sdem = (IN - λ_sdem * W) \ (0.4 .* randn(rng, N))
    y_sdem = fill(1.0, N) .+ X * [1.5, -0.8] .+ (W * X) * [0.5, -0.3] .+ u_sdem
    m_sdem = fit_sdem(W, X, y_sdem; varnames=["x1", "x2"])
    @test length(coef(m_sdem)) == 6  # intercept + 2β + 2θ + λ
    @test coefnames(m_sdem) == ["(Intercept)", "x1", "x2", "W_x1", "W_x2", "λ"]
    @test isfinite(loglikelihood(m_sdem))
    @test isapprox(m_sdem.lambda, λ_sdem; atol=0.20)

    # StatsAPI interface completeness
    for m in (m_slx, m_sac, m_sdem)
        @test length(stderror(m)) == length(coef(m))
        @test size(vcov(m)) == (length(coef(m)), length(coef(m)))
        @test dof_residual(m) == nobs(m) - length(coef(m))
    end
end
