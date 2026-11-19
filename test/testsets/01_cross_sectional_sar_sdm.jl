@testset "1. Cross-Sectional SAR & SDM" begin
    _progress("1. Cross-Sectional SAR & SDM")
    rng = MersenneTwister(101); W = make_rook_weights(20); N = size(W, 1)
    X = randn(rng, N, 2); β_true = [2.0, 1.5, -1.0]; θ_true = [0.8, -0.5]; ρ_true = 0.55; σ_true = 0.50
    y = (I - ρ_true * W) \ (β_true[1] .+ X * β_true[2:3] .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
    m_cheb = fit_spatial(SDM(), W, X, y; method=ChebyshevApprox(order=7))
    @test isapprox(m_cheb.rho, ρ_true; atol=0.08)
    @test length(coef(m_cheb)) == 6  # intercept + 2 β + 2 θ + ρ
    @test length(residuals(m_cheb)) == N

    m_sar = fit_spatial(SAR(), W, X, y; method=ChebyshevApprox(order=7))
    @test length(coef(m_sar)) == 4  # intercept + 2 β + ρ
end
