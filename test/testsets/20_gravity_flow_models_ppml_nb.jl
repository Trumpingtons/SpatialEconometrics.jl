@testset "20. Gravity / Flow Models (PPML & NB)" begin
    _progress("20. Gravity / Flow Models (PPML & NB)")
    rng = MersenneTwister(1212)
    N_orig = 15; N_dest = 12; N_obs = N_orig * N_dest
    origin_id = repeat(1:N_orig, inner=N_dest)
    dest_id = repeat(1:N_dest, outer=N_orig)
    dist = rand(rng, N_obs) .* 5 .+ 0.5
    gdp_orig = rand(rng, N_obs) .* 3 .+ 1
    X = hcat(log.(dist), log.(gdp_orig))
    β_true = [-1.2, 0.8]

    # True origin/dest FE
    α_o = randn(rng, N_orig) .* 0.5
    α_d = randn(rng, N_dest) .* 0.5
    η = X * β_true
    for n in 1:N_obs
        η[n] += α_o[origin_id[n]] + α_d[dest_id[n]]
    end
    μ_true = exp.(η)
    flows = Float64[max(0.0, rand(rng) < 0.95 ? Float64(round(μ_true[n] + sqrt(μ_true[n]) * randn(rng))) : 0.0) for n in 1:N_obs]
    flows = max.(flows, 0.0)

    m_ppml = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id; varnames=["log_dist", "log_gdp"], maxiter=200)
    @test m_ppml.iterations > 0
    @test length(coef(m_ppml)) == 2
    @test coefnames(m_ppml) == ["log_dist", "log_gdp"]
    @test coef(m_ppml)[1] < 0  # distance should be negative

    m_nb = fit_gravity(NegBinGravity(), flows, X, origin_id, dest_id; varnames=["log_dist", "log_gdp"])
    @test m_nb.alpha > 0.0
    @test length(coef(m_nb)) == 2
end
