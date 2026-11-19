@testset "26. Turing Extension" begin
    _progress("26. Turing Extension")
    if !_HAVE_TURING
        @info "Turing is not installed: skipping the Turing extension tests."
        @test_skip false
    else
        @eval using Turing, Distributions
        @test !isnothing(Base.get_extension(SpatialEconometrics,
                                            :SpatialEconometricsTuringExt))

        rng = MersenneTwister(2024); L = 10; N = L * L
        W = make_rook_weights(L)
        X = randn(rng, N, 2)
        β_true = [1.0, 1.5, -0.8]; ρ_true = 0.5; σ_true = 0.5
        Xa = hcat(ones(N), X)
        y = (I - ρ_true * W) \ (Xa * β_true .+ σ_true .* randn(rng, N))

        r = bayesian_sar(W, X, y; n_samples = 600, n_chains = 2,
                         varnames = ["x1", "x2"], rng = MersenneTwister(11))
        @test r isa AbstractBayesianSpatialResult
        @test r.coefnames == ["(Intercept)", "x1", "x2", "ρ", "σ"]
        @test length(r.posterior_mean) == 5 == length(r.posterior_std)
        @test all(isfinite, r.posterior_mean)
        @test all(>(0), r.posterior_std)
        @test r.N == N
        @test occursin("SAR", sprint(show, r))

        # The posterior has to concentrate on the truth. This is the test
        # that catches a mis-wired likelihood: a `~` whose left-hand side
        # is a local variable makes DynamicPPL sample it as one more
        # latent, and then every "posterior" comes back as its prior.
        @test isapprox(r.posterior_mean[2], β_true[2]; atol = 0.15)
        @test isapprox(r.posterior_mean[3], β_true[3]; atol = 0.15)
        @test isapprox(r.posterior_mean[4], ρ_true;    atol = 0.10)
        @test isapprox(r.posterior_mean[5], σ_true;    atol = 0.10)
        # The prior SD of β is sqrt(10) ≈ 3.16; the posterior is far tighter.
        @test all(r.posterior_std[1:3] .< 0.5)

        # ... and it has to agree with the MLE on the same data.
        mle = fit_spatial(SAR(), W, X, y)
        @test isapprox(r.posterior_mean[4], mle.rho; atol = 0.05)

        λ_true = 0.6
        y_sem = Xa * β_true .+ ((I - λ_true * W) \ (σ_true .* randn(rng, N)))
        r2 = bayesian_sem(W, X, y_sem; n_samples = 600, n_chains = 1,
                          varnames = ["x1", "x2"], rng = MersenneTwister(11))
        @test r2.coefnames == ["(Intercept)", "x1", "x2", "λ", "σ"]
        @test isapprox(r2.posterior_mean[4], λ_true; atol = 0.12)
        @test occursin("SEM", sprint(show, r2))

        θ_true = [0.6, -0.3]
        y_sdm = (I - ρ_true * W) \
                (Xa * β_true .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
        r3 = bayesian_sdm(W, X, y_sdm; n_samples = 600, n_chains = 1,
                          varnames = ["x1", "x2"], rng = MersenneTwister(11))
        @test r3.coefnames ==
              ["(Intercept)", "x1", "x2", "W_x1", "W_x2", "ρ", "σ"]
        @test isapprox(r3.posterior_mean[4], θ_true[1]; atol = 0.25)
        @test isapprox(r3.posterior_mean[5], θ_true[2]; atol = 0.25)

        # Default names when none are given.
        r4 = bayesian_sar(W, X, y; n_samples = 60, n_chains = 1,
                          rng = MersenneTwister(3))
        @test r4.coefnames == ["(Intercept)", "x1", "x2", "ρ", "σ"]

        # Same seed, same answer.
        r5 = bayesian_sar(W, X, y; n_samples = 60, n_chains = 1,
                          rng = MersenneTwister(3))
        @test r4.posterior_mean == r5.posterior_mean

        @test_throws DimensionMismatch bayesian_sar(W, X, y[1:end-1])
        @test_throws DimensionMismatch bayesian_sar(make_rook_weights(5), X, y)
    end
end
