@testset "6. GRAS / Cross-Entropy Calibration" begin
    _progress("6. GRAS / Cross-Entropy Calibration")
    rng = MersenneTwister(606)
    M_dim = 5; N_dim = 4
    X0 = randn(rng, M_dim, N_dim) .* 10
    u_true = vec(sum(X0; dims=2)); v_true = vec(sum(X0; dims=1))
    u_target = u_true .* (1 .+ 0.1 .* randn(rng, M_dim))
    adj = (sum(u_target) - sum(v_true)) / N_dim
    v_target = v_true .+ adj

    res = balance_gras_full(X0, u_target, v_target)
    @test res.converged
    @test isapprox(vec(sum(res.X_balanced; dims=2)), u_target; atol=1e-6)
    @test isapprox(vec(sum(res.X_balanced; dims=1)), v_target; atol=1e-6)
    @test res.kl_divergence >= 0
end
