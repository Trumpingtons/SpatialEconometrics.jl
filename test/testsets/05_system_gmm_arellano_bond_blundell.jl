@testset "5. System GMM (Arellano-Bond / Blundell-Bond)" begin
    _progress("5. System GMM (Arellano-Bond / Blundell-Bond)")
    rng = MersenneTwister(505); N = 80; T_tot = 7; K = 1
    γ_true = 0.4; β_true = [1.0]
    μ_i = randn(rng, N); Y = zeros(N, T_tot); X_ten = randn(rng, N, T_tot, K)
    for i in 1:N; Y[i, 1] = μ_i[i] + X_ten[i, 1, 1] * β_true[1] + randn(rng); end
    for t in 2:T_tot, i in 1:N
        Y[i, t] = μ_i[i] + γ_true * Y[i, t - 1] + X_ten[i, t, 1] * β_true[1] + 0.5 * randn(rng)
    end

    m_sys = fit_dynamic_gmm(SystemGMM(), Y, X_ten; max_lag=3)
    @test length(coef(m_sys)) == 2  # y_lag, x1
    @test m_sys.hansen_df >= 0
    @test coefnames(m_sys) == ["y_lag", "x1"]

    m_diff = fit_dynamic_gmm(DifferenceGMM(), Y, X_ten; max_lag=3)
    @test length(coef(m_diff)) == 2
end
