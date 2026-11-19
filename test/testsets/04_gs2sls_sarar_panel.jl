@testset "4. GS2SLS / SARAR Panel" begin
    _progress("4. GS2SLS / SARAR Panel")
    rng = MersenneTwister(404); L = 10; N = L * L; T_tot = 6; NT = N * T_tot; W = make_rook_weights(L)
    X_exog = randn(rng, NT, 2); β_true = [1.2, -0.6]; ρ_true = 0.3; λ_true = 0.25
    z1 = randn(rng, NT); z2 = randn(rng, NT)
    endog = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rng, NT); δ_true = 0.8

    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
    y = zeros(NT)
    for t in 1:T_tot
        idx = ((t - 1) * N + 1):(t * N)
        u_t = B_lu \ (0.4 .* randn(rng, N))
        y[idx] = S_lu \ (X_exog[idx, :] * β_true .+ endog[idx] .* δ_true .+ u_t)
    end

    m_gs = fit_gs2sls_panel(IndivFE(), W, X_exog, reshape(endog, NT, 1), hcat(z1, z2), y, N, T_tot)
    @test length(coef(m_gs)) == 5  # ρ, exog1, exog2, endog1, λ
    @test m_gs.T_post == T_tot - 1
    @test size(vcov(m_gs)) == (5, 5)
end
