@testset "8. FormulaDSL.jl (@formula, slag, fe, endo)" begin
    _progress("8. FormulaDSL.jl (@formula, slag, fe, endo)")
    rng = MersenneTwister(808); L = 15; N = L * L; T_per = 6; NT = N * T_per; W = make_rook_weights(L)
    X = randn(rng, NT, 2); β_true = [1.4, -0.9]; θ_true = [0.7, -0.35]; ρ_true = 0.45
    μ_i = randn(rng, N); ξ_t = randn(rng, T_per)

    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    y = zeros(Float64, NT); region_id = zeros(Int, NT); year_id = zeros(Int, NT)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        y[idx] = S_lu \ (μ_i .+ ξ_t[t] .+ X[idx, :] * β_true .+ (W * X[idx, :]) * θ_true .+ 0.5 .* randn(rng, N))
        region_id[idx] .= 1:N; year_id[idx]   .= 2020 + t
    end

    shuf_idx = sortperm(1:NT; by = i -> (region_id[i], year_id[i]))
    tbl = (region = region_id[shuf_idx], year = year_id[shuf_idx], y = y[shuf_idx], x1 = X[shuf_idx, 1], x2 = X[shuf_idx, 2])

    m_dsl = fit_spatial_panel(
        @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
        tbl, W; panel=:region, time=:year, approach=LeeYuTransformation()
    )
    @test coefnames(m_dsl) == ["x1", "x2", "W_x1", "W_x2", "ρ"]

    z1 = randn(rng, NT); z2 = randn(rng, NT); tax_endog = 0.6 .* z1 .- 0.4 .* z2 .+ 0.3 .* randn(rng, NT)
    tbl_iv = (region = region_id[shuf_idx], year = year_id[shuf_idx], y = y[shuf_idx], x1 = X[shuf_idx, 1], tax = tax_endog[shuf_idx], z1 = z1[shuf_idx], z2 = z2[shuf_idx])

    m_iv_dsl = fit_gs2sls_panel(
        @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
        tbl_iv, W; panel=:region, time=:year
    )
    @test coefnames(m_iv_dsl) == ["ρ (Wy)", "x1", "tax", "λ (Wu)"]
    # The formula route passes `errors` through.
    m_iv_het = fit_gs2sls_panel(
        @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
        tbl_iv, W; panel=:region, time=:year, errors=:heteroskedastic
    )
    @test m_iv_het.errors === :heteroskedastic && m_iv_dsl.errors === :homoskedastic
    @test coef(m_iv_het)[1:3] != coef(m_iv_dsl)[1:3]
end
