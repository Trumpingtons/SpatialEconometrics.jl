@testset "18. Spatial Tables & Output" begin
    _progress("18. Spatial Tables & Output")
    rng = MersenneTwister(1800); W = make_rook_weights(10); N = size(W, 1)
    X = randn(rng, N, 2)
    y = (I - 0.4 * W) \ (fill(1.0, N) .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))

    m = fit_spatial(SAR(), W, X, y)

    # CoefTable
    ct = coeftable(m)
    @test length(ct.names) == length(coef(m))
    @test length(ct.estimate) == length(coef(m))
    @test all(isfinite, ct.z_stat)
    @test all(p -> 0.0 <= p <= 1.0, ct.p_value)
    @test all(ct.ci_lower .<= ct.ci_upper)

    # Tables.jl interface
    @test Tables.istable(typeof(ct))
    @test :name in Tables.columnnames(ct)
    @test :estimate in Tables.columnnames(ct)
    col = Tables.getcolumn(ct, :estimate)
    @test length(col) == length(coef(m))

    # Export formats
    latex = to_latex(ct)
    @test occursin("\\begin{tabular}", latex)
    @test occursin("\\end{tabular}", latex)

    csv = to_csv(ct)
    @test occursin("name,estimate,std_error", csv)
    lines = split(strip(csv), '\n')
    @test length(lines) == length(coef(m)) + 1  # header + rows

    md = to_markdown(ct)
    @test occursin("| Variable |", md)

    # Summary stats
    ss = summary_stats(m; model_type="SAR")
    @test ss.n == N
    @test ss.k == length(coef(m))
    @test isfinite(ss.aic)
    @test isfinite(ss.bic)
    @test 0.0 <= ss.r2 <= 1.0

    # Multi-model regression table
    m_sem = fit_spatial_error(W, X, y)
    rt = regression_table(m, m_sem; model_names=["SAR", "SEM"])
    @test occursin("SAR", rt)
    @test occursin("SEM", rt)
    @test occursin("N", rt)
    @test occursin("Log-Lik", rt)
end
