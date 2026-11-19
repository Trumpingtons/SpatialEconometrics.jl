@testset "25. Makie Extension" begin
    _progress("25. Makie Extension")
    if !_HAVE_MAKIE
        @info "Makie is not installed: skipping the Makie extension tests."
        @test_skip false
    else
        @eval using Makie
        @test !isnothing(Base.get_extension(SpatialEconometrics,
                                            :SpatialEconometricsMakieExt))

        rng = MersenneTwister(5); L = 8; N = L * L
        W = make_rook_weights(L)
        coords = hcat(repeat(1.0:L, outer = L), repeat(1.0:L, inner = L))
        X = randn(rng, N, 2)
        y = (I - 0.5 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))
        m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])

        @test plot_coefficients(m) isa Makie.Figure
        @test plot_moran_scatter(y, W) isa Makie.Figure
        @test plot_spatial_effects(spatial_effects(m)) isa Makie.Figure

        lisa = moran_local(y, W; n_perm = 99, rng = MersenneTwister(1))
        @test plot_local_moran(lisa, coords) isa Makie.Figure

        g = fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(20),
                    varnames = ["income", "density"])
        @test plot_gwr_coefficients(g, "income") isa Makie.Figure

        # `level` must actually widen the interval that gets drawn.
        ci_span(f) = (seg = f.content[1].scene.plots[1][1][];
                      abs(seg[2][1] - seg[1][1]))
        @test ci_span(plot_coefficients(m; level = 0.99)) >
              ci_span(plot_coefficients(m; level = 0.95))

        @test_throws ArgumentError plot_coefficients(m; level = 0.0)
        @test_throws ArgumentError plot_coefficients(m; level = 1.5)
        @test_throws ErrorException plot_gwr_coefficients(g, "not_a_variable")
        @test_throws DimensionMismatch plot_local_moran(lisa, coords[1:end-1, :])
    end
end
