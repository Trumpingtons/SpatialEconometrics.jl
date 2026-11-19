@testset "24. Extension Entry Points" begin
    _progress("24. Extension Entry Points")
    # The extension functions are declared by the parent, so
    # `using SpatialEconometrics` alone makes them reachable. An
    # extension cannot usefully export anything of its own: nothing ever
    # `using`s it, so a function it declared itself would be invisible.
    for f in (:plot_coefficients, :plot_moran_scatter, :plot_local_moran,
              :plot_gwr_coefficients, :plot_spatial_effects,
              :bayesian_sar, :bayesian_sdm, :bayesian_sem)
        @test isdefined(SpatialEconometrics, f)
        @test Base.isexported(SpatialEconometrics, f)
    end
    @test isdefined(SpatialEconometrics, :AbstractBayesianSpatialResult)
    @test isabstracttype(AbstractBayesianSpatialResult)

    # Calling one without its weak dependency must say which package to
    # load, rather than raise a bare MethodError.
    for (f, pkg) in ((plot_coefficients, "Makie"), (plot_moran_scatter, "Makie"),
                     (bayesian_sar, "Turing"), (bayesian_sem, "Turing"))
        err = try; f(nothing); catch e; e; end
        @test err isa ErrorException
        @test occursin(pkg, err.msg)
    end
end
