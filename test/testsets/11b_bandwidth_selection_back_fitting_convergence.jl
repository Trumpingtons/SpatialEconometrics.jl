@testset "11b. Bandwidth selection & back-fitting convergence" begin
    _progress("11b. Bandwidth selection & back-fitting convergence")
    rng = MersenneTwister(2211); N = 45
    coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
    X = randn(rng, N, 2)
    y = 2.0 .+ (1.0 .+ 0.3 .* coords[:, 1]) .* X[:, 1] .-
        0.5 .* X[:, 2] .+ 0.5 .* randn(rng, N)

    # The adaptive bandwidth asks the KD-tree for k+1 neighbours, which
    # throws at k = N -- and k_max = min(N, 500) put exactly that value
    # in the search range. It survived only because the old golden-section
    # search never happened to evaluate the endpoint.
    @test fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(N)) isa GWRModel

    # With search = :scan and a span this short the selector scans every
    # candidate, so it must return the exact grid minimum. The default,
    # mgwr's golden-section search, is valid only on a unimodal objective;
    # the AICc curve in k is not unimodal (eleven sign changes on
    # Columbus), and there such a search returned k = 43 where the minimum
    # is k = 23, 2.3 AICc units worse.
    m_sel = fit_gwr(X, y, coords; search = :scan)
    @test 6 <= m_sel.bandwidth <= N
    @test all(k -> fit_gwr(X, y, coords;
                           bandwidth = AdaptiveBandwidth(k)).AICc >=
                   m_sel.AICc - 1e-9, 6:N)

    # Back-fitting runs in two phases and `converged` requires both: the
    # bandwidths settling and the coefficients reaching tol. Cut it short
    # and there is no fixed operator to take a trace of, so the reported
    # degrees of freedom are NaN rather than whatever the half-built
    # accumulator happened to hold.
    m_short = @test_logs (:warn,) match_mode=:any fit_mgwr(X, y, coords; max_iter = 2)
    @test !m_short.converged
    @test isnan(m_short.hat_trace)

    # Columbus is the case where back-fitting converges slowly: INC and
    # HOVAL are spatially correlated with each other, so the additive
    # components compete. @test_logs with no expected entry is the real
    # assertion here: it fails if any of the convergence warnings fire.
    # With mgwr's default range, every bandwidth lies between 42 and N.
    csv = joinpath(@__DIR__, "..", "..", "notebooks", "data", "columbus.csv")
    if isfile(csv)
        # Parsed by hand rather than with DelimitedFiles, which is no
        # longer an unconditional stdlib and is not a test dependency.
        lines = filter(!isempty, strip.(readlines(csv)))
        cnames = string.(split(lines[1], ','))
        rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
        getcol(c) = [r[findfirst(==(c), cnames)] for r in rows]
        yc = getcol("CRIME")
        Xc = hcat(getcol("INC"), getcol("HOVAL"))
        cc = hcat(getcol("X"), getcol("Y"))

        mc = @test_logs fit_mgwr(Xc, yc, cc)
        @test mc.converged
        @test all(42 .<= mc.bandwidths .<= length(yc))
        @test all(isfinite, mc.enp)
        @test sum(mc.enp) == mc.hat_trace
        @test isapprox(mc.sigma2,
                       sum(abs2, residuals(mc)) /
                       (length(yc) - mc.hat_trace); rtol = 1e-10)
    end
end
