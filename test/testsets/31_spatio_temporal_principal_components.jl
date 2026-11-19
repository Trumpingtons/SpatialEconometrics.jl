@testset "31. Spatio-temporal principal components" begin
    _progress("31. Spatio-temporal principal components")
    SF = SpatialEconometrics.SpatialFiltering
    SD = SpatialEconometrics.SpatialDiagnostics

    # The reference's own grid: 2002-2018 observed at the midpoints of
    # seventeen unit intervals, expanded on [0, 17] in five Fourier
    # functions per feature.
    tsq = collect(0.5:1.0:16.5)
    IV = (0.0, 17.0)
    Wq = row_standardize(make_rook_weights(6))
    Nq = size(Wq, 1)

    # Deterministic data, for the reason recorded in testset 30: a
    # stand-in for a draw has to be independent of the lattice.
    function det_normals31(n::Int; seed::UInt64 = 0x853c49e6748fea9b)
        s = seed
        nxt = () -> (s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
                     Float64((s >> 11) & 0x001fffffffffffff) / 9007199254740992.0)
        out = Vector{Float64}(undef, n); i = 1
        while i <= n
            u1 = max(nxt(), 1e-12); u2 = nxt()
            r = sqrt(-2 * log(u1)); th = 2pi * u2
            out[i] = r * cos(th); i += 1
            i <= n && (out[i] = r * sin(th); i += 1)
        end
        return out
    end

    pq, ntq = 3, length(tsq)
    noise = det_normals31(Nq * pq * ntq)
    Xq = Array{Float64}(undef, Nq, pq, ntq)
    let c = 0
        for i in 1:Nq, j in 1:pq, t in 1:ntq
            c += 1
            Xq[i, j, t] = noise[c] * 0.4 + sin(tsq[t] / 3) * (i % 5) / 3 +
                          cos(tsq[t] / 5) * j / 2
        end
    end

    @testset "31a. the Fourier basis is orthonormal" begin
        # The construction rests on the basis being orthonormal with
        # respect to the integral over the interval: that is what makes
        # the functional inner product <u, X> equal the dot product of
        # coefficient vectors, and so what makes gamma'S gamma the
        # variance of the functional principal component.
        ng = 100001
        tg = collect(range(IV[1], IV[2]; length = ng))
        P = fourier_basis(tg, 5; interval = IV)
        h = (IV[2] - IV[1]) / (ng - 1)
        w = [k == 1 || k == ng ? 0.5 : 1.0 for k in 1:ng]
        G = [sum(w .* P[:, i] .* P[:, j]) * h for i in 1:5, j in 1:5]
        @test maximum(abs.(G - I)) < 1e-12

        # Least squares in that basis is exact on a function already in
        # its span, which is the identity the expansion step needs.
        Pt = fourier_basis(tsq, 5; interval = IV)
        ctrue = [1.3, -0.7, 2.1, 0.4, -1.6]
        @test Pt \ (Pt * ctrue) ≈ ctrue atol = 1e-12

        # Padding the interval is not cosmetic. A Fourier basis is
        # periodic, so expanding on exactly [t_1, t_T] forces the fit to
        # take the same value at both ends; the default pads by half a
        # step, which is what [0, 17] is for this grid.
        # For this grid the default padding gives exactly the [0, 17]
        # the reference uses, which is the point of the default.
        @test fourier_basis(tsq, 5) ≈ fourier_basis(tsq, 5; interval = IV)
        # Unpadded, the first and last rows coincide -- the fit is
        # forced to return to where it started, whatever the data do.
        Pn = fourier_basis(tsq, 5; interval = (first(tsq), last(tsq)))
        @test Pn[1, :] ≈ Pn[end, :]
        Pp = fourier_basis(tsq, 5; interval = IV)
        @test !isapprox(Pp[1, :], Pp[end, :])
        @test_throws ArgumentError fourier_basis(tsq, 0)
        @test_throws ArgumentError fourier_basis([1.0], 1)
        @test_throws ArgumentError fourier_basis([3.0, 1.0, 2.0], 2)
    end

    @testset "31b. y'B*y is Geary's numerator" begin
        # Expanding sum_ij w_ij (y_i - y_j)^2 into a quadratic form is
        # the step that turns the ratio into an eigenproblem, so it is
        # checked against the raw double sum rather than assumed.
        y = det_normals31(Nq; seed = 0x9e3779b97f4a7c15)
        Wd = Matrix(Wq)
        raw = sum(Wd[i, j] * (y[i] - y[j])^2 for i in 1:Nq, j in 1:Nq)
        G = SF._geary_matrix(Wq)
        @test dot(y, G * y) ≈ raw atol = 1e-10
        @test dot(y, ((G + G') ./ 2) * y) ≈ raw atol = 1e-10
        # and the ratio it builds is the package's own Geary's c
        @test (Nq - 1) / (2 * sum(Wq)) * dot(y, G * y) / sum(abs2, y .- mean(y)) ≈
              SD.geary_c(y, Wq).C_stat atol = 1e-12
    end

    @testset "31c. every eigenvalue is its own objective" begin
        # The claim the whole method rests on: the eigenvalue of E is
        # not a proxy for the quantity being maximised, it IS that
        # quantity -- the functional principal component's variance
        # times the spatial autocorrelation of the scores it produces.
        # Both factors are recomputed here from other modules, so this
        # fails if either the scaling of E or the centring of A is
        # wrong.
        for meas in (:geary, :moran)
            m = stpca(Xq, Wq; measure = meas, n_basis = 5, times = tsq,
                      interval = IV)
            @test length(m.values) == pq * 5
            @test issorted(m.values; rev = true)
            @test m.vectors' * m.vectors ≈ I atol = 1e-10
            for k in 1:length(m.values)
                y = m.scores[:, k]
                @test mean(y) ≈ 0.0 atol = 1e-10
                ac = meas === :geary ? SD.geary_c(y, Wq).C_stat :
                                       moran_coefficient(y, Wq)
                var = sum(abs2, y) / Nq
                @test m.autocorr[k] ≈ ac atol = 1e-10
                @test m.variance[k] ≈ var atol = 1e-10
                @test m.values[k] ≈ var * ac atol = 1e-9
            end
            @test sum(abs, m.proportion) ≈ 1.0
        end
    end

    @testset "31d. the sign structure of the two measures" begin
        mg = stpca(Xq, Wq; measure = :geary, times = tsq, interval = IV)
        mm = stpca(Xq, Wq; measure = :moran, times = tsq, interval = IV)
        # E is a variance times a Geary ratio, and neither factor can be
        # negative, so E is non-negative definite. This is a property of
        # the construction, not of the sample.
        @test minimum(mg.values) > -1e-10
        @test all(>=(0.0), mg.autocorr)
        # Moran's I is a cross-product and can be negative, which is the
        # global/local split the eigenvalue signs carry. On data with
        # both structures present both signs appear.
        @test minimum(mm.values) < 0.0
        @test maximum(mm.values) > 0.0
        @test sign.(mm.values) == sign.(mm.autocorr)
    end

    @testset "31e. scale and rank behaviour" begin
        m1 = stpca(Xq, Wq; times = tsq, interval = IV)
        m3 = stpca(3.0 .* Xq, Wq; times = tsq, interval = IV)
        # Variance scales with c^2 and the ratios are scale free, so the
        # whole spectrum scales by c^2 and the directions do not move.
        @test m3.values ≈ 9 .* m1.values atol = 1e-8 * maximum(m1.values)
        @test abs.(m3.vectors' * m1.vectors) ≈ I atol = 1e-6
        @test m3.autocorr ≈ m1.autocorr atol = 1e-8

        # Adding a constant to a unit's series moves only the basis
        # coefficient of the constant function, and the coefficients are
        # centred, so a shift common to every unit changes nothing.
        @test stpca(Xq .+ 7.0, Wq; times = tsq, interval = IV).values ≈
              m1.values atol = 1e-8 * maximum(m1.values)

        # B = p * n_basis directions, but at most n - 1 can carry
        # variance when there are fewer units than coefficients.
        Xsmall = Xq[1:4, :, :]
        Wsmall = row_standardize(sparse(Float64[0 1 0 0; 1 0 1 0;
                                                0 1 0 1; 0 0 1 0]))
        ms = stpca(Xsmall, Wsmall; times = tsq, interval = IV)
        @test length(ms.values) == pq * 5
        @test count(>(1e-8 * maximum(ms.values)), ms.values) <= 3
    end

    @testset "31f. the single-feature method and feature shares" begin
        Xone = Xq[:, 1, :]
        m2d = stpca(Xone, Wq; times = tsq, interval = IV)
        m3d = stpca(reshape(Xone, Nq, 1, ntq), Wq; times = tsq, interval = IV)
        @test m2d.values ≈ m3d.values
        @test m2d.n_features == 1
        # One feature takes the whole of every component.
        @test feature_contributions(m2d, 1) ≈ [1.0]

        m = stpca(Xq, Wq; times = tsq, interval = IV, varnames = ["a", "b", "c"])
        for k in (1, 2, length(m.values))
            fc = feature_contributions(m, k)
            @test length(fc) == pq
            @test sum(fc) ≈ 1.0
            @test all(>=(0.0), fc)
        end
        # The shares are areas under |u_kj|, so refining the grid moves
        # them only by the trapezoid's own error.
        # |u_kj| has a corner wherever u_kj crosses zero, so the
        # trapezoid converges more slowly than it would on a smooth
        # integrand; the shares are stable to about 1e-5, which is
        # four orders finer than any difference between features.
        @test feature_contributions(m, 1; n_grid = 4096) ≈
              feature_contributions(m, 1; n_grid = 512) atol = 1e-4
        # A feature that is identically zero contributes nothing.
        Xz = copy(Xq); Xz[:, 3, :] .= 0.0
        mz = stpca(Xz, Wq; times = tsq, interval = IV)
        @test feature_contributions(mz, 1)[3] ≈ 0.0 atol = 1e-10

        # The weighting function is the direction read back as a
        # function of time, so its coefficients are the eigenvector's.
        U = component_weight(m, 1, tsq)
        @test size(U) == (ntq, pq)
        Pt = fourier_basis(tsq, 5; interval = IV)
        @test U[:, 2] ≈ Pt * m.vectors[6:10, 1] atol = 1e-12
        @test m.varnames == ["a", "b", "c"]
    end

    @testset "31g. planted structure lands where it should" begin
        # One draw pins what the Monte Carlo in docs/AUDIT.md measures
        # as a rate: a smooth field is the leading :moran component and
        # never the leading :geary one.
        se = spatial_eigenvectors(Wq)
        smooth = se.vectors[:, 1]
        @test moran_coefficient(smooth, Wq) > 0.8
        prof = sin.(2pi .* tsq ./ 17)
        Xp = copy(Xq)
        for i in 1:Nq
            Xp[i, 1, :] .+= 3.0 * smooth[i] .* prof
        end
        mm = stpca(Xp, Wq; measure = :moran, times = tsq, interval = IV)
        mg = stpca(Xp, Wq; measure = :geary, times = tsq, interval = IV)
        cm = [abs(cor(mm.scores[:, k], smooth)) for k in 1:length(mm.values)]
        cg = [abs(cor(mg.scores[:, k], smooth)) for k in 1:length(mg.values)]
        @test argmax(cm) == 1
        # The planted field is spread across the spectrum rather than
        # captured whole -- the Monte Carlo median of this correlation
        # is about 0.56 at this N -- so the claim is that the leading
        # component is where most of it goes, not that it takes it all.
        @test cm[1] > 0.5
        @test cm[1] > 1.5 * maximum(cm[2:end])
        @test argmax(cg) != 1

        # The sign groups: :moran splits global from local, :geary
        # cannot, which is the construction and not the sample.
        gm = component_groups(mm)
        @test !isempty(gm.positive) && !isempty(gm.negative)
        @test all(>(0.0), mm.values[gm.positive])
        @test all(<(0.0), mm.values[gm.negative])
        @test issorted(mm.values[gm.positive]; rev = true)
        @test issorted(abs.(mm.values[gm.negative]); rev = true)
        @test 1 in gm.positive
        @test isempty(component_groups(mg).negative)
        # and the planted feature is the one the component reads
        @test argmax(feature_contributions(mm, 1)) == 1
    end

    @testset "31h. what stpca rejects" begin
        @test_throws ArgumentError stpca(Xq, Wq; measure = :getis)
        @test_throws DimensionMismatch stpca(Xq, row_standardize(make_rook_weights(5)))
        @test_throws DimensionMismatch stpca(Xq, Wq; times = tsq[1:5])
        # more basis functions than observation times is not identified
        @test_throws ArgumentError stpca(Xq, Wq; n_basis = ntq + 1, times = tsq)
        Xbad = copy(Xq); Xbad[1, 1, 1] = NaN
        @test_throws ArgumentError stpca(Xbad, Wq; times = tsq)
        @test_throws ArgumentError stpca(Xq, Wq; times = tsq, varnames = ["a"])
        @test_throws ArgumentError stpca(Xq, spzeros(Nq, Nq); times = tsq)
        m = stpca(Xq, Wq; times = tsq, interval = IV)
        @test_throws ArgumentError component_weight(m, 0, tsq)
        @test_throws ArgumentError component_weight(m, 999, tsq)
        @test_throws ArgumentError feature_contributions(m, 1; n_grid = 1)
        @test occursin("Geary", sprint(show, m))
        @test occursin("Moran", sprint(show,
            stpca(Xq, Wq; measure = :moran, times = tsq, interval = IV)))
    end
end
