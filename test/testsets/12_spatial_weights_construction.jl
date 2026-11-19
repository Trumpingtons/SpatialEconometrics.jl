@testset "12. Spatial Weights Construction" begin
    _progress("12. Spatial Weights Construction")
    rng = MersenneTwister(1200)
    N = 100
    coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)

    # knn weights
    W_knn = knn_weights(coords, 5)
    @test size(W_knn) == (N, N)
    @test all(abs.(vec(sum(W_knn, dims=2)) .- 1.0) .< 1e-10)  # row-standardized

    # knn binary + symmetric
    W_knn_sym = knn_weights(coords, 4; binary=true, symmetrize=true, row_standardize=false)
    @test issymmetric(W_knn_sym) || norm(W_knn_sym - W_knn_sym', Inf) < 1e-10
    @test all(nonzeros(W_knn_sym) .== 1.0)

    # distance-decay weights
    W_dist = distance_weights(coords, 3.0; decay=:inverse)
    @test size(W_dist) == (N, N)
    sc_dist = spatial_connectivity(W_dist)
    @test sc_dist.N == N
    @test sc_dist.avg_neighbors > 0

    # bisquare decay
    W_bisq = distance_weights(coords, 4.0; decay=:bisquare, row_standardize=false)
    @test all(nonzeros(W_bisq) .>= 0.0)
    @test all(nonzeros(W_bisq) .<= 1.0)

    # distance band
    W_band = band_weights(coords, 1.0, 3.0)
    @test size(W_band) == (N, N)

    # block-diagonal
    W1 = make_rook_weights(4); W2 = make_rook_weights(5)
    W_block = block_diagonal_weights([W1, W2])
    @test size(W_block) == (16 + 25, 16 + 25)
    @test nnz(W_block) == nnz(W1) + nnz(W2)

    # standardization
    W_raw = knn_weights(coords, 3; row_standardize=false)
    W_rs = row_standardize(W_raw)
    row_sums = vec(sum(W_rs, dims=2))
    @test all(s -> s ≈ 0.0 || s ≈ 1.0, row_sums)

    W_spec = spectral_standardize(W_raw)
    @test size(W_spec) == (N, N)

    W_mm = minmax_standardize(W_raw)
    @test size(W_mm) == (N, N)

    # connectivity diagnostics
    sc = spatial_connectivity(W_knn)
    @test sc.N == N
    @test sc.n_islands == 0
    @test sc.avg_neighbors ≈ 5.0 atol=0.5
    @test sc.spectral_radius > 0

    # GAL I/O round-trip
    gal_path = tempname() * ".gal"
    ids = string.(1:N)
    W_for_gal = make_rook_weights(10)
    write_gal(gal_path, W_for_gal, string.(1:100))
    W_read, ids_read = read_gal(gal_path)
    @test size(W_read) == size(W_for_gal)
    @test length(ids_read) == 100

    # GWT I/O round-trip
    gwt_path = tempname() * ".gwt"
    write_gwt(gwt_path, W_for_gal, string.(1:100))
    W_read2, ids_read2 = read_gwt(gwt_path)
    @test size(W_read2) == size(W_for_gal)
end
