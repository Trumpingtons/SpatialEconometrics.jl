@testset "7. Driscoll-Kraay & Conley HAC" begin
    _progress("7. Driscoll-Kraay & Conley HAC")
    rng = MersenneTwister(707); L = 10; N = L * L; T_per = 8; W = make_rook_weights(L)
    X = randn(rng, N * T_per, 2); ρ_true = 0.4
    S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
    μ_i = randn(rng, N); y = zeros(N * T_per)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        y[idx] = S_lu \ (μ_i .+ X[idx, :] * [1.0, -0.5] .+ (0.3 .+ 0.5 .* abs.(randn(rng, N))) .* randn(rng, N))
    end

    m_pan = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)
    V_dk = robust_vcov(m_pan, DriscollKraayVCOV())
    @test size(V_dk) == size(vcov(m_pan))
    @test all(eigvals(Symmetric(V_dk)) .>= -1e-10)  # PSD

    coords = hcat(repeat(1.0:L, L), repeat(1.0:L, inner=L))
    V_con = robust_vcov(m_pan, ConleySpatialHAC(coords, 3.0))
    @test size(V_con) == size(vcov(m_pan))
    @test all(eigvals(Symmetric(V_con)) .>= -1e-10)

    # ── the scale of the sandwich, pinned exactly ──
    # A sandwich covariance is equivariant in the scale of y: multiply y
    # by c and every beta standard error must scale by exactly c, while
    # rho's is unchanged because rho is dimensionless. This held only
    # by accident of sigma2 ~ 1. `extract_spatial_jacobian_scores`
    # returned `sigma2 .* vcov_qml` as the bread, but `vcov_qml` is
    # already sigma2 * (Z'Z)^-1 and the meat is built from Z'e with no
    # 1/sigma2, so the bread carried sigma2 twice over: the variance
    # came out scaled by sigma2^4 and the standard error by sigma2^2.
    # Against a 200-replication sampling distribution the robust errors
    # were four times too small; with sigma2 > 1 they are too large
    # instead. Every assertion above passes under that bug -- they test
    # shape, symmetry and positive-semidefiniteness, all of which a
    # misscaled sandwich keeps -- so the scale needs pinning on its own.
    # On the tolerance, which is measured rather than guessed. The
    # equivariance is exact in exact arithmetic, but rho comes from a
    # numerical search that lands differently per Julia version, so the
    # two fits are not bit-identical: rho differs by 2e-11 between them on
    # 1.13.3 and by 4e-8 on 1.10.13, and the Driscoll-Kraay standard error
    # amplifies the latter to 6.3e-7 relative. An earlier rtol of 1e-8 was
    # chosen because it passed on 1.13, where the worst deviation is
    # 2.6e-11; it failed on 1.10 in CI. That is the argument for measuring
    # a margin instead of taking the tightest number that happens to pass.
    # 1e-4 sits 160x above the worst observed deviation and roughly 1e6
    # below the relative deviation of 80 to 240 that the bug produced, so
    # it discriminates with room at both ends.
    c_scale = 3.0
    m_scaled = fit_spatial_panel(SAR(), TwoWayFE(), W, X, c_scale .* y, N, T_per)
    for spec in (DriscollKraayVCOV(), ConleySpatialHAC(coords, 3.0))
        se_1 = sqrt.(diag(robust_vcov(m_pan, spec)))
        se_c = sqrt.(diag(robust_vcov(m_scaled, spec)))
        @test se_c[1:2] ≈ c_scale .* se_1[1:2] rtol = 1e-4
        @test se_c[end] ≈ se_1[end] rtol = 1e-4
    end

    # ── and pinned against the sandwich built by hand ──
    # Rebuilt from the formula and the model's own public fields:
    # Z = [X_tilde, W(I-rho*W)^-1 X_tilde*beta], meat = sum_t (Z_t'e_t)(Z_t'e_t)',
    # and bread = vcov / sigma2. The bread is the QML covariance de-scaled,
    # not (Z'Z)^-1: that is deliberate, and a first version of this test
    # asserted (Z'Z)^-1 and failed. The two agree on the beta block, but
    # the QML information matrix's rho block carries the log-determinant
    # trace terms, which (Z'Z)^-1 knows nothing about, and because that
    # matrix is inverted jointly its beta block is not (Z'Z)^-1 either --
    # measured 1.7% away, the Schur complement of the beta-rho cross
    # terms. Using the model's own covariance as bread is what makes the
    # rho row of the robust table mean anything. max_lag = 0 and no
    # small-sample correction leave the meat as H*H' with nothing folded
    # in, so this is exact.
    beta_hat = coef(m_pan)[1:2]
    Slu_h = lu(sparse(1.0I, N, N) - m_pan.rho * W)
    Xb_h = m_pan.X_tilde * beta_hat
    GXb_h = vec(W * (Slu_h \ reshape(Xb_h, N, T_per)))
    Z_h = hcat(m_pan.X_tilde, GXb_h)
    H_h = zeros(size(Z_h, 2), T_per)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        H_h[:, t] = Z_h[idx, :]' * m_pan.residuals[idx]
    end
    bread_h = vcov(m_pan) ./ m_pan.sigma2
    V_hand = bread_h * (H_h * H_h') * bread_h'
    V_got = robust_vcov(m_pan, DriscollKraayVCOV(max_lag = 0,
                                                 small_sample_correct = false))
    @test V_got ≈ V_hand rtol = 1e-10
end
