@testset "11c. MGWR degrees of freedom against a solved fixed point" begin
    _progress("11c. MGWR degrees of freedom against a solved fixed point")
    # The back-fitting fixed point is a linear system, so at small N it
    # can be solved outright and both degrees-of-freedom paths checked
    # against the answer instead of against each other. Every bug found
    # in this part of the package was a formula that no test compared to
    # a closed form, so this is the test that should have existed.
    #
    #   f_k = S^(k) (y - sum_{j != k} f_j)
    #   => R_k + S^(k) sum_{j != k} R_j = S^(k)
    #
    # which is one (NK x NK) solve with N right-hand sides.
    GW = SpatialEconometrics.GWREstimation

    rng = MersenneTwister(3311); N = 30
    coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
    X = randn(rng, N, 2)
    y = 2.0 .+ (1.0 .+ 0.3 .* coords[:, 1]) .* X[:, 1] .-
        0.5 .* X[:, 2] .+ 0.5 .* randn(rng, N)
    X_aug = hcat(ones(N), X)
    K = 3
    kern = BisquareKernel()
    tree = GW._build_tree(coords)

    m_ex = fit_mgwr(X, y, coords; tol = 1e-8, max_iter = 4000)
    @test m_ex.converged
    @test m_ex.df_method === :exact

    Sk = map(1:K) do k
        bw = AdaptiveBandwidth(round(Int, m_ex.bandwidths[k]))
        xk = X_aug[:, k]
        S = zeros(N, N)
        for i in 1:N
            w = GW._get_weights(tree, coords, i, bw, kern)
            den = sum(w[j] * xk[j]^2 for j in 1:N)
            den <= 1e-12 && continue
            for j in 1:N
                S[i, j] = xk[i] * w[j] * xk[j] / den
            end
        end
        S
    end

    A = zeros(N * K, N * K); Bm = zeros(N * K, N)
    for k in 1:K
        rows = ((k - 1) * N + 1):(k * N)
        A[rows, rows] = Matrix(I, N, N)
        for j in 1:K
            j == k && continue
            A[rows, ((j - 1) * N + 1):(j * N)] = Sk[k]
        end
        Bm[rows, :] = Sk[k]
    end
    Rst = A \ Bm
    truth = [tr(Rst[((k - 1) * N + 1):(k * N), :]) for k in 1:K]

    # The solved operator must reproduce the fit it is the operator of.
    f_hat = zeros(N)
    for k in 1:K
        f_hat .+= Rst[((k - 1) * N + 1):(k * N), :] * y
    end
    @test isapprox(f_hat, m_ex.fitted; atol = 1e-6)

    # The accumulator must match it to the tolerance it was run at.
    @test isapprox(m_ex.enp, truth; atol = 1e-4)
    @test isapprox(m_ex.hat_trace, sum(truth); atol = 1e-4)
    @test all(iszero, m_ex.enp_se)       # exact, so no Monte Carlo error

    # The probe estimator must agree with the same truth, within its own
    # reported error. 4 SEs rather than 2: against a solved fixed point
    # these z-scores have standard deviation about 1.04 at 200 probes
    # (1.28 at 50), because a quadratic form in Rademacher variables is
    # not Gaussian and its variance is estimated from the same probes.
    m_st = fit_mgwr(X, y, coords; inference = :stochastic,
                    rng = MersenneTwister(606), tol = 1e-8, max_iter = 4000)
    @test m_st.df_method === :stochastic
    @test all(isfinite, m_st.enp_se)
    @test all(>(0.0), m_st.enp_se)
    @test all(abs.(m_st.enp .- truth) .<= 4 .* m_st.enp_se)
    @test abs(m_st.hat_trace - sum(truth)) <= 4 * sqrt(sum(abs2, m_st.enp_se))

    # Whatever is asked of inference, the fit must come out the same: the
    # degrees of freedom are read off the frozen smoother afterwards and
    # cannot feed back into the back-fitting.
    m_no = fit_mgwr(X, y, coords; inference = false, tol = 1e-8, max_iter = 4000)
    @test m_no.df_method === :none
    @test m_no.coef_local == m_ex.coef_local
    @test m_st.coef_local == m_ex.coef_local
    @test all(isnan, m_no.enp_se)

    # Memory, not principle, is what gates the exact path, so the
    # stochastic one must still work where the exact one gives up.
    m_gate = @test_logs (:warn,) match_mode=:any fit_mgwr(X, y, coords; max_inference_n = 5)
    @test m_gate.df_method === :none
    m_sg = fit_mgwr(X, y, coords; inference = :stochastic, max_inference_n = 5,
                    rng = MersenneTwister(77))
    @test m_sg.df_method === :stochastic
    @test isfinite(m_sg.hat_trace)

    @test_throws ArgumentError fit_mgwr(X, y, coords; inference = :exact)
end
