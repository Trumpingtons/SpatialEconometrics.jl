@testset "15b. Variances of SLX, SEM, SDEM and SAC" begin
    _progress("15b. Variances of SLX, SEM, SDEM and SAC")
    # None of these had ever been checked against a value. SAC and SDEM
    # turned out not to compute a variance for their spatial parameters at
    # all: the diagonal slots held 1/N, so every reported standard error
    # for rho and lambda was exactly 1/sqrt(N) whatever the data said.
    rng = MersenneTwister(1_500_021)
    L = 12; N = L * L
    W = make_rook_weights(L); Wd = Matrix(W)
    X = hcat(randn(rng, N), randn(rng, N)); Xa = hcat(ones(N), X)
    u = (I - 0.4 .* Wd) \ randn(rng, N)
    y = (I - 0.5 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+ u)

    function numhess(f, th)
        n = length(th); h = 1e-5 .* max.(abs.(th), 1.0); H = zeros(n, n)
        for i in 1:n, j in 1:n
            ei = zeros(n); ei[i] = h[i]; ej = zeros(n); ej[j] = h[j]
            H[i, j] = (f(th + ei + ej) - f(th + ei - ej) -
                       f(th - ei + ej) + f(th - ei - ej)) / (4h[i] * h[j])
        end
        H
    end
    ld(A) = first(logabsdet(A))

    # --- SLX is plain OLS, so the variance is exactly knowable. It used
    # the ML sigma2 = RSS/N while dof_residual reported N - P, which
    # understated every standard error by sqrt((N-P)/N).
    m_slx3 = fit_slx(W, X, y)
    Xs = hcat(ones(N), X, Wd * X); Ps = size(Xs, 2)
    e_slx3 = y .- Xs * (Xs \ y)
    V_ols = (dot(e_slx3, e_slx3) / (N - Ps)) .* inv(Symmetric(Xs' * Xs))
    @test isapprox(vcov(m_slx3), V_ols; rtol = 1e-10)
    @test isapprox(stderror(m_slx3), sqrt.(diag(V_ols)); rtol = 1e-10)
    @test dof_residual(m_slx3) == N - Ps
    # the ML variance is a different matrix, and smaller
    V_ml = (dot(e_slx3, e_slx3) / N) .* inv(Symmetric(Xs' * Xs))
    @test !isapprox(vcov(m_slx3), V_ml; rtol = 1e-6)
    @test all(diag(vcov(m_slx3)) .> diag(V_ml))
    # sigma2 itself must stay the ML value, so the likelihood is unchanged
    @test isapprox(m_slx3.sigma2, dot(e_slx3, e_slx3) / N; rtol = 1e-10)

    # --- SEM against the expected information matrix, written out here.
    # In the error model beta is asymptotically independent of (lambda,
    # sigma2), so the matrix is block diagonal.
    y_err = Xa * [1.0, 2.0, -1.0] .+ ((I - 0.4 .* Wd) \ randn(rng, N))
    m_sem3 = fit_spatial_error(W, X, y_err; method = ExactLU())
    P = 3
    B = I - m_sem3.lambda .* Wd; GB = Wd * inv(B)
    Xf = B * Xa; s2 = m_sem3.sigma2
    Ie = zeros(P + 2, P + 2)
    Ie[1:P, 1:P] = (transpose(Xf) * Xf) ./ s2
    Ie[P + 1, P + 1] = tr(GB * GB) + tr(transpose(GB) * GB)
    Ie[P + 1, P + 2] = tr(GB) / s2; Ie[P + 2, P + 1] = Ie[P + 1, P + 2]
    Ie[P + 2, P + 2] = N / (2 * s2^2)
    se_sem_ref = sqrt.(diag(inv(Symmetric(Ie)))[1:(P + 1)])
    @test isapprox(stderror(m_sem3), se_sem_ref; rtol = 1e-8)
    @test isapprox(m_sem3.lambda_se, se_sem_ref[end]; rtol = 1e-8)
    # lambda-s standard error used 30 unseeded probes and moved 8% between
    # identical fits; beta-s, built from Xf-prime-Xf, never did.
    sems = [fit_spatial_error(W, X, y_err; method = ExactLU()) for _ in 1:4]
    for mm in sems
        @test vcov(mm) == vcov(sems[1])
    end

    # --- SDEM is an error model in augmented regressors, so the same
    # expected information applies with X replaced by [1 X WX].
    Xd = hcat(ones(N), X, Wd * X)
    y_sd = Xd * [1.0, 2.0, -1.0, 0.5, -0.3] .+
           ((I - 0.4 .* Wd) \ randn(rng, N))
    m_sdem3 = fit_sdem(W, X, y_sd; method = ExactLU())
    Pd = 5
    Bd2 = I - m_sdem3.lambda .* Wd; GBd = Wd * inv(Bd2)
    s2d = m_sdem3.sigma2
    Id = zeros(Pd + 2, Pd + 2)
    Id[1:Pd, 1:Pd] = (transpose(Bd2 * Xd) * (Bd2 * Xd)) ./ s2d
    Id[Pd + 1, Pd + 1] = tr(GBd * GBd) + tr(transpose(GBd) * GBd)
    Id[Pd + 1, Pd + 2] = tr(GBd) / s2d
    Id[Pd + 2, Pd + 1] = Id[Pd + 1, Pd + 2]
    Id[Pd + 2, Pd + 2] = N / (2 * s2d^2)
    se_sdem_ref = sqrt.(diag(inv(Symmetric(Id)))[1:(Pd + 1)])
    @test isapprox(stderror(m_sdem3)[end], se_sdem_ref[end]; rtol = 1e-8)
    @test !isapprox(stderror(m_sdem3)[end], 1 / sqrt(N); rtol = 1e-3)

    # --- SAC: the joint information for (rho, lambda). Checked against a
    # numerical Hessian of the full log-likelihood, which shares none of
    # the algebra. Expected and observed information differ in finite
    # samples, so the spatial parameters get a loose tolerance; separately
    # a 400-replication Monte Carlo at N = 196 put the reported standard
    # errors at 0.94 (rho), 0.96 (lambda) and 0.98 (beta1) of the sampling
    # standard deviation, with 93-94% coverage of the nominal 95%.
    m_sac3 = fit_sac(W, W, X, y; method = ExactLU())
    f_sac = th -> begin
        bb = th[1:P]; rr = th[P + 1]; ll2 = th[P + 2]; ss = th[P + 3]
        ss <= 0 && return -Inf
        A2 = I - rr .* Wd; B2 = I - ll2 .* Wd
        e2 = B2 * (A2 * y .- Xa * bb)
        -0.5N * log(2pi) - 0.5N * log(ss) - dot(e2, e2) / (2ss) +
            ld(A2) + ld(B2)
    end
    Hs = numhess(f_sac, vcat(coef(m_sac3)[1:P], m_sac3.rho,
                             m_sac3.lambda, m_sac3.sigma2))
    se_sac_num = sqrt.(diag(inv(-Symmetric(Hs)))[1:(P + 2)])
    @test isapprox(stderror(m_sac3)[1:P], se_sac_num[1:P]; rtol = 0.06)
    @test isapprox(stderror(m_sac3)[P + 1], se_sac_num[P + 1]; rtol = 0.2)
    @test isapprox(stderror(m_sac3)[P + 2], se_sac_num[P + 2]; rtol = 0.2)

    # The old answer was 1/sqrt(N) for both, so pin that it is gone and
    # that the standard errors actually respond to the data.
    @test !isapprox(stderror(m_sac3)[P + 1], 1 / sqrt(N); rtol = 1e-3)
    @test !isapprox(stderror(m_sac3)[P + 2], 1 / sqrt(N); rtol = 1e-3)
    @test stderror(m_sac3)[P + 1] != stderror(m_sac3)[P + 2]
    @test size(vcov(m_sac3)) == (P + 2, P + 2)
    @test all(isfinite, vcov(m_sac3))
    @test issymmetric(round.(vcov(m_sac3); digits = 12))

    y2 = (I - 0.1 .* Wd) \ (Xa * [1.0, 2.0, -1.0] .+
                            ((I - 0.7 .* Wd) \ randn(rng, N)))
    m_sac4 = fit_sac(W, W, X, y2; method = ExactLU())
    @test stderror(m_sac4)[P + 1] != stderror(m_sac3)[P + 1]
    @test stderror(m_sac4)[P + 2] != stderror(m_sac3)[P + 2]

    # Above the size gate the probe path runs instead: seeded it must
    # repeat, and it must land near the exact answer.
    a1 = fit_sac(W, W, X, y; method = ExactLU(), max_exact_n = 0,
                 vcov_probes = 600, rng = MersenneTwister(9))
    a2 = fit_sac(W, W, X, y; method = ExactLU(), max_exact_n = 0,
                 vcov_probes = 600, rng = MersenneTwister(9))
    @test vcov(a1) == vcov(a2)
    @test isapprox(stderror(a1)[P + 1], stderror(m_sac3)[P + 1]; rtol = 0.2)
    @test isapprox(stderror(a1)[P + 2], stderror(m_sac3)[P + 2]; rtol = 0.2)
    # the probe path must not disturb the fit itself
    @test a1.rho == m_sac3.rho
    @test a1.lambda == m_sac3.lambda
    @test coef(a1)[1:P] == coef(m_sac3)[1:P]
end
