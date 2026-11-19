@testset "30. Bayesian specification comparison" begin
    _progress("30. Bayesian specification comparison")
    # Deterministic data throughout, since `randn(::MersenneTwister)` is
    # not promised to give the same numbers across Julia releases and
    # this testset asserts which specification wins. The disturbances
    # need their own small generator rather than a smooth function of
    # the index: on a lattice, `sin(i)` is spatially autocorrelated --
    # consecutive indices are neighbours -- so a design whose errors are
    # built that way is not the independent-error case it looks like,
    # and the specification comparison correctly refuses to call it one.
    # An LCG with Box-Muller gives a stream that is both independent of
    # the lattice and identical on every version.
    function det_normals(n::Int; seed::UInt64 = 0x2545f4914f6cdd1d)
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

    W5 = row_standardize(make_rook_weights(5))
    N5 = size(W5, 1)
    X5 = hcat(Float64[sin(i / 2) for i in 1:N5],
              Float64[cos(i / 3) for i in 1:N5])
    e5 = 0.4 .* det_normals(N5)
    y5 = (I - 0.5 * W5) \ (1.0 .+ X5 * [1.5, -0.8] .+ (W5 * X5) * [0.7, 0.0] .+ e5)

    # The same construction on a 16 x 16 lattice, where the posterior
    # over the spatial parameter is sharp and interior rather than
    # spread across the whole prior support.
    W16 = row_standardize(make_rook_weights(16))
    N16 = size(W16, 1)
    # Covariates from the same stream: a smooth function of the index
    # would make `W*X` nearly collinear with `X`, which is a hard
    # Durbin design rather than the ordinary one this is meant to be.
    X16 = hcat(det_normals(N16; seed = 0xd1b54a32d192ed03),
               det_normals(N16; seed = 0x27bb2ee687b0b0fd))
    e16 = 0.5 .* det_normals(N16; seed = 0x9e3779b97f4a7c15)
    mu16 = X16 * [1.0, -0.8] .+ (W16 * X16) * [0.7, 0.0]
    y16_sdm = (I - 0.6 * W16) \ (mu16 .+ e16)

    @testset "30a. the marginal against an exact closed form" begin
        # With k = 3 columns and N = 25 observations the gamma term is
        # Gamma(11) = 10!, so the whole constant can be written out
        # exactly and compared against the implementation's Lanczos
        # logarithm and its QR determinant. This is the part that is
        # easy to get wrong and impossible to notice: dropping
        # |Z'Z|^(-1/2) or Gamma((n-k)/2) leaves a quantity that still
        # ranks two specifications of equal width correctly, and
        # mis-ranks every comparison between designs of different width.
        Z = hcat(ones(N5), X5)
        m = N5 - size(Z, 2)
        @test m == 22
        S = sum(abs2, y5 .- Z * (Z \ y5))
        ref = -0.5 * m * log(pi) - 0.5 * log(det(Z' * Z)) +
              log(factorial(big(10))) - 0.5 * m * log(S)
        @test log_marginal_likelihood(:ols, y5, X5, W5) ≈ Float64(ref) atol = 1e-10

        # And the Durbin design, where |Z'Z| is what penalises the two
        # extra columns: k = 5, so the gamma term is Gamma(10) = 9!.
        Zd = hcat(ones(N5), X5, W5 * X5)
        md = N5 - size(Zd, 2)
        Sd = sum(abs2, y5 .- Zd * (Zd \ y5))
        refd = -0.5 * md * log(pi) - 0.5 * log(det(Zd' * Zd)) +
               log(factorial(big(9))) - 0.5 * md * log(Sd)
        @test log_marginal_likelihood(:slx, y5, X5, W5) ≈ Float64(refd) atol = 1e-10
    end

    @testset "30b. the error family against an independent integral" begin
        # An independently written :sem marginal: dense log|I - lambda W|
        # by `logabsdet`, |Z'Z| by `det` of the filtered cross-product,
        # and the trapezoid written out longhand. The point of the check
        # is the determinant term, which moves with lambda for an error
        # model and is constant for a lag model -- dropping it is a
        # mistake that only shows up when error specifications are
        # compared against lag ones.
        Z = hcat(ones(N5), X5)
        k = size(Z, 2); m = N5 - k
        ng = 21; a, b = -0.9, 0.9
        grid = range(a, b; length = ng); h = step(grid)
        Wd = Matrix(W5)
        l = Float64[]
        for lam in grid
            F = I - lam * Wd
            Zf = F * Z; yf = F * y5
            Sf = sum(abs2, yf .- Zf * (Zf \ yf))
            push!(l, first(logabsdet(F)) - 0.5 * m * log(pi) -
                     0.5 * log(det(Zf' * Zf)) + log(factorial(big(10)) |> Float64) -
                     0.5 * m * log(Sf) - log(b - a))
        end
        mx = maximum(l)
        w = [i == 1 || i == ng ? 0.5 : 1.0 for i in 1:ng]
        ref = mx + log(sum(w .* exp.(l .- mx)) * h)
        @test log_marginal_likelihood(:sem, y5, X5, W5;
            rho_bounds = (-0.9, 0.9), n_grid = ng) ≈ ref atol = 1e-9
    end

    @testset "30c. the no-parameter specifications are the limit" begin
        # :ols and :slx are the lambda = 0 members of the families they
        # belong to, so integrating over a vanishing interval around
        # zero has to reproduce their closed forms exactly. This pins
        # the quadrature's normalisation against the analytic branch:
        # a prior that did not integrate to one over `rho_bounds` would
        # shift one and not the other.
        for (flat, family) in ((:ols, :sar), (:ols, :sem),
                               (:slx, :sdm), (:slx, :sdem))
            @test log_marginal_likelihood(flat, y5, X5, W5) ≈
                  log_marginal_likelihood(family, y5, X5, W5;
                      rho_bounds = (-1e-8, 1e-8), n_grid = 4) atol = 1e-8
        end
    end

    @testset "30d. quadrature and log-determinant settings" begin
        # Where the integrand rises and falls away to nothing inside the
        # interval, the trapezoid rule converges geometrically rather
        # than as O(h^2), because every Euler-Maclaurin boundary term
        # vanishes. The two lag specifications on this sample are that
        # case -- the integrand is 39 and 66 logs below its peak at the
        # ends of the support -- and 100 nodes is already at machine
        # precision.
        for s in (:sar, :sdm)
            v = [log_marginal_likelihood(s, y16_sdm, X16, W16; n_grid = g)
                 for g in (100, 200, 800)]
            @test maximum(abs.(v .- v[3])) < 1e-9
        end
        # The decay is a property of the sample, not of the rule, and it
        # fails in exactly the case worth knowing about: an error
        # specification fitted to lag data wants its parameter at the
        # edge of the stationary interval, so the integrand is only 8
        # logs down at rho = 0.995, the boundary term survives, and the
        # rate falls back to O(h^2). The same thing happens to every
        # specification at N = 25, where the posterior is wide enough to
        # be truncated by the prior's support. Both are harmless -- the
        # absolute error at the default 200 nodes stays below 1e-4,
        # three orders under the gaps that separate specifications --
        # but refining has to reduce it, which a quadrature that had
        # stopped depending on its grid would not do.
        for (s, y, X, W, g3) in ((:sem, y16_sdm, X16, W16, 800),
                                 (:sdem, y16_sdm, X16, W16, 800),
                                 (:sar, y5, X5, W5, 6400),
                                 (:sdm, y5, X5, W5, 6400),
                                 (:sem, y5, X5, W5, 6400),
                                 (:sdem, y5, X5, W5, 6400))
            v = [log_marginal_likelihood(s, y, X, W; n_grid = ng)
                 for ng in (100, 200, g3)]
            @test abs(v[2] - v[3]) < abs(v[1] - v[3]) < 1e-3
            @test 2.5 < (v[1] - v[3]) / (v[2] - v[3]) < 5.0
        end
        # Chebyshev log-determinants with deterministic traces are an
        # approximation to log|I - rho W|, so the marginal inherits that
        # error -- about 1e-5 at order 16 here -- and nothing worse.
        for s in (:sar, :sdm)
            @test log_marginal_likelihood(s, y5, X5, W5; method = ExactLU()) ≈
                  log_marginal_likelihood(s, y5, X5, W5;
                      method = ChebyshevApprox(order = 16, n_probes = 0)) atol = 1e-4
        end
    end

    @testset "30e. posterior probabilities" begin
        r = bayes_spec_probs(y5, X5, [W5])
        @test sum(r.prob) ≈ 1.0
        @test sum(r.prob_spec) ≈ 1.0
        @test sum(r.prob_w) ≈ 1.0
        @test r.prob_w[1] ≈ 1.0
        @test r.best[1] == r.specs[argmax(r.prob_spec)]
        @test r.prob_global + r.prob_local +
              r.prob_spec[findfirst(==(:ols), r.specs)] ≈ 1.0
        # The probabilities are the normalised marginals, nothing else.
        lm = [log_marginal_likelihood(s, y5, X5, W5) for s in r.specs]
        p = exp.(lm .- maximum(lm)); p ./= sum(p)
        @test r.prob_spec ≈ p atol = 1e-10
        @test vec(r.log_marginal) ≈ lm atol = 1e-10

        # Two weight matrices: the margins have to be the row and column
        # sums of the same table, and the winner its largest cell.
        W5b = row_standardize(make_queen_weights(5))
        r2 = bayes_spec_probs(y5, X5, [W5, W5b]; w_names = ["rook", "queen"])
        @test size(r2.prob) == (6, 2)
        @test sum(r2.prob) ≈ 1.0
        @test r2.prob_spec ≈ vec(sum(r2.prob; dims = 2))
        @test r2.prob_w ≈ vec(sum(r2.prob; dims = 1))
        idx = argmax(r2.prob)
        @test r2.best == (r2.specs[idx[1]], r2.w_names[idx[2]])
        @test occursin("P(global)", sprint(show, r2))
    end

    @testset "30f. the generating specification is recovered" begin
        # The two structures that are actually hard to tell apart:
        # global spillovers through y against local ones through the
        # disturbance, both carrying the same Durbin terms. One draw
        # proves nothing on its own -- `docs/AUDIT.md` records the
        # selection rates over repeated samples -- but it pins the
        # winner on a sample that cannot change.
        r_sdm = bayes_spec_probs(y16_sdm, X16, [W16])
        @test r_sdm.best[1] == :sdm
        @test r_sdm.prob_spec[findfirst(==(:sdm), r_sdm.specs)] > 0.9
        @test r_sdm.prob_global > 0.9

        y_sdem = mu16 .+ ((I - 0.6 * W16) \ e16)
        r_sdem = bayes_spec_probs(y_sdem, X16, [W16])
        @test r_sdem.best[1] == :sdem
        @test r_sdem.prob_spec[findfirst(==(:sdem), r_sdem.specs)] > 0.9
        @test r_sdem.prob_local > 0.9

        # Independent errors and no feedback: the SLX specification,
        # which both Durbin alternatives nest at their parameter's zero.
        r_slx = bayes_spec_probs(mu16 .+ e16, X16, [W16])
        @test r_slx.best[1] == :slx

        # And no spillovers at all, where the Durbin block is noise.
        r_ols = bayes_spec_probs(X16 * [1.0, -0.8] .+ e16, X16, [W16])
        @test r_ols.best[1] == :ols
    end

    @testset "30g. the prior on the spatial parameter does not wash out" begin
        # A marginal likelihood is not a likelihood ratio: the prior is
        # over a parameter the flat specifications do not have, so
        # widening its support penalises the spatial ones and leaves
        # :ols and :slx untouched. The size of the penalty is exactly
        # the width ratio whenever the posterior mass sits inside both
        # intervals, which for `y5` -- generated at rho = 0.5 -- it
        # does: halving the support must add log 2 and nothing else.
        wide = log_marginal_likelihood(:sar, y5, X5, W5;
                                       rho_bounds = (-0.995, 0.995))
        half = log_marginal_likelihood(:sar, y5, X5, W5;
                                       rho_bounds = (0.0, 0.995))
        # The residual is the posterior mass below zero that the half
        # interval drops, which at N = 25 is not quite nothing.
        @test half - wide ≈ log(1.99 / 0.995) atol = 1e-5
        for s in (:ols, :slx)
            @test log_marginal_likelihood(s, y5, X5, W5;
                      rho_bounds = (-0.995, 0.995)) ==
                  log_marginal_likelihood(s, y5, X5, W5;
                      rho_bounds = (0.0, 0.5))
        end
        # A Beta prior concentrates the same support differently, so it
        # has to produce a different answer -- if it did not, the prior
        # argument would not be reaching the integrand.
        bp = bayes_spec_probs(y5, X5, [W5]; prior = :beta, beta_shape = 3.0)
        un = bayes_spec_probs(y5, X5, [W5])
        @test bp.prob_spec != un.prob_spec
        @test bp.prior === :beta
    end

    @testset "30h. the SDEM roadmap" begin
        W5b = row_standardize(make_queen_weights(5))
        r = sdem_roadmap(y5, X5, [W5, W5b]; w_names = ["rook", "queen"])
        @test r.specs == [:sdem, :slx, :sem]
        @test size(r.score) == (2, 3)
        @test r.criterion === :marginal
        for i in 1:2
            row = r.score[i, :]
            o = sortperm(row; rev = true)
            @test r.best_spec[i] == r.specs[o[1]]
            @test r.margin[i] ≈ row[o[1]] - row[o[2]]
            @test r.score[i, :] ≈ [log_marginal_likelihood(s, y5, X5, [W5, W5b][i])
                                   for s in r.specs] atol = 1e-10
        end
        flat = argmax(r.score)
        @test r.best_w == r.w_names[flat[1]]
        @test r.best_overall == r.specs[flat[2]]
        @test occursin("roadmap", sprint(show, r))

        # AIC and BIC are minimised, not maximised, and the search has
        # to read them that way; BIC penalises the wider design harder,
        # so the two need not agree, but each must agree with itself.
        for c in (:aic, :bic)
            rc = sdem_roadmap(y5, X5, [W5, W5b]; criterion = c,
                              w_names = ["rook", "queen"])
            @test rc.criterion === c
            for i in 1:2
                o = sortperm(rc.score[i, :])
                @test rc.best_spec[i] == rc.specs[o[1]]
                @test rc.margin[i] ≈ rc.score[i, o[2]] - rc.score[i, o[1]]
            end
            @test rc.best_overall == rc.specs[argmin(rc.score)[2]]
        end
        # One specification leaves nothing to compare against.
        r1 = sdem_roadmap(y5, X5, [W5]; specs = [:sdem])
        @test r1.margin == [Inf]
        @test r1.best_overall === :sdem
    end

    @testset "30i. what the entry points reject" begin
        @test_throws ArgumentError log_marginal_likelihood(:sarar, y5, X5, W5)
        @test_throws ArgumentError log_marginal_likelihood(:ols, y5, X5, W5;
                                                           prior = :normal)
        @test_throws ArgumentError log_marginal_likelihood(:sar, y5, X5, W5;
                                                           rho_bounds = (0.9, -0.9))
        @test_throws ArgumentError bayes_spec_probs(y5, X5, [W5]; specs = Symbol[])
        @test_throws ArgumentError bayes_spec_probs(y5, X5,
                                       SparseMatrixCSC{Float64, Int}[])
        @test_throws ArgumentError bayes_spec_probs(y5, X5, [W5];
                                                    w_names = ["a", "b"])
        @test_throws ArgumentError sdem_roadmap(y5, X5, [W5]; criterion = :mse)

        # The identification guard of testset 29 reaches this module too,
        # and a Durbin specification can be deficient when the plain one
        # is not: `W*X` collapses into the span of `X` whenever `W` maps
        # the covariates onto themselves.
        Xc = hcat(ones(N5) .* 2.0, Float64[sin(i / 2) for i in 1:N5])
        err = try
            log_marginal_likelihood(:ols, y5, Xc, W5); nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("not identified", err.msg)
        @test occursin("log_marginal_likelihood:", err.msg)

        # n > k, or there is no marginal to speak of.
        W2 = row_standardize(make_rook_weights(2))
        X2 = hcat(Float64[sin(i) for i in 1:4], Float64[cos(i) for i in 1:4],
                  Float64[sin(2i) for i in 1:4])
        @test_throws ArgumentError log_marginal_likelihood(:ols,
            Float64[1.0, 2.0, 3.0, 4.0], X2, W2)
    end
end
