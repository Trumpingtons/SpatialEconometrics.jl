# ---------------------------------------------------------------------------
# 32. Short-panel spatial dynamics with common shocks
#
# `y_t = ρWy_t + φy_{t−1} + X_tβ + 1δ_t + Λf_t + ε_t`, t = 1, …, T, with T
# small (here 5), N large, and r = 2 common shocks f_t whose effect differs by
# unit through the loadings λ_i. Bai & Chen (2026), a preprint.
#
# With T fixed, the N loadings cannot be estimated one by one. Bai & Chen
# project them on observed controls instead, λ_i = Az_i + η_i ("spatially
# enriched random loadings", SERL). The controls z_i stack the unit means of
# X, the initial outcome y₀, and their spatial lags up to order L. The
# likelihood is conditional on y₀, so it is right only if the controls carry
# everything y₀ says about λ_i. When y₀ comes out of a long-running spatial
# process, it carries the neighbours' loadings too, and L is the question.
#
# Run: julia --project=. examples/32_common_shocks_short_panel.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, coefnames, stderror

const RNG = MersenneTwister(20_261_114)

# ── 1. Simulate Bai & Chen's long-running design (their §6.2.1) ────────────
#
# 8-nearest-neighbour W on random locations. The loadings λ_i ~ N(0, I₂) also
# shift the regressors, x_is = 0.5λ_i + √0.75 h_is with h an AR(1). The
# outcome runs 100 periods from zero before y₀ is taken, then T = 5 more.
# ρ = 0.5, φ = 0.3, β = (0.8, −0.3), no time effects, unit-variance errors.

N, T, ρ, φ, β = 1000, 5, 0.5, 0.3, [0.8, -0.3]
W = knn_weights(rand(RNG, N, 2), 8)

function simulate(rng; presample = 100)
    B = lu(sparse(I - ρ * W))
    λ = randn(rng, N, 2); h = randn(rng, N, 2); y = zeros(N)
    ys = Vector{Float64}[]; xs = Matrix{Float64}[]
    for s in 1:(presample + T)
        h = 0.8 .* h .+ 0.6 .* randn(rng, N, 2)
        x = 0.5 .* λ .+ sqrt(0.75) .* h
        y = B \ (φ .* y .+ x * β .+ λ * randn(rng, 2) .+ randn(rng, N))
        s >= presample && (push!(ys, y); push!(xs, s == presample ? zeros(N, 2) : x))
    end
    return reduce(vcat, ys), reduce(vcat, xs)   # period 0 first, units fastest
end

y, X = simulate(RNG)
ev = eigvals(Matrix(W))       # reused by every fit below

# ── 2. The fit with L = 1 ──────────────────────────────────────────────────

fit1 = fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = 1,
                                      eigenvalues = ev, varnames = ["x1", "x2"])
println(fit1)
println("SERL controls: ", join(fit1.controlnames, ", "))
println("Estimated factor path F = [I₂; F₂] (rows are periods):")
for t in 1:T
    @printf("  t = %d   %7.3f %7.3f\n", t, fit1.F[t, 1], fit1.F[t, 2])
end

# ── 3. L = 0 against L = 1 ─────────────────────────────────────────────────
#
# L = 0 uses each unit's own mean of X and own y₀. Its y₀ also carries the
# neighbours' loadings through (I − ρW)⁻¹, which L = 0 leaves in η_i, so
# Assumption 2 fails.

fit0 = fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = 0,
                                      eigenvalues = ev, varnames = ["x1", "x2"])
truth = vcat(ρ, φ, β)
println("\nThe same panel, two SERL orders (true values in the last column):")
@printf("  %-5s %8s %7s %8s %7s %6s\n", "", "L = 0", "SE", "L = 1", "SE", "true")
for i in 1:4
    @printf("  %-5s %8.3f %7.3f %8.3f %7.3f %6.2f\n", coefnames(fit1)[i], coef(fit0)[i],
            stderror(fit0)[i], coef(fit1)[i], stderror(fit1)[i], truth[i])
end
println("""
  One panel shows sampling noise as well as bias. The Monte Carlo in
  validation/mc_serl_sdpd.jl runs this design 500 times at each N; see the
  manual page for what it found.""")

# ── 3b. Q: what summarises the regressors ──────────────────────────────────
#
# The default Q_X is each unit's mean of x over periods 1..T. Here
# x_it = 0.5λ_i + √0.75 h_it with h an AR(1), so the best predictor of λ_i
# from x_i1, …, x_iT weights the periods unequally and the mean leaves part of
# λ_i in η_i, correlated with X. That biases β̂ by an amount that does not
# shrink with N: in this design about +.009 and −.010, measured over 200
# panels at N = 2000 (see the manual page). Passing every period of x as Q
# removes it. The bias is about half an SE at N = 2000, so one panel cannot
# show it; the table only shows that the alternative fits.

Xp = [reshape(X[:, k], N, T + 1)[:, 2:end] for k in 1:2]     # N×T per regressor
fitQ = fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = 1, eigenvalues = ev,
                                      Q = hcat(Xp...), varnames = ["x1", "x2"])
println("\nL = 1 with Q = unit means and with Q = every period of x:")
@printf("  %-5s %8s %7s %8s %7s %6s\n", "", "means", "SE", "periods", "SE", "true")
for i in 1:4
    @printf("  %-5s %8.3f %7.3f %8.3f %7.3f %6.2f\n", coefnames(fit1)[i], coef(fit1)[i],
            stderror(fit1)[i], coef(fitQ)[i], stderror(fitQ)[i], truth[i])
end
println("""
  The bias the means leave is about half an SE at N = 2000, and less at this
  N, so one panel cannot show it. validation/mc_serl_sdpd_diagnostics.jl
  measures it over 200 panels.""")

# ── 4. The pair correction ─────────────────────────────────────────────────
#
# The ρ and φ scores of one unit depend on every other unit's errors through
# (I − ρW)⁻¹. The covariance keeps those pair terms; the preferred version
# scales them by N/(N − p), p the number of parameters, nuisance ones
# included. It is a finite-sample device that Bai & Chen do not claim is
# unbiased.

p = length(fit1.alpha)
@printf("\np = %d parameters, so N/(N − p) = %.3f\n", p, N / (N - p))
@printf("  %-5s %12s %12s\n", "", "SE uncorr.", "SE pair×")
for i in 1:4
    @printf("  %-5s %12.4f %12.4f\n", coefnames(fit1)[i], sqrt(fit1.vcov_uncorrected[i, i]),
            stderror(fit1)[i])
end
