# ---------------------------------------------------------------------------
# 33. Spatial log-GARCH: spatial dependence in the variance
#
# y_i = √h_i ε_i,  log h = λW log h + ρW log y² + α + zδ   (Sato & Matsuda 2021)
#
# A unit's volatility depends on its neighbours' volatility (λ) and on their
# squared outcomes (ρ). With Y = log y² the model is a spatial ARMA(1,1),
# R(λ+ρ)Y = Xβ + R(λ)V, so the mean of Y identifies only λ + ρ. λ alone is
# identified through the error part R(λ)V, and is estimated less precisely.
#
# Run: julia --project=. examples/33_spatial_log_garch.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, coefnames, stderror

const RNG = MersenneTwister(20_261_116)

# ── 1. Simulate on a 30 × 30 rook lattice ──────────────────────────────────
#
# λ = 0.45, ρ = 0.45, α = 0.5, δ = 1, normal ε. Simulate Y from the SARMA form.

L = 30; n = L^2
W = make_rook_weights(L)
λ, ρ, α, δ = 0.45, 0.45, 0.5, 1.0
z = randn(RNG, n)
ε = randn(RNG, n)
Y = lu(sparse(I - (λ + ρ) * W)) \ (α .+ δ .* z .+ (I - λ * W) * log.(ε .^ 2))
y = sqrt.(exp.(Y)) .* sign.(ε)

# ── 2. Fit ──────────────────────────────────────────────────────────────────

ev = eigvals(Matrix(W))              # reusable across fits on this W
m = fit_spatial_log_garch(y, W; Z = z, eigenvalues = ev)
println(m)
@printf("True values: δ = %.2f, λ = %.2f, ρ = %.2f, α = %.2f\n", δ, λ, ρ, α)

# ── 3. Why the sandwich ────────────────────────────────────────────────────
#
# With normal ε, log ε² is log χ²₁: skewed (−2.1) with excess kurtosis 4.
# The Gaussian information −Ĥ⁻¹ assumes it is normal; the sandwich does not.

println("\nStandard errors: sandwich against Gaussian information")
seg = sqrt.(diag(m.vcov_gaussian))
for (k, nm) in enumerate(coefnames(m))
    @printf("  %-3s  sandwich %.4f   Gaussian %.4f\n", nm, stderror(m)[k], seg[k])
end

# ── 4. λ against λ + ρ ─────────────────────────────────────────────────────
#
# The estimates of λ and ρ trade against each other; their sum is estimated
# far more precisely than either.

Vλρ = m.vcov[2:3, 2:3]
@printf("\ncorr(λ̂, ρ̂) = %.3f\n", Vλρ[1, 2] / sqrt(Vλρ[1, 1] * Vλρ[2, 2]))
@printf("SE(λ̂) = %.3f, SE(ρ̂) = %.3f, SE(λ̂ + ρ̂) = %.3f\n", sqrt(Vλρ[1, 1]),
        sqrt(Vλρ[2, 2]), sqrt(sum(Vλρ)))

# ── 5. Fitted volatility ───────────────────────────────────────────────────

logh_true = Y .- log.(ε .^ 2)
@printf("\ncor(log ĥ, true log h) = %.3f\n", cor(m.log_volatility, logh_true))

# ── 6. A covariate-distance W ──────────────────────────────────────────────
#
# Suppose locations are unknown but two covariates track them with noise.
# The true W is 10-nearest-neighbour on locations; covariate_weights builds
# the proxy from the noisy covariates. The proxy keeps only part of the
# true neighbours, and ρ̂ is biased towards zero. One sample shows only one
# draw; validation/mc_spatial_log_garch.jl (design C) measures the bias.

S = rand(RNG, n, 2)
Wtrue = knn_weights(S, 10; binary = true)
Yk = lu(sparse(I - (λ + ρ) * Wtrue)) \ (α .+ δ .* z .+ (I - λ * Wtrue) * log.(randn(RNG, n) .^ 2))
yk = sqrt.(exp.(Yk))
println("\nTrue W = 10-NN on locations; fitted with the true W and two proxies:")
@printf("  %-22s %8s %8s %8s\n", "W", "shared", "λ̂", "ρ̂")
for (label, Wfit) in (("true", Wtrue),
                      ("covariates, noise .01", covariate_weights(S .+ 0.01 .* randn(RNG, n, 2), 10)),
                      ("covariates, noise .05", covariate_weights(S .+ 0.05 .* randn(RNG, n, 2), 10)))
    shared = sum((Wfit .> 0) .& (Wtrue .> 0)) / sum(Wtrue .> 0)
    mk = fit_spatial_log_garch(yk, Wfit; Z = z)
    @printf("  %-22s %8.2f %8.3f %8.3f\n", label, shared, mk.lambda, mk.rho)
end
println("  (true λ = ρ = 0.45; \"shared\" is the share of true neighbours the W keeps)")
