# ---------------------------------------------------------------------------
# 31. Short-panel spatial dynamics with correlated random effects
#
# `y_t = ρy_{t−1} + λ₁W₁y_t + λ₂W₂y_{t−1} + X_tβ + Zγ + μ + α_t + u_t`,
# `u_t = λ₃W₃u_t + v_t`, `μ = X̄π + ε`, observed for t = 0, 1, …, T with T
# small (here 3). Li & Yang (2021).
#
# With T fixed, y₀ is endogenous: the process ran for m periods before it was
# first observed, so y₀ carries ε. The conditional QMLE treats y₀ as given and
# is inconsistent. The M-estimator corrects each biased score component by its
# expectation, which depends on y₀ only through m. So m is an assumption the
# user states, not an estimate; section 3 shows what a wrong m costs here.
#
# The standard errors are a sandwich whose middle, Var(S*), is a corrected
# plug-in estimate. It is valid for non-normal errors. It is not robust to
# heteroskedasticity, and neither is the estimator.
#
# Run: julia --project=. examples/31_cre_dynamic_panel.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, Random, Statistics, Printf
import StatsAPI: coef, coefnames, stderror

const RNG = MersenneTwister(20_261_113)

# ── 1. Simulate Li & Yang's design (their Table 1, DGP1) ───────────────────
#
# Queen contiguity for W₁ = W₃ on a 10 × 10 grid, group interaction for W₂,
# β = γ = π = σ²_v = φ = 1, ρ = 0.3, λ₁ = λ₂ = λ₃ = 0.2, m = 10. X_t is
# persistent, as in their footnote 7: a unit mean plus an ARMA(1,1). The errors
# are χ²(3), centred and scaled, so they are skewed (μ3 = 1.63) and
# heavy-tailed (μ4 = 7). This is one cell of validation/mc_cre_sdpd.jl.

L = 10; n = L^2; T = 3; m_true = 10
W1 = Matrix(make_queen_weights(L)); W3 = W1
W2 = zeros(n, n)
for b in 0:(L - 1), i in 1:L, j in 1:L
    i == j && continue
    W2[b*L+i, b*L+j] = 1 / (L - 1)
end

function simulate(rng; ρ = 0.3, λ1 = 0.2, λ2 = 0.2, λ3 = 0.2, m = m_true, burn = 50)
    draw(k) = (sum(randn(rng, k, 3) .^ 2; dims = 2)[:, 1] .- 3) ./ sqrt(6)
    # X_t = μ_x + 0.01t + ζ_t, (1 − 0.5L)ζ_t = ϵ_t + 0.5ϵ_{t−1}, ϵ_t ~ N(0, 4)
    ϵ = [2 .* randn(rng, n) for _ in 1:(burn + m + T + 1)]
    ζ = zeros(n); Xs = Vector{Vector{Float64}}()
    for (j, t) in enumerate(-(m + burn) + 1:T)
        ζ = 0.5 .* ζ .+ ϵ[j+1] .+ 0.5 .* ϵ[j]
        t >= -m && push!(Xs, ζ)
    end
    μx = randn(rng, n) .+ sum(ϵ[burn+1:end]) ./ (T + m + 1)
    Xs = [μx .+ 0.01t .+ Xs[i] for (i, t) in enumerate(-m:T)]      # X_{−m}, …, X_T
    Xbar = sum(Xs[m+1:end]) ./ (T + 1)                              # mean over 0..T
    z = Float64.(rand(rng, n) .< 0.5)
    μ = Xbar .+ draw(n)                                             # μ = X̄π + ε
    α = vcat(ones(m), randn(rng, T) .+ 1)
    B1 = I - λ1 * W1; B3 = I - λ3 * W3
    y = zeros(n); ys = Vector{Vector{Float64}}()
    for (j, t) in enumerate(-m+1:T)
        y = B1 \ (ρ .* y .+ λ2 .* (W2 * y) .+ Xs[j+1] .+ z .+ μ .+ α[j] .+ (B3 \ draw(n)))
        t >= 0 && push!(ys, y)
    end
    return reduce(vcat, ys), reshape(reduce(vcat, Xs[m+1:end]), :, 1), reshape(z, :, 1)
end

y, X, Z = simulate(RNG)

# ── 2. The M-estimator, beside the conditional QMLE ────────────────────────

fit = fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = m_true, W2 = W2, W3 = W3,
                                    varnames = ["x"], zvarnames = ["z"], compute_cqmle = true)
println(fit)

println("\nM-estimator against the conditional QMLE (true values in the last column):")
truth = Dict("x" => 1.0, "z" => 1.0, "mean_x" => 1.0, "sigma2_v" => 1.0, "phi" => 1.0,
             "rho" => 0.3, "lambda1" => 0.2, "lambda2" => 0.2, "lambda3" => 0.2)
se = stderror(fit); seh = sqrt.(diag(fit.vcov_hessian))
@printf("  %-9s %8s %8s %9s %8s %6s %6s\n", "", "CQMLE", "M-est", "sandwich", "Hessian", "ratio", "true")
for (i, nm) in enumerate(coefnames(fit))
    haskey(truth, nm) || continue
    @printf("  %-9s %8.3f %8.3f %9.3f %8.3f %6.2f %6.2f\n", nm, fit.cqmle[i], coef(fit)[i],
            se[i], seh[i], se[i] / seh[i], truth[nm])
end
println("""
  One sample shows sampling noise as well as bias, and in this one the CQMLE's
  ρ̂ happens to be the closer to 0.3. The bias shows across samples: in
  validation/mc_cre_sdpd.jl, which runs this design 1000 times, the CQMLE
  averages ρ̂ = 0.350 and π̂ = 0.860, and the M-estimator 0.299 and 0.999.

  The sandwich and Hessian standard errors differ most for σ²_v and φ (the
  'ratio' column). Their scores are quadratic in the errors, so their variance
  carries the excess kurtosis of the χ² errors, which the Hessian ones ignore.""")
@printf("  Estimated shock moments: μ3ε = %.2f, μ4ε = %.2f, μ3v = %.2f, μ4v = %.2f\n",
        fit.moments...)
println("  (for these errors: μ3 = 1.63, μ4 = 7)")

# ── 3. The assumed m ───────────────────────────────────────────────────────
#
# Li & Yang report that the M-estimator is robust to a wrong m when the
# process is stationary, because 𝓑ᵐ dies out quickly. Refit with m = 2 and
# m = 50. A fit that finds no root says so instead of returning an estimate.

println("\nThe same data, a different assumed m:")
@printf("  %-6s %8s %8s %8s\n", "m", "ρ̂", "λ̂₁", "φ̂")
for mm in (2, m_true, 50)
    f = try
        fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = mm, W2 = W2, W3 = W3)
    catch err
        println("  ", mm, "      no root found")
        continue
    end
    b = Dict(zip(coefnames(f), coef(f)))
    @printf("  %-6d %8.3f %8.3f %8.3f\n", mm, b["rho"], b["lambda1"], b["phi"])
end
println("""
  With ρ = 0.3 and λ's of 0.2, 𝓑ᵐ is negligible by m = 10, so m = 10 and
  m = 50 agree to three decimals. Here even m = 2 moves ρ̂ by less than a
  tenth of its standard error. m = 0 would say y₀ is exogenous, and then the
  M-estimator is the conditional QMLE.""")
