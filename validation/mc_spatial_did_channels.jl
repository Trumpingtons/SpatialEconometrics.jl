# Monte Carlo for fit_spatial_did(...; channels = :split), Chagas, Azzoni &
# Almeida (2016) eq. (7): the unrestricted two-channel spatial DiD, fitted by
# OLS (`errors = :iid`, default `vcov = :cluster`) and by spatial-error ML
# (`errors = :spatial`).
#
# Questions:
#   1. Does the Wald test of equal channels (`spillover_equality`) hold its
#      size when the channels are equal, and does it have power when they
#      are not?
#   2. Do the four effects' intervals cover, under spatially uncorrelated,
#      correlated and heteroskedastic errors, for each error model?
#   3. What does the pooled model report for the control spillover when the
#      channels differ (the bias the split removes)?
#   4. How imprecise is the direct effect `τ` once the treated channel is
#      free, against the total effect on the treated?
#
# A. Two-period, a 20 x 20 rook lattice (N = 400), errors (I - ρW)^-1 ε in
#    each period, ρ in {0, 0.6}. Two treatment layouts, fixed across
#    replications: a contiguous block of 64 units and a scattered 30% draw.
#    ε is N(0,1), or heteroskedastic with sd 2 on the treated. τ = 1,
#    δ_C = 0.8, δ_T in {0.8, 0} (null and alternative).
# B. A panel in the shape of Chagas et al.'s own simulation (their §5,
#    eq. 9): N = 200 units at normal pseudo-coordinates, k = 10 nearest
#    neighbours, T = 10, treatment redrawn as Bernoulli(0.5) every period,
#    unit and period effects, a covariate x and its lag Wx (coefficients 1
#    and 1, passed in X), a spatial error (I - λW)^-1 ε with λ in {0, 0.5},
#    α = 1, b1 = 0 and b2 in {0.1, 0.4, 0.8}, plus b1 = b2 = 0.4 for the
#    test's size. Their Table 7 is a reference for the design, not a target
#    for the numbers: they report means and intervals of their ML fits.
#
# Run: julia --project=. validation/mc_spatial_did_channels.jl
#      (MC_REPS=50 for a pilot)
# Output: validation/mc_spatial_did_channels_output.txt (written by the script)
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror

const R = parse(Int, get(ENV, "MC_REPS", "2000"))
const Z95 = 1.959963984540054
const OUT = joinpath(@__DIR__, "mc_spatial_did_channels_output.txt")
io = open(OUT, "w")
say(s) = (println(s); println(io, s); flush(io))
t_start = time()

say("# fit_spatial_did, channels = :split — Monte Carlo ($R replications per cell)")
say("Julia $(VERSION). Coverage is of nominal 95% intervals; its Monte Carlo sd at R = $R is " *
    @sprintf("%.4f", sqrt(0.95 * 0.05 / R)) * ". `rej` is the equality test's rejection rate at 5%. " *
    "`iid` is OLS with the default `vcov = :cluster`; `spatial` is `errors = :spatial`.")
say("")

# ── A. Two-period lattice ─────────────────────────────────────────────────
side = 20; N = side^2
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
W = make_rook_weights(side); Wt = row_standardize(W)
μ = collect(range(-1.0, 1.0, length = N))
layouts = Dict(
    "block" => [(7 <= coords[i, 1] <= 14 && 7 <= coords[i, 2] <= 14) ? 1.0 : 0.0 for i in 1:N],
    "scattered" => (r = MersenneTwister(7); [rand(r) < 0.3 ? 1.0 : 0.0 for _ in 1:N]),
)

say("## A. Two-period, 20 x 20 rook lattice, errors (I − ρW)⁻¹ε")
say("")
say("| layout | ρ | ε | δ_T | errors | rej | cov τ | cov spill T | cov spill C | cov total T | sd τ̂ / sd total | spill C bias: pooled | split |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for lay in ("block", "scattered"), (ρe, het) in ((0.0, false), (0.6, false), (0.6, true)),
    δT in (0.8, 0.0), errs in (:iid, :spatial)
    D = layouts[lay]
    E = Wt * D
    ĒT = mean(E[D .== 1]); ĒC = mean(E[D .== 0])
    τ, γ, δC = 1.0, 0.5, 0.8
    truth = (τ, δT * ĒT, δC * ĒC, τ + δT * ĒT)
    Afac = lu(sparse(1.0I, N, N) - ρe * W)
    sdv = het ? (1 .+ D) : ones(N)
    rng = MersenneTwister(20261119)
    rej = 0; cov = zeros(Int, 4); dir = Float64[]; tot = Float64[]; pool_c = Float64[]; split_c = Float64[]
    for _ in 1:R
        u0 = Afac \ (sdv .* randn(rng, N)); u1 = Afac \ (sdv .* randn(rng, N))
        y = hcat(μ .+ u0, μ .+ γ .+ τ .* D .+ δT .* D .* E .+ δC .* (1 .- D) .* E .+ u1)
        m = fit_spatial_did(y, D, W; channels = :split, errors = errs)
        rej += m.spillover_equality.p_value < 0.05
        est = (m.att_direct, m.att_spill_treated, m.att_spill_control, m.att_total_treated)
        se = (m.att_direct_se, m.att_spill_treated_se, m.att_spill_control_se, m.att_total_treated_se)
        for k in 1:4
            cov[k] += abs(est[k] - truth[k]) <= Z95 * se[k]
        end
        push!(dir, m.att_direct); push!(tot, m.att_total_treated); push!(split_c, m.att_spill_control)
        push!(pool_c, fit_spatial_did(y, D, W; errors = errs).att_spill_control)
    end
    say(@sprintf("| %s | %.1f | %s | %.1f | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %+.4f | %+.4f |",
                 lay, ρe, het ? "het." : "N(0,1)", δT, errs, rej / R, (cov ./ R)...,
                 std(dir) / std(tot), mean(pool_c) - truth[3], mean(split_c) - truth[3]))
end
say("")
for lay in ("block", "scattered")
    D = layouts[lay]; rep = check_interference(exposure(D, W))
    say(@sprintf("%s: %d treated, %d treated with no treated neighbour, var(exposure | treated) = %.4f, cor(D, D×WD) = %.3f, mean exposure treated %.3f / controls %.3f.",
                 lay, rep.n_treated, rep.n_treated_unexposed, rep.exposure_var_treated,
                 cor(D, D .* (Wt * D)), rep.mean_exposure_treated, rep.mean_exposure_control))
end
say("")

# ── B. Chagas et al.'s simulation design ─────────────────────────────────
say("## B. Chagas et al. §5 design: N = 200, kNN (k = 10), T = 10, D_it ~ Bernoulli(0.5)")
say("")
say("| λ | b1 | b2 | errors | mean b̂1 | mean b̂2 | sd b̂2 | cov α | cov b1 | cov b2 | rej |")
say("|---|---|---|---|---|---|---|---|---|---|---|")
NB, TB = 200, 10
rngc = MersenneTwister(11)
cB = randn(rngc, NB, 2)
WB = knn_weights(cB, 10)
WBt = row_standardize(WB)
for λe in (0.0, 0.5), (b1, b2) in ((0.0, 0.1), (0.0, 0.4), (0.0, 0.8), (0.4, 0.4)), errs in (:iid, :spatial)
    AB = lu(sparse(1.0I, NB, NB) - λe * WB)
    rng = MersenneTwister(20261120)
    α = 1.0
    cnt = zeros(Int, 3); rej = 0; B1 = Float64[]; B2 = Float64[]
    for _ in 1:R
        μB = randn(rng, NB); λB = randn(rng, TB)
        Dp = Float64.(rand(rng, NB, TB) .< 0.5)
        x = randn(rng, NB, TB)
        X = zeros(NB, 2, TB)
        y = zeros(NB, TB)
        for t in 1:TB
            Et = WBt * Dp[:, t]
            wx = WBt * x[:, t]
            X[:, 1, t] = x[:, t]; X[:, 2, t] = wx
            y[:, t] = μB .+ λB[t] .+ α .* Dp[:, t] .+ b1 .* Dp[:, t] .* Et .+
                      b2 .* (1 .- Dp[:, t]) .* Et .+ x[:, t] .+ wx .+ (AB \ randn(rng, NB))
        end
        m = fit_spatial_did(y, Dp, WB; X = X, channels = :split, errors = errs)
        θ = coef(m); s = stderror(m)
        # coefficient order: D, D×WD, (1-D)×WD, X1, X2
        cnt[1] += abs(θ[1] - α) <= Z95 * s[1]
        cnt[2] += abs(θ[2] - b1) <= Z95 * s[2]
        cnt[3] += abs(θ[3] - b2) <= Z95 * s[3]
        rej += m.spillover_equality.p_value < 0.05
        push!(B1, θ[2]); push!(B2, θ[3])
    end
    say(@sprintf("| %.1f | %.1f | %.1f | %s | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f | %.3f |",
                 λe, b1, b2, errs, mean(B1), mean(B2), std(B2), (cnt ./ R)..., rej / R))
end
say("")
say(@sprintf("Run time %.0f s.", time() - t_start))
close(io)
