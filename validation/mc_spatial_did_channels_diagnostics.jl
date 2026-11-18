# Why do the split model's intervals under-cover in mc_spatial_did_channels.jl?
#
# Coverage there was 0.67-0.86 in the block layout and the equality test
# rejected a true null 22% of the time. This script separates the candidate
# causes, at the null δ_T = δ_C = 0.8:
#   - the error's spatial correlation (ρ_err = 0.6 against 0);
#   - the split itself (pooled against split, on the same draws);
#   - the covariance (:cluster, :ols, Conley at cutoffs 1.5 and 2.5, and in
#     section A2 at 4, 6 and 9 for the block layout).
# Panel B repeats it for Chagas et al.'s design: spatial error 0.5 against
# 0, with unit-clustered, Driscoll-Kraay and Conley standard errors.
#
# Run: julia --project=. validation/mc_spatial_did_channels_diagnostics.jl
# Output: validation/mc_spatial_did_channels_diagnostics_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror

const R = parse(Int, get(ENV, "MC_REPS", "1000"))
const Z95 = 1.959963984540054
const OUT = joinpath(@__DIR__, "mc_spatial_did_channels_diagnostics_output.txt")
io = open(OUT, "w")
say(s) = (println(s); println(io, s); flush(io))
say("# Diagnostics for the split model's coverage ($R replications per cell, Julia $(VERSION))")
say("")

side = 20; N = side^2
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
W = make_rook_weights(side); Wt = row_standardize(W)
μ = collect(range(-1.0, 1.0, length = N))
layouts = Dict(
    "block" => [(7 <= coords[i, 1] <= 14 && 7 <= coords[i, 2] <= 14) ? 1.0 : 0.0 for i in 1:N],
    "scattered" => (r = MersenneTwister(7); [rand(r) < 0.3 ? 1.0 : 0.0 for _ in 1:N]),
)
vcovs = [("cluster", (;)), ("ols", (vcov = :ols,)),
         ("conley 1.5", (vcov = :conley, coords = coords, dist_cutoff = 1.5)),
         ("conley 2.5", (vcov = :conley, coords = coords, dist_cutoff = 2.5))]

say("## A. Two-period lattice, δ_T = δ_C = 0.8. Coverage of the four effects (direct, spill T, spill C, total T) and the equality test's rejection rate")
say("")
say("| layout | ρ_err | vcov | model | cov τ | cov spill T | cov spill C | cov total T | rej |")
say("|---|---|---|---|---|---|---|---|---|")
for lay in ("block", "scattered"), ρe in (0.0, 0.6)
    D = layouts[lay]; E = Wt * D
    ĒT = mean(E[D .== 1]); ĒC = mean(E[D .== 0])
    τ, γ, δ = 1.0, 0.5, 0.8
    truth = (τ, δ * ĒT, δ * ĒC, τ + δ * ĒT)
    Afac = lu(sparse(1.0I, N, N) - ρe * W)
    for (vn, kw) in vcovs, model in (:pooled, :split)
        rng = MersenneTwister(20261121)
        cov = zeros(Int, 4); rej = 0
        for _ in 1:R
            u0 = Afac \ randn(rng, N); u1 = Afac \ randn(rng, N)
            y = hcat(μ .+ u0, μ .+ γ .+ τ .* D .+ δ .* E .+ u1)
            m = fit_spatial_did(y, D, W; channels = model, kw...)
            est = (m.att_direct, m.att_spill_treated, m.att_spill_control, m.att_total_treated)
            se = (m.att_direct_se, m.att_spill_treated_se, m.att_spill_control_se, m.att_total_treated_se)
            for k in 1:4
                cov[k] += abs(est[k] - truth[k]) <= Z95 * se[k]
            end
            model === :split && (rej += m.spillover_equality.p_value < 0.05)
        end
        say(@sprintf("| %s | %.1f | %s | %s | %.3f | %.3f | %.3f | %.3f | %s |", lay, ρe, vn, model,
                     (cov ./ R)..., model === :split ? @sprintf("%.3f", rej / R) : "—"))
    end
end
say("")

say("## A2. Wider Conley cutoffs: block layout, ρ_err = 0.6. `SE/sd` is the mean reported SE of `att_total_treated` over its Monte Carlo sd")
say("")
say("| cutoff | model | cov τ | cov spill T | cov spill C | cov total T | rej | SE/sd total |")
say("|---|---|---|---|---|---|---|---|")
let D = layouts["block"], E = Wt * D
    ĒT = mean(E[D .== 1]); ĒC = mean(E[D .== 0])
    τ, γ, δ = 1.0, 0.5, 0.8
    truth = (τ, δ * ĒT, δ * ĒC, τ + δ * ĒT)
    Afac = lu(sparse(1.0I, N, N) - 0.6 * W)
    for cut in (4.0, 6.0, 9.0), model in (:pooled, :split)
        rng = MersenneTwister(20261121)
        cov = zeros(Int, 4); rej = 0; tot = Float64[]; tse = Float64[]
        for _ in 1:R
            u0 = Afac \ randn(rng, N); u1 = Afac \ randn(rng, N)
            y = hcat(μ .+ u0, μ .+ γ .+ τ .* D .+ δ .* E .+ u1)
            m = fit_spatial_did(y, D, W; channels = model, vcov = :conley,
                                coords = coords, dist_cutoff = cut)
            est = (m.att_direct, m.att_spill_treated, m.att_spill_control, m.att_total_treated)
            se = (m.att_direct_se, m.att_spill_treated_se, m.att_spill_control_se, m.att_total_treated_se)
            for k in 1:4
                cov[k] += abs(est[k] - truth[k]) <= Z95 * se[k]
            end
            model === :split && (rej += m.spillover_equality.p_value < 0.05)
            push!(tot, m.att_total_treated); push!(tse, m.att_total_treated_se)
        end
        say(@sprintf("| %.1f | %s | %.3f | %.3f | %.3f | %.3f | %s | %.3f |", cut, model, (cov ./ R)...,
                     model === :split ? @sprintf("%.3f", rej / R) : "—", mean(tse) / std(tot)))
    end
end
say("")

say("## B. Chagas et al. design (N = 200, kNN k = 10, T = 10), b1 = b2 = 0.4")
say("")
say("| λ_err | vcov | cov α | cov b1 | cov b2 | rej |")
say("|---|---|---|---|---|---|")
NB, TB = 200, 10
rngc = MersenneTwister(11)
cB = randn(rngc, NB, 2)
WB = knn_weights(cB, 10); WBt = row_standardize(WB)
vB = [("cluster (unit)", (;)),
      ("Driscoll-Kraay", (vcov = DriscollKraayVCOV(),)),
      ("conley 0.3", (vcov = :conley, coords = cB, dist_cutoff = 0.3)),
      ("conley 0.6", (vcov = :conley, coords = cB, dist_cutoff = 0.6))]
for λe in (0.0, 0.5)
    AB = lu(sparse(1.0I, NB, NB) - λe * WB)
    for (vn, kw) in vB
        rng = MersenneTwister(20261120)
        cnt = zeros(Int, 3); rej = 0
        for _ in 1:R
            μB = randn(rng, NB); λB = randn(rng, TB)
            Dp = Float64.(rand(rng, NB, TB) .< 0.5)
            x = randn(rng, NB, TB)
            X = zeros(NB, 2, TB); y = zeros(NB, TB)
            for t in 1:TB
                Et = WBt * Dp[:, t]; wx = WBt * x[:, t]
                X[:, 1, t] = x[:, t]; X[:, 2, t] = wx
                y[:, t] = μB .+ λB[t] .+ Dp[:, t] .+ 0.4 .* Et .+ x[:, t] .+ wx .+ (AB \ randn(rng, NB))
            end
            m = fit_spatial_did(y, Dp, WB; X = X, channels = :split, kw...)
            θ = coef(m); s = stderror(m)
            cnt[1] += abs(θ[1] - 1.0) <= Z95 * s[1]
            cnt[2] += abs(θ[2] - 0.4) <= Z95 * s[2]
            cnt[3] += abs(θ[3] - 0.4) <= Z95 * s[3]
            rej += m.spillover_equality.p_value < 0.05
        end
        say(@sprintf("| %.1f | %s | %.3f | %.3f | %.3f | %.3f |", λe, vn, (cnt ./ R)..., rej / R))
    end
end
close(io)
