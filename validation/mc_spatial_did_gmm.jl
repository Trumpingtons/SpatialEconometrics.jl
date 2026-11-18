# Heteroskedasticity-robust spatial error: `errors = :spatial_gmm` in
# fit_spatial_did and `estimator = :gmm` in fit_spatial_error, against the
# maximum-likelihood fits they sit beside (AUDIT finding 35).
#
# A. Two-period DiD, 20 x 20 rook lattice, errors (I − 0.6W)⁻¹(s ∘ ε),
#    channels = :split at the null δ_T = δ_C = 0.8. Three patterns for the
#    error sd s: constant; doubled on the treated; 1 + 2·(exposure), so it
#    rises with the share of treated neighbours. Contiguous block and
#    scattered treated layouts.
# B. Chagas et al.'s staggered design (N = 200, kNN k = 10, T = 10, λ = 0.5,
#    treatment redrawn each period), b1 = b2 = 0.4: constant sd, and sd
#    doubled on one half of the map and again on treated unit-periods. The
#    panel GMM is this package's extension of Kelejian & Prucha's
#    cross-section theory, so this part is its evidence.
# C. Cross-section SEM in the spirit of Arraiz et al. (2010): 500 points,
#    a quarter of them packed into the north-east corner, W the
#    row-standardised binary distance band at 0.1, error sd proportional to each
#    unit's number of neighbours (scaled to mean 1), against constant sd.
#
# Run: julia --project=. validation/mc_spatial_did_gmm.jl
#      (MC_REPS=50 for a pilot)
# Output: validation/mc_spatial_did_gmm_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror

const R = parse(Int, get(ENV, "MC_REPS", "2000"))
const Z95 = 1.959963984540054
const OUT = joinpath(@__DIR__, "mc_spatial_did_gmm_output.txt")
io = open(OUT, "w")
say(s) = (println(s); println(io, s); flush(io))
t0 = time()
say("# Spatial-error GMM against ML — Monte Carlo ($R replications per cell)")
say("Julia $(VERSION). Coverage is of nominal 95% intervals; its Monte Carlo sd at R = $R is $(round(sqrt(0.95 * 0.05 / R), digits = 4)). `rej` is the rejection rate at 5% of a true null. `ML` is `errors = :spatial` / `estimator = :ml`, `GMM` is `errors = :spatial_gmm` / `estimator = :gmm`.")
say("")

# ── A ──────────────────────────────────────────────────────────────────
say("## A. Two-period DiD, 20 x 20 rook lattice, λ = 0.6, channels = :split, δ_T = δ_C = 0.8")
say("")
say("| layout | error sd | fit | rej | cov τ | cov spill T | cov spill C | cov total T | sd total | mean λ̂ | cov λ | failed |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
side = 20; N = side^2
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
W = make_rook_weights(side); Wt = row_standardize(W)
μ = collect(range(-1.0, 1.0, length = N))
layouts = [("block", [(7 <= coords[i, 1] <= 14 && 7 <= coords[i, 2] <= 14) ? 1.0 : 0.0 for i in 1:N]),
           ("scattered", (rr = MersenneTwister(7); [rand(rr) < 0.3 ? 1.0 : 0.0 for _ in 1:N]))]
A6 = lu(sparse(1.0I, N, N) - 0.6 * W)
for (lay, D) in layouts
    E = Wt * D
    ĒT = mean(E[D .== 1]); ĒC = mean(E[D .== 0])
    τ, γ, δ = 1.0, 0.5, 0.8
    truth = (τ, δ * ĒT, δ * ĒC, τ + δ * ĒT)
    for (sdlab, sdv) in (("constant", ones(N)), ("2 on treated", 1 .+ D), ("1 + 2·exposure", 1 .+ 2 .* E))
        res = Dict(e => (cov = zeros(Int, 4), rej = Ref(0), tot = Float64[], lam = Float64[],
                         covl = Ref(0), fail = Ref(0)) for e in (:spatial, :spatial_gmm))
        rng = MersenneTwister(20261118)
        for _ in 1:R
            u0 = A6 \ (sdv .* randn(rng, N)); u1 = A6 \ (sdv .* randn(rng, N))
            y = hcat(μ .+ u0, μ .+ γ .+ τ .* D .+ δ .* E .+ u1)
            for e in (:spatial, :spatial_gmm)
                r = res[e]
                m = try
                    fit_spatial_did(y, D, W; channels = :split, errors = e)
                catch err
                    err isa InterruptException && rethrow()
                    r.fail[] += 1; continue
                end
                est = (m.att_direct, m.att_spill_treated, m.att_spill_control, m.att_total_treated)
                se = (m.att_direct_se, m.att_spill_treated_se, m.att_spill_control_se, m.att_total_treated_se)
                for k in 1:4
                    r.cov[k] += abs(est[k] - truth[k]) <= Z95 * se[k]
                end
                r.rej[] += m.spillover_equality.p_value < 0.05
                push!(r.tot, m.att_total_treated); push!(r.lam, m.lambda)
                r.covl[] += abs(m.lambda - 0.6) <= Z95 * m.lambda_se
            end
        end
        for e in (:spatial, :spatial_gmm)
            r = res[e]; n = length(r.tot)
            say(@sprintf("| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.4f | %.4f | %.3f | %d |",
                         lay, sdlab, e === :spatial ? "ML" : "GMM", r.rej[] / n, (r.cov ./ n)...,
                         std(r.tot), mean(r.lam), r.covl[] / n, r.fail[]))
        end
    end
end
say("")

# ── B ──────────────────────────────────────────────────────────────────
say("## B. Chagas et al.'s staggered design, λ = 0.5, b1 = b2 = 0.4")
say("")
say("| error sd | fit | cov α | cov b1 | cov b2 | rej | sd b̂2 | mean λ̂ | cov λ | failed |")
say("|---|---|---|---|---|---|---|---|---|---|")
NB, TB = 200, 10
rngc = MersenneTwister(11); cB = randn(rngc, NB, 2)
WB = knn_weights(cB, 10); WBt = row_standardize(WB)
AB = lu(sparse(1.0I, NB, NB) - 0.5 * WB)
half = 1 .+ (cB[:, 1] .> 0)
for (sdlab, het) in (("constant", false), ("2 on east half, x2 on treated", true))
    res = Dict(e => (cnt = zeros(Int, 3), rej = Ref(0), b2 = Float64[], lam = Float64[],
                     covl = Ref(0), fail = Ref(0)) for e in (:spatial, :spatial_gmm))
    rng = MersenneTwister(20261120)
    for _ in 1:R
        μB = randn(rng, NB); λB = randn(rng, TB)
        Dp = Float64.(rand(rng, NB, TB) .< 0.5); x = randn(rng, NB, TB)
        X = zeros(NB, 2, TB); y = zeros(NB, TB)
        for t in 1:TB
            Et = WBt * Dp[:, t]; wx = WBt * x[:, t]
            X[:, 1, t] = x[:, t]; X[:, 2, t] = wx
            sd = het ? half .* (1 .+ Dp[:, t]) : ones(NB)
            y[:, t] = μB .+ λB[t] .+ Dp[:, t] .+ 0.4 .* Et .+ x[:, t] .+ wx .+ (AB \ (sd .* randn(rng, NB)))
        end
        for e in (:spatial, :spatial_gmm)
            r = res[e]
            m = try
                fit_spatial_did(y, Dp, WB; X = X, channels = :split, errors = e)
            catch err
                err isa InterruptException && rethrow()
                r.fail[] += 1; continue
            end
            θ = coef(m); s = stderror(m)
            r.cnt[1] += abs(θ[1] - 1.0) <= Z95 * s[1]
            r.cnt[2] += abs(θ[2] - 0.4) <= Z95 * s[2]
            r.cnt[3] += abs(θ[3] - 0.4) <= Z95 * s[3]
            r.rej[] += m.spillover_equality.p_value < 0.05
            push!(r.b2, θ[3]); push!(r.lam, m.lambda)
            r.covl[] += abs(m.lambda - 0.5) <= Z95 * m.lambda_se
        end
    end
    for e in (:spatial, :spatial_gmm)
        r = res[e]; n = length(r.b2)
        say(@sprintf("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.4f | %.4f | %.3f | %d |",
                     sdlab, e === :spatial ? "ML" : "GMM", (r.cnt ./ n)..., r.rej[] / n,
                     std(r.b2), mean(r.lam), r.covl[] / n, r.fail[]))
    end
end
say("")

# ── C ──────────────────────────────────────────────────────────────────
say("## C. Cross-section SEM, distance band on an irregular map, β = (1, 1, −1)")
say("")
rngp = MersenneTwister(5)
nC = 500
pts = vcat(0.5 .+ 0.5 .* rand(rngp, nC ÷ 4, 2), rand(rngp, nC - nC ÷ 4, 2))
Wb = distance_weights(pts, 0.1; decay = :power, alpha = 0.0, row_standardize = false)
nnb = vec(sum(Wb .!= 0, dims = 2))
keep = nnb .> 0
pts = pts[keep, :]; Wb = distance_weights(pts, 0.1; decay = :power, alpha = 0.0, row_standardize = false); nC = size(pts, 1)
nnb = vec(sum(Wb .!= 0, dims = 2)); WC = row_standardize(Wb)
XC = randn(MersenneTwister(6), nC, 2)
say(@sprintf("%d points (isolated ones dropped); neighbours per unit: min %d, median %d, max %d.", nC, minimum(nnb), Int(median(nnb)), maximum(nnb)))
say("")
say("| λ | error sd | fit | mean λ̂ | rmse λ̂ | cov λ | cov β1 | cov β2 | failed |")
say("|---|---|---|---|---|---|---|---|---|")
for λ0 in (-0.5, 0.0, 0.5, 0.8), (sdlab, sdv) in (("constant", ones(nC)), ("∝ neighbours", nnb ./ mean(nnb)))
    AC = lu(sparse(1.0I, nC, nC) - λ0 * WC)
    res = Dict(e => (lam = Float64[], covl = Ref(0), cb = zeros(Int, 2), fail = Ref(0)) for e in (:ml, :gmm))
    rng = MersenneTwister(20261119)
    for _ in 1:R
        y = 1.0 .+ XC * [1.0, -1.0] .+ AC \ (sdv .* randn(rng, nC))
        for e in (:ml, :gmm)
            r = res[e]
            m = try
                fit_spatial_error(WC, XC, y; estimator = e)
            catch err
                err isa InterruptException && rethrow()
                r.fail[] += 1; continue
            end
            b = coef(m); s = stderror(m)
            push!(r.lam, m.lambda)
            r.covl[] += abs(m.lambda - λ0) <= Z95 * m.lambda_se
            r.cb[1] += abs(b[2] - 1.0) <= Z95 * s[2]
            r.cb[2] += abs(b[3] + 1.0) <= Z95 * s[3]
        end
    end
    for e in (:ml, :gmm)
        r = res[e]; n = length(r.lam)
        say(@sprintf("| %.1f | %s | %s | %.4f | %.4f | %.3f | %.3f | %.3f | %d |", λ0, sdlab,
                     e === :ml ? "ML" : "GMM", mean(r.lam), sqrt(mean((r.lam .- λ0) .^ 2)),
                     r.covl[] / n, (r.cb ./ n)..., r.fail[]))
    end
end
say("")
say(@sprintf("Run time %.0f s.", time() - t0))
close(io)
