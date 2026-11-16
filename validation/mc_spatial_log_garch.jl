# ---------------------------------------------------------------------------
# Monte Carlo for fit_spatial_log_garch (Sato & Matsuda 2021), roadmap #16.
#
#   A. Sato & Matsuda's Table 1 design: rook W on an L×L grid (n = 100, 400),
#      z ~ N(0,1) fixed, α = 0.5, δ = 1, (λ, ρ) = (.9, .05), (.45, .45),
#      (.05, .9), three laws for ε, 1000 replications. Bias and RMSE are
#      compared with their table; coverage, which they do not report, is new.
#   B. Intercept only (no Z), Fülle & Otto's case, (λ, ρ) = (.45, .45),
#      normal ε, n = 100, 400, 1600, 500 replications: does the error's
#      moving-average part identify λ in practice?
#   C. A proxy W: the data come from 10-nearest-neighbour W on random
#      locations S; the fit uses covariate_weights on (1 − ω)S + ωE, E ~ N(0, I),
#      ω = 0, .01, .03, .1, n = 400, 1600, 200 replications. ω = 0 rebuilds the
#      true W. The share of true neighbours the proxy keeps is printed; at a
#      fixed ω it falls as n grows, because the points get closer together
#      while the noise does not shrink. This measures the bias a proxy W
#      leaves in ρ̂ and λ̂.
#
# A fit throws when the search finds no interior maximum; the summaries are
# over the fits that return. With row-standardised W and an intercept the
# quasi-likelihood rises without bound as λ → 1, so a sample with no interior
# maximum sends the search to the edge of `bounds`; those are counted apart.
#
# ε is scaled so that E ε² = 1: normal; χ²₃/√15; log-normal exp(Z − 1).
# Under the log-normal law log ε² is itself normal, so there (only) the
# Gaussian covariance −Ĥ⁻¹ is also valid; coverage under both is reported.
#
# Run:  julia --project=. validation/mc_spatial_log_garch.jl
#       (R_SCALE=0.05 for a pilot at 5% of the replications)
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: stderror, coef

const SCALE = parse(Float64, get(ENV, "R_SCALE", "1"))
reps(R) = max(10, round(Int, SCALE * R))
const ZQ = 1.959963984540054

draw_eps(rng, law, n) =
    law === :normal  ? randn(rng, n) :
    law === :chi3    ? sum(randn(rng, n) .^ 2 for _ in 1:3) ./ sqrt(15) :
    law === :lognorm ? exp.(randn(rng, n) .- 1) : error("law")

function simulate_y(rng, Fa, Rλ, α, Zδ, law, n)
    U = log.(draw_eps(rng, law, n) .^ 2)
    Y = Fa \ (α .+ Zδ .+ Rλ * U)
    return sqrt.(exp.(Y))
end

struct Acc
    est::Vector{Vector{Float64}}
    se::Vector{Vector{Float64}}
    seg::Vector{Vector{Float64}}
    α::Vector{Float64}
end
Acc() = Acc(Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Float64[])

function run_cell(rng, W, ev, Wfit, evfit, Z, λ, ρ, α, δ, law, R)
    n = size(W, 1)
    Fa = lu(sparse(I - (λ + ρ) * W)); Rλ = sparse(I - λ * W)
    Zδ = Z === nothing ? zeros(n) : Z * δ
    acc = Acc(); fails = 0; edge = 0
    for r in 1:R
        y = simulate_y(rng, Fa, Rλ, α, Zδ, law, n)
        m = try
            fit_spatial_log_garch(y, Wfit; Z = Z, eigenvalues = evfit)
        catch e
            e isa ErrorException || rethrow()
            fails += 1; occursin("edge of `bounds`", e.msg) && (edge += 1); continue
        end
        push!(acc.est, coef(m)); push!(acc.se, stderror(m))
        push!(acc.seg, sqrt.(max.(0, diag(m.vcov_gaussian)))); push!(acc.α, m.alpha)
    end
    return acc, (fails, edge)
end

function summarise(acc, truth)
    E = reduce(hcat, acc.est)'; S = reduce(hcat, acc.se)'; Sg = reduce(hcat, acc.seg)'
    bias = vec(mean(E, dims = 1)) .- truth
    sd = vec(std(E, dims = 1))
    rmse = sqrt.(vec(mean((E .- truth') .^ 2, dims = 1)))
    ratio = vec(mean(S, dims = 1)) ./ sd
    cov = vec(mean(abs.(E .- truth') .<= ZQ .* S, dims = 1))
    covg = vec(mean(abs.(E .- truth') .<= ZQ .* Sg, dims = 1))
    return bias, sd, rmse, ratio, cov, covg
end

io = open(joinpath(@__DIR__, "mc_spatial_log_garch_output.txt"), "w")
out(s) = (println(s); println(io, s); flush(io); flush(stdout))
out("Monte Carlo: fit_spatial_log_garch   (R_SCALE = $SCALE, $(Threads.nthreads()) thread)")
out("coverage of nominal 95% intervals; cov = sandwich, covG = Gaussian −Ĥ⁻¹; SE/sd = mean SE / sd")
out("fail = fits that threw (summaries are over the rest); at a bound = the search ran to the edge of `bounds`")
t0 = time()

# ── A ──────────────────────────────────────────────────────────────────────
out("\n== A. Sato & Matsuda Table 1 design (α = .5, δ = 1), fixed z ==")
for L in (10, 20)
    W = make_rook_weights(L); n = L^2; ev = eigvals(Matrix(W))
    rng = MersenneTwister(1000 + L)
    z = randn(rng, n)
    for (λ, ρ) in ((0.9, 0.05), (0.45, 0.45), (0.05, 0.9)), law in (:normal, :chi3, :lognorm)
        R = reps(1000)
        acc, fails = run_cell(rng, W, ev, W, ev, reshape(z, :, 1), λ, ρ, 0.5, [1.0], law, R)
        b, sd, rm, rt, cv, cg = summarise(acc, [1.0, λ, ρ])
        αb = mean(acc.α) - 0.5; αr = sqrt(mean((acc.α .- 0.5) .^ 2))
        out(@sprintf("n=%4d (λ,ρ)=(%.2f,%.2f) %-7s R=%d fail=%d (at a bound %d)  [%.0fs]",
                     n, λ, ρ, law, R, fails..., time() - t0))
        for (k, nm) in enumerate(("δ", "λ", "ρ"))
            out(@sprintf("   %-2s bias %+.3f  rmse %.3f  sd %.3f  SE/sd %.2f  cov %.3f  covG %.3f",
                         nm, b[k], rm[k], sd[k], rt[k], cv[k], cg[k]))
        end
        out(@sprintf("   α  bias %+.3f  rmse %.3f   (no SE)", αb, αr))
    end
end

# ── B ──────────────────────────────────────────────────────────────────────
out("\n== B. Intercept only, (λ, ρ) = (.45, .45), normal ε ==")
for L in (10, 20, 40)
    W = make_rook_weights(L); n = L^2; ev = eigvals(Matrix(W))
    rng = MersenneTwister(2000 + L); R = reps(500)
    acc, fails = run_cell(rng, W, ev, W, ev, nothing, 0.45, 0.45, 0.5, Float64[], :normal, R)
    b, sd, rm, rt, cv, cg = summarise(acc, [0.45, 0.45])
    out(@sprintf("n=%4d R=%d fail=%d (at a bound %d)  [%.0fs]", n, R, fails..., time() - t0))
    for (k, nm) in enumerate(("λ", "ρ"))
        out(@sprintf("   %-2s bias %+.3f  rmse %.3f  sd %.3f  SE/sd %.2f  cov %.3f  covG %.3f",
                     nm, b[k], rm[k], sd[k], rt[k], cv[k], cg[k]))
    end
    E = reduce(hcat, acc.est)'
    out(@sprintf("   λ+ρ bias %+.3f  sd %.3f;  corr(λ̂, ρ̂) %.3f", mean(E[:, 1] .+ E[:, 2]) - 0.9,
                 std(E[:, 1] .+ E[:, 2]), cor(E[:, 1], E[:, 2])))
end

# ── C ──────────────────────────────────────────────────────────────────────
out("\n== C. Proxy W: true W = 10-NN on locations S; fit W = covariate_weights((1−ω)S + ωE, 10) ==")
out("   (λ, ρ) = (.45, .45), α = .5, δ = 1, normal ε, fixed z, S and E")
for n in (400, 1600)
    rng = MersenneTwister(3000 + n)
    S = rand(rng, n, 2); E = randn(rng, n, 2); z = randn(rng, n)
    W = knn_weights(S, 10; binary = true); ev = eigvals(Matrix(W))
    for ω in (0.0, 0.01, 0.03, 0.1)
        Wf = covariate_weights((1 - ω) .* S .+ ω .* E, 10)
        evf = eigvals(Matrix(Wf))
        shared = sum(Wf .> 0 .&& W .> 0) / sum(W .> 0)
        R = reps(200)
        acc, fails = run_cell(rng, W, ev, Wf, evf, reshape(z, :, 1), 0.45, 0.45, 0.5, [1.0], :normal, R)
        b, sd, rm, rt, cv, cg = summarise(acc, [1.0, 0.45, 0.45])
        out(@sprintf("n=%4d ω=%.2f  neighbours shared with true W %.2f  R=%d fail=%d (at a bound %d)  [%.0fs]",
                     n, ω, shared, R, fails..., time() - t0))
        for (k, nm) in enumerate(("δ", "λ", "ρ"))
            out(@sprintf("   %-2s bias %+.3f  rmse %.3f  sd %.3f  SE/sd %.2f  cov %.3f",
                         nm, b[k], rm[k], sd[k], rt[k], cv[k]))
        end
    end
end
out(@sprintf("\ntotal %.0f s", time() - t0))
close(io)
