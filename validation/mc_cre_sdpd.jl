# Monte Carlo for fit_cre_dynamic_spatial_panel (Li & Yang 2021, CRE-SDPD):
# bias of the CQMLE and the M-estimator, and the coverage of the sandwich
# standard errors from the corrected plug-in Var(S*). Recorded in docs/AUDIT.md.
#
# Design: Li & Yang (2021) Table 1, DGP1, T = 3, m = 10.
#   W1 = W3: queen contiguity on an r×c grid, row-standardised;
#   W2: group interaction, k = round(√n) groups of sizes U(.5n̄, 1.5n̄)
#       adjusted to sum to n (their footnote 8, α = 0.5);
#   X_t: footnote 7, X_t = μ_x + g t 1_n + ζ_t, (1 − θ1 L)ζ_t = ϵ_t + θ2 ϵ_{t−1},
#        ϵ_t ~ N(0, σ1²), μ_x = e + mean_{t=−m..T} ϵ_t, e ~ N(0, σ2²),
#        (g, θ1, θ2, σ1, σ2) = (.01, .5, .5, 2, 1);
#   Z ~ Bernoulli(.5); μ = X̄π + ε, X̄ the mean over 0..T;
#   β = γ = π = σv² = φ = 1, ρ = .3, λ1 = λ2 = λ3 = .2;
#   α_t ~ N(1, 1); y_{−m} = 0;
#   v: normal, normal mixture (10% N(0,4), 90% N(0,1)) or χ²(3), standardised.
# ε uses the same family as v here (the paper's text is garbled at that
# point), which also exercises the ε moments μ3ε, μ4ε of the variance.
#
# The CQMLE columns average the replications where its optimiser converged
# (it can stop at φ = 0 or fail); the count is in each cell's last row.
#
# Each cell reseeds its own generator, so any cell can be rerun alone and
# prints the same row.
#
# Run: julia --project=. validation/mc_cre_sdpd.jl
# Output: validation/mc_cre_sdpd_output.txt (markdown rows)
using SpatialEconometrics, LinearAlgebra, Random, Statistics, Printf
import StatsAPI: coef, coefnames, stderror

function queen(r, c)
    n = r * c
    W = zeros(n, n)
    id(i, j) = (j - 1) * r + i
    for i in 1:r, j in 1:c, di in -1:1, dj in -1:1
        (di == 0 && dj == 0) && continue
        ii, jj = i + di, j + dj
        (1 <= ii <= r && 1 <= jj <= c) && (W[id(i, j), id(ii, jj)] = 1)
    end
    return W ./ sum(W; dims = 2)
end

function group(rng, n)
    k = round(Int, sqrt(n)); n̄ = n / k
    sz = round.(Int, n̄ .* (0.5 .+ rand(rng, k)))
    while sum(sz) != n                        # adjust so the sizes sum to n
        i = rand(rng, 1:k)
        sum(sz) > n ? (sz[i] > 2 && (sz[i] -= 1)) : (sz[i] += 1)
    end
    W = zeros(n, n); s = 0
    for g in sz
        r = s+1:s+g
        W[r, r] .= 1 / (g - 1)
        for i in r; W[i, i] = 0; end
        s += g
    end
    return W
end

function draws(rng, errs, k)
    if errs === :normal
        return randn(rng, k)
    elseif errs === :mixture                   # var .1·4 + .9·1 = 1.3
        return [(rand(rng) < 0.1 ? 2.0 : 1.0) * randn(rng) for _ in 1:k] ./ sqrt(1.3)
    else                                       # χ²(3), mean 3, var 6
        return (sum(randn(rng, k, 3) .^ 2; dims = 2)[:, 1] .- 3) ./ sqrt(6)
    end
end

function sim(rng, n, T, m, W1, W2, W3, errs; ρ = 0.3, λ1 = 0.2, λ2 = 0.2, λ3 = 0.2)
    g, θ1, θ2, σ1, σ2 = 0.01, 0.5, 0.5, 2.0, 1.0
    burn = 50
    ϵ = [σ1 .* randn(rng, n) for _ in 1:(burn + m + T + 1)]
    ζ = zeros(n); Xs = Vector{Vector{Float64}}()
    for (j, t) in enumerate(-(m + burn) + 1:T)
        ζ = θ1 .* ζ .+ ϵ[j + 1] .+ θ2 .* ϵ[j]
        t >= -m && push!(Xs, ζ)
    end
    μx = σ2 .* randn(rng, n) .+ sum(ϵ[burn+1:end]) ./ (T + m + 1)
    Xs = [μx .+ g * t .+ Xs[i] for (i, t) in enumerate(-m:T)]      # X_{−m..T}
    Xbar = sum(Xs[m+1:end]) ./ (T + 1)
    z = Float64.(rand(rng, n) .< 0.5)
    ε = draws(rng, errs, n)
    μ = Xbar .+ ε
    α = vcat(ones(m), randn(rng, T) .+ 1)          # α_{−m+1..0} = 1, α_{1..T} ~ N(1,1)
    B1 = I - λ1 * W1; B3 = I - λ3 * W3
    y = zeros(n); ys = Vector{Vector{Float64}}()
    for (j, t) in enumerate(-m+1:T)
        u = B3 \ draws(rng, errs, n)
        y = B1 \ (ρ .* y .+ λ2 .* (W2 * y) .+ Xs[j+1] .+ z .+ μ .+ α[j] .+ u)
        t >= 0 && push!(ys, y)
    end
    return reduce(vcat, ys), reshape(reduce(vcat, Xs[m+1:end]), :, 1), reshape(z, :, 1)
end

const TRUTH = Dict("x" => 1.0, "z" => 1.0, "mean_x" => 1.0, "sigma2_v" => 1.0, "phi" => 1.0,
                   "rho" => 0.3, "lambda1" => 0.2, "lambda2" => 0.2, "lambda3" => 0.2)
const SHOW = ["x", "z", "mean_x", "sigma2_v", "phi", "rho", "lambda1", "lambda2", "lambda3"]

function cell(r, c, errs; T = 3, m = 10, reps = 500, seed = 20261113)
    n = r * c
    rng = MersenneTwister(seed)
    W1 = queen(r, c); W3 = W1; W2 = group(rng, n)
    q = 1.959963984540054
    est = Dict(nm => Float64[] for nm in SHOW); cq = Dict(nm => Float64[] for nm in SHOW)
    se = Dict(nm => Float64[] for nm in SHOW); seh = Dict(nm => Float64[] for nm in SHOW)
    cov = Dict(nm => 0 for nm in SHOW); covh = Dict(nm => 0 for nm in SHOW)
    fails = 0; secs = 0.0
    for _ in 1:reps
        y, X, Z = sim(rng, n, T, m, W1, W2, W3, errs)
        t0 = time()
        f = try
            fit_cre_dynamic_spatial_panel(y, X, Z, W1, n, T + 1; m = m, W2 = W2, W3 = W3,
                                          varnames = ["x"], zvarnames = ["z"], compute_cqmle = true)
        catch err
            fails == 0 && @warn "first failed fit" exception = err
            nothing
        end
        secs += time() - t0
        isnothing(f) && (fails += 1; continue)
        s = stderror(f); sh = sqrt.(max.(diag(f.vcov_hessian), 0))
        for nm in SHOW
            i = findfirst(==(nm), coefnames(f))
            b = coef(f)[i]
            push!(est[nm], b)
            isnan(f.cqmle[i]) || push!(cq[nm], f.cqmle[i])
            push!(se[nm], s[i]); push!(seh[nm], sh[i])
            cov[nm] += abs(b - TRUTH[nm]) <= q * s[i]
            covh[nm] += abs(b - TRUTH[nm]) <= q * sh[i]
        end
    end
    ok = reps - fails
    rows = String[]
    for nm in SHOW
        push!(rows, @sprintf("| %d | %s | %s | %.2f | %.3f (%.3f) | %.3f (%.3f) | %.3f | %.3f | %.3f | %.3f |",
            n, errs, nm, TRUTH[nm], mean(cq[nm]), std(cq[nm]), mean(est[nm]), std(est[nm]),
            mean(se[nm]), mean(seh[nm]), cov[nm] / ok, covh[nm] / ok))
    end
    push!(rows, @sprintf("|  | %s | (%d of %d fits; CQMLE converged in %d of them; %.1f s per fit) | | | | | | | |",
                         errs, ok, reps, length(cq["rho"]), secs / reps))
    return rows
end

out = joinpath(@__DIR__, "mc_cre_sdpd_output.txt")
open(out, "w") do io
    hdr = "| n | errors | param | true | CQMLE mean (sd) | M-est mean (sd) | mean rse | mean Hessian se | cover rse | cover Hessian |\n|---|---|---|---|---|---|---|---|---|---|"
    println(io, hdr); println(hdr)
    for (r, c, reps) in ((10, 10, 1000), (10, 20, 500)), errs in (:normal, :mixture, :chi2)
        for row in cell(r, c, errs; reps = reps)
            println(io, row); println(row)
        end
        flush(io); flush(stdout)
    end
end
