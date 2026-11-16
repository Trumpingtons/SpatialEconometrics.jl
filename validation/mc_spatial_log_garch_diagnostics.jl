# ---------------------------------------------------------------------------
# Diagnostics for one cell of mc_spatial_log_garch.jl, design A:
# n = 100 (rook 10 × 10), (λ, ρ) = (.05, .9), α = .5, δ = 1.
#
# There our RMSEs of λ̂ and ρ̂ are 35–47% above Sato & Matsuda's Table 1, while
# every other cell agrees. Three explanations are tested on the same samples:
#
#   start  Newton started at the true (λ, λ + ρ) instead of the grid's best
#          point: does a start at the truth find a different local maximum?
#   λ ≥ 0  the quasi-likelihood maximised over λ ≥ 0 only (the edge λ = 0 is
#          allowed): does a narrower search region cut the tail of λ̂?
#   z      the default fit, repeated over five further draws of the fixed z:
#          was the main run's z draw unusual?
#
# Run:  julia --project=. validation/mc_spatial_log_garch_diagnostics.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
const G = SpatialEconometrics.SpatialGARCH

const L = 10; const n = L^2
const λ0, ρ0, α0, δ0 = 0.05, 0.9, 0.5, 1.0
const a0 = λ0 + ρ0
const W = make_rook_weights(L)
const Ws = SparseMatrixCSC{Float64, Int}(W)
const ev = ComplexF64.(eigvals(Matrix(W)))
const Fa = lu(sparse(I - a0 * W)); const Rλ = sparse(I - λ0 * W)

draw_eps(rng, law) =
    law === :normal  ? randn(rng, n) :
    law === :chi3    ? sum(randn(rng, n) .^ 2 for _ in 1:3) ./ sqrt(15) :
    exp.(randn(rng, n) .- 1)

simulate(rng, z, law) = sqrt.(exp.(Fa \ (α0 .+ δ0 .* z .+ Rλ * log.(draw_eps(rng, law) .^ 2))))

# The concentrated log-likelihood on the edge λ = 0, maximised over a:
# a grid, then golden-section search around its best point.
function edge_max(d)
    f(a) = G._concentrated(d, 0.0, a)[1]
    grid = range(-0.985, 0.985, length = 198)
    k = argmax(f.(grid))
    lo = grid[max(k - 1, 1)]; hi = grid[min(k + 1, end)]
    φ = (sqrt(5) - 1) / 2
    for _ in 1:60
        c = hi - φ * (hi - lo); e = lo + φ * (hi - lo)
        f(c) > f(e) ? (hi = e) : (lo = c)
    end
    a = (lo + hi) / 2
    return a, f(a)
end

summ(v, t) = (mean(v) - t, sqrt(mean((v .- t) .^ 2)))

io = open(joinpath(@__DIR__, "mc_spatial_log_garch_diagnostics_output.txt"), "w")
out(s) = (println(s); println(io, s); flush(io))
out("Diagnostics: design A, n = 100, (λ, ρ) = (.05, .9); bias / RMSE")
out("S&M Table 1, normal: λ −.027 / .139, ρ −.011 / .108;  chi3: λ −.023 / .141, ρ −.017 / .115;")
out("                     lognorm: λ −.029 / .132, ρ −.006 / .108")
t0 = time()

# ── start and λ ≥ 0, on the main run's z ──────────────────────────────────
z = randn(MersenneTwister(1000 + L), n)          # the main run's z
X = hcat(ones(n), z)
out(@sprintf("main run's z: sd %.2f", std(z)))
for law in (:normal, :chi3, :lognorm)
    rng = MersenneTwister(5000 + length(String(law)))
    R = 1000
    λd = Float64[]; ρd = Float64[]; λs = Float64[]; ρs = Float64[]
    λc = Float64[]; ρc = Float64[]
    nfail_d = 0; nfail_s = 0; ndiff = 0; nlower = 0; nedge = 0
    for r in 1:R
        y = simulate(rng, z, law)
        Y = log.(y .^ 2); d = G._GData(Y, Ws * Y, X, Ws, ev, n, 2)
        # default fit
        md = try fit_spatial_log_garch(y, W; Z = z, eigenvalues = ev) catch e
            e isa ErrorException || rethrow(); nothing end
        md === nothing && (nfail_d += 1; continue)
        push!(λd, md.lambda); push!(ρd, md.rho)
        # Newton from the truth
        rs = try G._newton(d, λ0, a0, -0.99, 0.99; tol = 1e-8, maxiter = 100) catch e
            e isa ErrorException || rethrow(); nothing end
        if rs === nothing
            nfail_s += 1
        else
            push!(λs, rs[1]); push!(ρs, rs[2] - rs[1])
            if abs(rs[1] - md.lambda) > 1e-4
                ndiff += 1
                rs[3] < md.loglik - 1e-8 && (nlower += 1)
            end
        end
        # λ ≥ 0: the interior root if it has λ ≥ 0 and beats the edge,
        # otherwise the best point on the edge λ = 0
        ae, le = edge_max(d)
        if md.lambda >= 0 && md.loglik >= le
            push!(λc, md.lambda); push!(ρc, md.rho)
        else
            push!(λc, 0.0); push!(ρc, ae); nedge += 1
        end
    end
    out(@sprintf("\n%-7s R=%d  [%.0fs]", law, R, time() - t0))
    for (lab, lv, rv) in (("default (grid start)", λd, ρd), ("Newton from truth", λs, ρs),
                          ("λ ≥ 0", λc, ρc))
        bl, rl = summ(lv, λ0); br, rr = summ(rv, ρ0)
        out(@sprintf("   %-22s fits %4d   λ %+.3f / %.3f   ρ %+.3f / %.3f", lab, length(lv), bl, rl, br, rr))
    end
    out(@sprintf("   default failures %d; from-truth failures %d; from truth reached a different root in %d fits, %d of them with a lower likelihood",
                 nfail_d, nfail_s, ndiff, nlower))
    out(@sprintf("   λ ≥ 0: %d of %d fits on the edge λ = 0 (default λ̂ < 0 in %d)", nedge, length(λc), count(<(0), λd)))
end

# ── z: five further draws, default fit, normal ε ─────────────────────────
out("\nz draws (normal ε, default fit, R = 1000 each)")
for s in 1:5
    zz = randn(MersenneTwister(7000 + s), n)
    rng = MersenneTwister(8000 + s)
    λv = Float64[]; ρv = Float64[]; nf = 0
    for r in 1:1000
        y = simulate(rng, zz, :normal)
        m = try fit_spatial_log_garch(y, W; Z = zz, eigenvalues = ev) catch e
            e isa ErrorException || rethrow(); nothing end
        m === nothing && (nf += 1; continue)
        push!(λv, m.lambda); push!(ρv, m.rho)
    end
    bl, rl = summ(λv, λ0); br, rr = summ(ρv, ρ0)
    out(@sprintf("   z draw %d: sd(z) %.2f  fits %4d  λ %+.3f / %.3f   ρ %+.3f / %.3f  [%.0fs]",
                 s, std(zz), length(λv), bl, rl, br, rr, time() - t0))
end
out(@sprintf("\ntotal %.0f s", time() - t0))
close(io)
