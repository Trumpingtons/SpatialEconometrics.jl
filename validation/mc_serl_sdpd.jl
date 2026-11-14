# Monte Carlo for fit_common_shock_dynamic_panel (Bai & Chen 2026, SERL-QML):
# does coverage at SERL order L = 1 hold as N grows, when y0 comes from a
# long-running spatial process? Recorded in docs/AUDIT.md.
#
# Design: Bai & Chen (2026) §6.2.1, the long-running initial condition.
#   W: row-normalised 8-nearest-neighbour graph on uniform random locations,
#      drawn once per N and held fixed across replications;
#   λ_i ~ N(0, I₂); h_is = 0.8 h_i,s−1 + √(1 − 0.8²) v_is, v_is ~ N(0, I₂),
#      started from its stationary distribution;
#   x_is = 0.5 λ_i + √(1 − 0.5²) h_is (K = 2, elementwise with r = 2);
#   y starts at 0 and runs 100 presample periods; y0 is the last of them;
#   f_s ~ N(0, I₂), drawn once per T over presample and sample and held fixed;
#   ε ~ N(0, 1), time effects zero, φ = 0.3, β = (0.8, −0.3), ρ ∈ {0.2, 0.5}.
#   Q_X: the unit means of x over the sample periods; z_i(L) stacks
#   Q_X, WQ_X, …, W^L Q_X, y0, Wy0, …, W^L y0, each centred and scaled.
# Each panel is fitted with L = 0 and L = 1 (paired). Intervals are 95%,
# from the sandwich with the pair component scaled by N/(N − p) (eq. 23).
#
# The paper reports N ∈ {500, 1000}; the cells beyond are this package's
# question: at L = 0, coverage of ρ falls with N (their Table 4), and at
# L = 1 they measure only two N.
#
# Each cell reseeds its own generator, so any cell can be rerun alone.
#
# Run: julia --project=. validation/mc_serl_sdpd.jl [R] [N,T,ρ ...]
#      e.g. julia --project=. validation/mc_serl_sdpd.jl 500 500,5,0.5 1000,5,0.5
# Output: one markdown row per (cell, L) on stdout.
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror

const φ0 = 0.3
const β0 = [0.8, -0.3]
const PRESAMPLE = 100

network(N) = knn_weights(rand(MersenneTwister(1000 + N), N, 2), 8)
factors(T) = randn(MersenneTwister(2000 + T), PRESAMPLE + T, 2)

function simulate(rng, W, N, T, ρ, f)
    B = lu(sparse(I - ρ * W))
    λ = randn(rng, N, 2)
    h = randn(rng, N, 2)                        # stationary start, variance 1
    y = zeros(N)
    ys = Vector{Float64}[]; xs = Matrix{Float64}[]
    for s in 1:(PRESAMPLE + T)
        h = 0.8 .* h .+ sqrt(1 - 0.64) .* randn(rng, N, 2)
        x = 0.5 .* λ .+ sqrt(0.75) .* h
        y = B \ (φ0 .* y .+ x * β0 .+ λ * f[s, :] .+ randn(rng, N))
        if s >= PRESAMPLE
            push!(ys, y); push!(xs, s == PRESAMPLE ? zeros(N, 2) : x)
        end
    end
    return reduce(vcat, ys), reduce(vcat, xs)
end

function cell(R, N, T, ρ; Ls = (0, 1))
    W = network(N)
    ev = eigvals(Matrix(W))
    f = factors(T)
    rng = MersenneTwister(hash((N, T, ρ)))
    θ0 = vcat(ρ, φ0, β0)
    est = Dict(L => fill(NaN, R, 4) for L in Ls)
    cov = Dict(L => fill(false, R, 4) for L in Ls)
    fails = Dict(L => 0 for L in Ls)
    t0 = time()
    for rep in 1:R
        y, X = simulate(rng, W, N, T, ρ, f)
        for L in Ls
            fit = try
                fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = L, eigenvalues = ev)
            catch err
                err isa ErrorException || rethrow()
                fails[L] += 1
                continue
            end
            b = coef(fit); se = stderror(fit)
            est[L][rep, :] = b
            cov[L][rep, :] = abs.(b .- θ0) .<= 1.959964 .* se
        end
    end
    for L in Ls
        ok = .!isnan.(est[L][:, 1])
        e = est[L][ok, :]; c = cov[L][ok, :]
        bias = vec(mean(e; dims = 1)) .- θ0
        sd = vec(std(e; dims = 1))
        cv = vec(mean(c; dims = 1))
        @printf("| %d | %d | %.1f | %d | %+.4f | %.4f | %+.4f | %.4f | %.3f | %.3f | %.3f | %.3f | %d/%d |\n",
                N, T, ρ, L, bias[1], sd[1], bias[2], sd[2], cv[1], cv[2], cv[3], cv[4],
                count(ok), R)
    end
    @printf("<!-- cell N=%d T=%d ρ=%.1f: %.0f s -->\n", N, T, ρ, time() - t0)
    flush(stdout)
end

R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
cells = length(ARGS) >= 2 ? [Tuple(parse.(Float64, split(a, ","))) for a in ARGS[2:end]] :
        [(500, 5, 0.5), (1000, 5, 0.5), (2000, 5, 0.5), (4000, 5, 0.5)]
println("| N | T | ρ | L | bias ρ | sd ρ | bias φ | sd φ | cov ρ | cov φ | cov β₁ | cov β₂ | fits |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (N, T, ρ) in cells
    cell(R, Int(N), Int(T), ρ)
end
