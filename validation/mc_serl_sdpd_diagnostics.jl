# Diagnostics for validation/mc_serl_sdpd.jl: why coverage of β falls with N
# at SERL order L = 1 in the long-run design. Recorded in docs/AUDIT.md.
#
# Part 1 (R reps, N given; recorded at 4000): bias, sd and mean SE of every parameter
#   - in Bai & Chen's baseline design (§6.1.1), where x is independent of λ
#     and y0 = 1'λ + ν, so L = 0 is correctly specified: is the variance right?
#   - in the long-run design at L = 1 and L = 2: does a higher order remove
#     the bias? The long-run panels are the first R of the main MC's cell.
# Part 2 (R reps, N given): the long-run design at L = 1 with Q = the unit
#   means of x (the default) and with Q = every period of x.
#
# Run: julia --project=. validation/mc_serl_sdpd_diagnostics.jl R part N
#      part 1: julia --project=. validation/mc_serl_sdpd_diagnostics.jl 200 1 4000
#      part 2: julia --project=. validation/mc_serl_sdpd_diagnostics.jl 200 2 2000
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror
const φ0 = 0.3; const β0 = [0.8, -0.3]; const PRE = 100
network(N) = knn_weights(rand(MersenneTwister(1000 + N), N, 2), 8)
factors(T) = randn(MersenneTwister(2000 + T), PRE + T, 2)
function sim_long(rng, W, N, T, ρ, f)
    B = lu(sparse(I - ρ * W)); λ = randn(rng, N, 2); h = randn(rng, N, 2); y = zeros(N)
    ys = Vector{Float64}[]; xs = Matrix{Float64}[]
    for s in 1:(PRE + T)
        h = 0.8 .* h .+ sqrt(1 - 0.64) .* randn(rng, N, 2)
        x = 0.5 .* λ .+ sqrt(0.75) .* h
        y = B \ (φ0 .* y .+ x * β0 .+ λ * f[s, :] .+ randn(rng, N))
        s >= PRE && (push!(ys, y); push!(xs, s == PRE ? zeros(N, 2) : x))
    end
    reduce(vcat, ys), reduce(vcat, xs)
end
# Baseline (§6.1.1): x independent of λ, y0 = 1'λ + ν, so L = 0 is exact.
function sim_base(rng, W, N, T, ρ, f)
    B = lu(sparse(I - ρ * W)); λ = randn(rng, N, 2); y0 = λ * ones(2) .+ randn(rng, N)
    ys = [y0]; xs = [zeros(N, 2)]
    for t in 1:T
        x = randn(rng, N, 2)
        push!(ys, B \ (φ0 .* ys[end] .+ x * β0 .+ λ * f[PRE + t, :] .+ randn(rng, N))); push!(xs, x)
    end
    reduce(vcat, ys), reduce(vcat, xs)
end
function run(design, R, N, T, ρ, Ls)
    W = network(N); ev = eigvals(Matrix(W)); f = factors(T)
    rng = MersenneTwister(hash((N, T, ρ)))      # same seed as the MC: long-run reps match it
    θ0 = vcat(ρ, φ0, β0); E = Dict(L => fill(NaN, R, 4) for L in Ls); S = Dict(L => fill(NaN, R, 4) for L in Ls)
    t0 = time()
    for rep in 1:R
        y, X = (design == :long ? sim_long : sim_base)(rng, W, N, T, ρ, f)
        for L in Ls
            fit = try fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = L, eigenvalues = ev)
            catch err; err isa ErrorException || rethrow(); continue; end
            E[L][rep, :] = coef(fit); S[L][rep, :] = stderror(fit)
        end
    end
    for L in Ls, k in 1:4
        ok = .!isnan.(E[L][:, 1]); e = E[L][ok, k]; s = S[L][ok, k]
        b = mean(e) - θ0[k]; sd = std(e)
        @printf("| %s | %d | %d | %s | %+.4f | %.4f | %.4f | %.2f | %+.2f | %.3f | %d |\n", design, N, L,
                ("ρ", "φ", "β₁", "β₂")[k], b, sd, mean(s), mean(s) / sd, b / sd,
                mean(abs.(e .- θ0[k]) .<= 1.959964 .* s), count(ok))
    end
    @printf("<!-- %s N=%d: %.0f s -->\n", design, N, time() - t0); flush(stdout)
end
function run2(R, N, T, ρ)
    W = network(N); ev = eigvals(Matrix(W)); f = factors(T)
    rng = MersenneTwister(hash((N, T, ρ))); θ0 = vcat(ρ, φ0, β0)
    E = Dict(q => fill(NaN, R, 4) for q in (:mean, :full)); S = deepcopy(E); t0 = time()
    for rep in 1:R
        y, X = sim_long(rng, W, N, T, ρ, f)
        Qfull = reduce(hcat, [reshape(X[:, k], N, T + 1)[:, 2:end] for k in 1:2])
        for q in (:mean, :full)
            fit = try fit_common_shock_dynamic_panel(y, X, W, N, T + 1; r = 2, L = 1, eigenvalues = ev,
                                                     Q = q == :full ? Qfull : nothing)
            catch err; err isa ErrorException || rethrow(); continue; end
            E[q][rep, :] = coef(fit); S[q][rep, :] = stderror(fit)
        end
    end
    for q in (:mean, :full), k in 1:4
        ok = .!isnan.(E[q][:, 1]); e = E[q][ok, k]; s = S[q][ok, k]; b = mean(e) - θ0[k]; sd = std(e)
        @printf("| %s | %d | %s | %+.4f | %.4f | %.2f | %+.2f | %.3f | %d |\n", q, N, ("ρ", "φ", "β₁", "β₂")[k],
                b, sd, mean(s) / sd, b / sd, mean(abs.(e .- θ0[k]) .<= 1.959964 .* s), count(ok))
    end
    @printf("<!-- %.0f s -->\n", time() - t0)
end
R, part, N = parse.(Int, ARGS[1:3])
if part == 1
    println("| design | N | L | par | bias | sd | mean SE | SE/sd | bias/sd | cov | fits |")
    println("|---|---|---|---|---|---|---|---|---|---|---|")
    run(:base, R, N, 5, 0.5, (0,))
    run(:long, R, N, 5, 0.5, (1, 2))
else
    println("| Q | N | par | bias | sd | SE/sd | bias/sd | cov | fits |\n|---|---|---|---|---|---|---|---|---|")
    run2(R, N, 5, 0.5)
end
