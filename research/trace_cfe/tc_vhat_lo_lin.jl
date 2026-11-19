# Theory step 2: the leave-one-out V̂ when the linear part dominates V.
#
# tc_vhat_lo.jl found the linear part of the leave-one-out V̂ biased down by up to 16% at K/n = 0.39, but
# the total within 3%, because the quadratic part made up most of V. Here f is scaled so that the linear
# part is 90% of V (Vl/V = 0.9), which exposes that bias undiluted. Otherwise the design and seeds are
# tc_vhat_lo.jl's (n = 900, kNN W, Chebyshev surface basis plus one covariate, A = MW + MDM, R = 1000).
#
# Expected: with sᵢ = y₀ᵢûᵢ/Mᵢᵢ and â = b + GQ′ε, E[âᵢ²sᵢ] − E[âᵢ²]σᵢ² contains 2bᵢfᵢ(GQ′ΣM)ᵢᵢ/Mᵢᵢ, which
# scales with f² like Vl itself, so the linear part's relative bias should not shrink as f grows. It is
# zero with constant variances, because Q′M = 0. Power check: the oracle (true σ²) in the same cells.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

function check(; n = 900, dsurf = 12, R = 1000, law = :normal, var = :smooth, seed = 5, share = 0.9)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hcat(surfbasis(u, v, dsurf), x)
    D = tc_design(B, W)
    s = sin.(2π .* u) .* cos.(π .* v)
    f = D.Q * (D.Q' * s) .+ x
    σ2 = var == :homo ? ones(n) :
         var == :smooth ? exp.(3 .* (u .+ v .- 1)) :
         exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    Vq = 2 * dot(σ2, D.As2 * σ2)
    b0 = Mproj(D, D.Wd * f)
    c = sqrt(share / (1 - share) * Vq / sum(b0 .^ 2 .* σ2))   # Vl = c² Σ b0²σ²
    f = c .* f; b = c .* b0
    Vl = sum(b .^ 2 .* σ2); V = Vq + Vl
    H2 = D.G
    parts(s, a) = (2 * dot(s, D.As2 * s),
                   sum((a .^ 2 .- vec(sum((H2 * (D.Q' * (s .* D.Q))) .* H2, dims = 2))) .* s))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    keys_ = (:oracle, :hc2, :lo)
    q = Dict(k => zeros(R) for k in keys_); l = Dict(k => zeros(R) for k in keys_)
    for r in 1:R
        ε = sqrt.(σ2) .* draw()
        y0 = f .+ ε
        û = Mproj(D, y0)
        a = Mproj(D, D.Wd * (y0 .- û))
        for (k, sv) in ((:oracle, σ2), (:hc2, û .^ 2 ./ D.Mii), (:lo, y0 .* û ./ D.Mii))
            q[k][r], l[k][r] = parts(sv, a)
        end
    end
    zt(x, t) = (mean(x) - t) / (std(x) / sqrt(R))
    @printf("%-6s %-6s K %3d K/n %.3f Vl/V %.2f |", law, var, D.K, D.K / n, Vl / V)
    for k in keys_
        tot = q[k] .+ l[k]
        @printf(" %s %.3f (z %5.1f) [quad %.3f lin %.3f] |", k, mean(tot) / V, zt(tot, V), mean(q[k]) / Vq, mean(l[k]) / Vl)
    end
    @printf(" sd(V̂)/V: hc2 %.3f lo %.3f\n", std(q[:hc2] .+ l[:hc2]) / V, std(q[:lo] .+ l[:lo]) / V)
end

t = @elapsed check(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. E V̂/V with its z (R = 1000; |z| > 3 is a real bias).\n")
for law in (:normal, :chisq), var in (:homo, :smooth, :rough)
    for dsurf in (8, 12, 17, 21, 25)
        check(; dsurf, law, var)
    end
    println()
end
