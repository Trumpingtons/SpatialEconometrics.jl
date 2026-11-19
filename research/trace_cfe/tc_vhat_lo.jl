# Theory step 2: Kline, Saggio & Sølvsten's (2020) leave-one-out variances in the robust V̂.
#
# tc_vhat.jl found the HC2 plug-in sᵢ = ûᵢ²/Mᵢᵢ biased upward under rough heteroskedasticity, by up to 13–17%
# at K/n ≈ 0.3–0.4, because E ûᵢ² = Σⱼ Mᵢⱼ²σⱼ² mixes in other units' variances; and Cattaneo, Jansson &
# Newey's (M∘M)⁻¹û² fails because the leverage reaches 1. KSS's leave-one-out estimate is
#     sᵢ = y₀ᵢ ûᵢ / Mᵢᵢ,   y₀ = y − ρWy = f + ε,   û = My₀ = Mε (f ∈ span B),
# and E[y₀ᵢûᵢ] = E[εᵢ Σⱼ Mᵢⱼεⱼ] = Mᵢᵢσᵢ², so E sᵢ = σᵢ² exactly, for any variances and any Mᵢᵢ > 0.
# It equals y₀ᵢ(y₀ᵢ − ŷ₋ᵢ), the residual of a fit that leaves unit i out.
#
# The linear part's noise term is linear in s, so it becomes unbiased too. Two biases remain, both from
# products of dependent estimates: Σᵢⱼ Asᵢⱼ² sᵢsⱼ in the quadratic part, and Σᵢ âᵢ²sᵢ in the linear part.
# KSS remove the first by sample splitting; this script measures how large they are before building that.
#
# tc_vhat.jl's design exactly (same seed, n = 900, kNN W, Chebyshev surface basis plus one covariate, the
# exact-diagonal moment A = MW + MDM, R = 1000, evaluated at the true ρ). For each estimator, E V̂/V with
# its z, and the two parts separately: quad = 2s′(As∘As)s against 2σ²′(As∘As)σ², and
# lin = Σ(âᵢ² − noiseᵢ)sᵢ against Σbᵢ²σᵢ².
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

function check(; n = 900, dsurf = 12, R = 1000, law = :normal, var = :smooth, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hcat(surfbasis(u, v, dsurf), x)
    D = tc_design(B, W)
    h = 1 .- D.Mii
    s = sin.(2π .* u) .* cos.(π .* v)
    f = D.Q * (D.Q' * s) .+ x
    σ2 = var == :homo ? ones(n) :
         var == :smooth ? exp.(3 .* (u .+ v .- 1)) :
         exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    b = Mproj(D, D.Wd * f)
    Vq = 2 * dot(σ2, D.As2 * σ2); Vl = sum(b .^ 2 .* σ2); V = Vq + Vl
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
    @printf("%-6s %-6s K %3d K/n %.3f maxh %.3f |", law, var, D.K, D.K / n, maximum(h))
    for k in keys_
        tot = q[k] .+ l[k]
        @printf(" %s %.3f (z %5.1f) [quad %.3f lin %.3f] |", k, mean(tot) / V, zt(tot, V), mean(q[k]) / Vq, mean(l[k]) / Vl)
    end
    @printf(" sd(V̂)/V: hc2 %.3f lo %.3f\n", std(q[:hc2] .+ l[:hc2]) / V, std(q[:lo] .+ l[:lo]) / V)
end

t = @elapsed check(; R = 5)
println("warm-up $(round(t, digits = 1)) s. E V̂/V with its z (R = 1000; |z| > 3 is a real bias); quad and lin are each part's mean over its true value.\n")
for law in (:normal, :chisq), var in (:homo, :smooth, :rough)
    for dsurf in (8, 12, 17, 21, 25)
        check(; dsurf, law, var)
    end
    println()
end
