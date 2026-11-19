# Theory step 2: is leverage near 1 the cause of the leave-one-out collapse?
#
# tc_vhat_lo_lin.jl (Chebyshev surface basis, leverage reaching 1 from K/n ≈ 0.19) found the leave-one-out
# quadratic part collapsing with a strong signal. THEORY.md argues that at a unit with Mᵢᵢ = 0 no estimator
# invariant to f ∈ span B can see σᵢ², so leverage bounded away from 1 is necessary. This script repeats
# that check with a local tensor hat basis, whose leverage stays bounded at the same K/n, and adds Cattaneo,
# Jansson & Newey's (M∘M)⁻¹û², which is f-free and becomes usable when M∘M is well conditioned.
# Prediction: with bounded leverage the leave-one-out quadratic part stays near 1 at every K/n.
# Power check: the oracle (true σ²), and the Chebyshev results in tc_vhat_lo_lin_output.txt.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function check(; n = 900, kh = 10, R = 1000, law = :normal, var = :smooth, seed = 5, share = 0.9)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hcat(hatbasis(u, v, kh), x)
    D = try
        tc_design(B, W)
    catch e
        e isa SingularException || rethrow()
        @printf("%-6s %-6s K/n %.3f: M∘M is singular (some unit has leverage 1), so the exact-diagonal moment and CJN do not exist; cell skipped
", law, var, rank(B) / n)
        return
    end
    s = sin.(2π .* u) .* cos.(π .* v)
    f = D.Q * (D.Q' * s) .+ x
    σ2 = var == :homo ? ones(n) :
         var == :smooth ? exp.(3 .* (u .+ v .- 1)) :
         exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    Vq = 2 * dot(σ2, D.As2 * σ2)
    b0 = Mproj(D, D.Wd * f)
    c = share === nothing ? 1.0 : sqrt(share / (1 - share) * Vq / sum(b0 .^ 2 .* σ2))   # Vl = c² Σ b0²σ²
    f = c .* f; b = c .* b0
    Vl = sum(b .^ 2 .* σ2); V = Vq + Vl
    H2 = D.G
    parts(s, a) = (2 * dot(s, D.As2 * s),
                   sum((a .^ 2 .- vec(sum((H2 * (D.Q' * (s .* D.Q))) .* H2, dims = 2))) .* s))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    MMf = lu(D.M .* D.M)
    keys_ = (:oracle, :hc2, :lo, :cjn)
    q = Dict(k => zeros(R) for k in keys_); l = Dict(k => zeros(R) for k in keys_)
    for r in 1:R
        ε = sqrt.(σ2) .* draw()
        y0 = f .+ ε
        û = Mproj(D, y0)
        a = Mproj(D, D.Wd * (y0 .- û))
        for (k, sv) in ((:oracle, σ2), (:hc2, û .^ 2 ./ D.Mii), (:lo, y0 .* û ./ D.Mii), (:cjn, MMf \ (û .^ 2)))
            q[k][r], l[k][r] = parts(sv, a)
        end
    end
    zt(x, t) = (mean(x) - t) / (std(x) / sqrt(R))
    @printf("%-6s %-6s K %3d K/n %.3f maxh %.3f Vl/V %.2f |", law, var, D.K, D.K / n, maximum(1 .- D.Mii), Vl / V)
    for k in keys_
        tot = q[k] .+ l[k]
        @printf(" %s %.3f (z %5.1f) [quad %.3f lin %.3f] |", k, mean(tot) / V, zt(tot, V), mean(q[k]) / Vq, mean(l[k]) / Vl)
    end
    @printf(" sd(V̂)/V: hc2 %.3f lo %.3f\n", std(q[:hc2] .+ l[:hc2]) / V, std(q[:lo] .+ l[:lo]) / V)
end

t = @elapsed check(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. E V̂/V with its z (R = 1000; |z| > 3 is a real bias).\n")
# `julia tc_vhat_lo_hat.jl weak` keeps tc_vhat.jl's unscaled f, so the quadratic part dominates V (its output
# is tc_vhat_lo_hat_weak_output.txt); with no argument the linear part is 90% of V.
const SHARE = "weak" in ARGS ? nothing : 0.9
println(SHARE === nothing ? "f unscaled (weak signal)" : "f scaled so that Vl/V = $SHARE")
for law in (:normal, :chisq), var in (:homo, :smooth, :rough)
    for kh in (7, 10, 13, 15, 19)
        check(; kh, law, var, share = SHARE)
    end
    println()
end
