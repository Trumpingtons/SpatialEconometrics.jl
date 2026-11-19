# Theory step 2: coverage of ρ̂ with each variance estimate, at the feasible root.
#
# tc_vhat_lo_hat.jl measured E V̂/V at the true ρ. This script asks the question that matters: does the 95%
# interval ρ̂ ± 1.96 √V̂/|g′(ρ̂)| cover ρ₀? The estimator is the exact-diagonal TC-CFE of fit_all (tc_lib.jl),
# and V̂ is evaluated at ρ̂ with four plug-ins: the oracle (true σ²), HC2 ûᵢ²/Mᵢᵢ (the coded one; Proposition 2
# covers it for K/n → 0), Kline, Saggio & Sølvsten's leave-one-out y₀ᵢûᵢ/Mᵢᵢ, and Cattaneo, Jansson & Newey's
# (M∘M)⁻¹û². Design: tc_vhat_lo_hat.jl's (n = 900, kNN W, tensor hat basis plus one covariate), y generated
# with ρ₀ = 0.4, R = 1000 (MC SE of a coverage near 0.95: 0.007). "weak" keeps tc_vhat.jl's f; "strong"
# scales f so the linear part is 90% of V at ρ₀. A negative V̂ counts as not covering and is reported.
# Power check: the oracle's coverage, and the homoskedastic V̂ (pooled σ̂²) under rough variances.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function vparts(D, W, y, ρ, s)
    y0 = y .- ρ .* (W * y)
    u = Mproj(D, y0)
    a = Mproj(D, W * (y0 .- u))
    noise = vec(sum((D.G * (D.Q' * (s .* D.Q))) .* D.G, dims = 2))
    2 * dot(s, D.As2 * s) + sum((a .^ 2 .- noise) .* s)
end

function cell(; n = 900, kh = 10, R = 1000, law = :normal, var = :rough, signal = :weak, ρ0 = 0.4, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    D = tc_design(hcat(hatbasis(u, v, kh), x), W)
    f = D.Q * (D.Q' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    σ2 = var == :homo ? ones(n) : var == :smooth ? exp.(3 .* (u .+ v .- 1)) : exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    if signal == :strong
        Vq = 2 * dot(σ2, D.As2 * σ2); b0 = Mproj(D, D.Wd * f)
        f .*= sqrt(9 * Vq / sum(b0 .^ 2 .* σ2))
    end
    MMf = lu(D.M .* D.M)
    Fac = lu(sparse(I - ρ0 * W))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    ks = (:oracle, :hc2, :lo, :cjn, :homo)
    cov = Dict(k => 0 for k in ks); neg = Dict(k => 0 for k in ks); ρs = zeros(R)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* draw())
        Wy = W * y; W2y = W * Wy
        e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
        ρ, a, b = cfe_root(e0, e1 .+ D.d .* e0, e1, e2 .+ D.d .* e1)
        ρs[r] = ρ
        y0 = y .- ρ .* Wy; û = Mproj(D, y0)
        hc2 = û .^ 2 ./ D.Mii
        for (k, s) in ((:oracle, σ2), (:hc2, hc2), (:lo, y0 .* û ./ D.Mii), (:cjn, MMf \ (û .^ 2)), (:homo, fill(mean(hc2), n)))
            V = vparts(D, W, y, ρ, s)
            if V <= 0
                neg[k] += 1
            elseif abs(ρ - ρ0) <= 1.96 * sqrt(V) / abs(b - 2ρ * a)
                cov[k] += 1
            end
        end
    end
    @printf("%-6s %-6s %-6s K %3d K/n %.3f maxh %.3f | bias %+.4f sd %.4f |", law, var, signal, D.K, D.K / n,
            maximum(1 .- D.Mii), mean(ρs) - ρ0, std(ρs))
    for k in ks
        @printf(" %s %.3f%s", k, cov[k] / R, neg[k] > 0 ? "(neg $(neg[k]))" : "")
    end
    println()
end

t = @elapsed cell(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. Coverage of the 95% interval, R = 1000 (MC SE 0.007).\n")
for signal in (:weak, :strong), law in (:normal, :chisq), var in (:smooth, :rough)
    for kh in (7, 10, 13, 15)
        cell(; kh, law, var, signal)
    end
    println()
end
