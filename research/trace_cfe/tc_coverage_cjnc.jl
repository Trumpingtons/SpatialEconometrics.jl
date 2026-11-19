# Theory step 2: coverage of ρ̂ with CJN's V̂ corrected for both parts' exact bias (tc_cjn_bias.jl,
# tc_cjn_lin.jl), against HC2 and plain CJN, at the feasible root.
#
# Same design, seeds and estimator as tc_coverage_vhat.jl, so its oracle, HC2 and CJN columns must repeat
# exactly (a check of the check). The corrected V̂ is CJN's minus the plug-in quadratic bias
# 4tr(ΩΛ[(MSM)∘(MSM)]Λ) and the plug-in cross and noise terms, with S = Diag(s) and s = (M∘M)⁻¹û² at ρ̂;
# the skewness term is left out. tc_cjn_lin.jl found its E V̂/V within 1.6% at K/n ≤ 0.19 and worse than
# plain CJN at 0.25. A negative V̂ counts as not covering and is reported.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
MSM(D, s) = (SQ = s .* D.Q; QS = D.Q' * SQ; Diagonal(s) .- D.Q * SQ' .- SQ * D.Q' .+ D.Q * (QS * D.Q'))
HSM(D, s) = (GS = D.G * (D.Q' .* s'); GS .- (GS * D.Q) * D.Q')

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
    Λ = inv(D.M .* D.M); T = Λ * D.As2 * Λ; A1 = ((D.G * D.Q') .^ 2) * Λ
    corr(s) = (X = MSM(D, s) .^ 2; -4 * sum(T .* X) - 2 * sum(Λ .* HSM(D, s) .^ 2) + 2 * sum((A1 * X) .* Λ'))
    Fac = lu(sparse(I - ρ0 * W))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    ks = (:oracle, :hc2, :cjn, :cjnc)
    cov = Dict(k => 0 for k in ks); neg = Dict(k => 0 for k in ks); ρs = zeros(R)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* draw())
        Wy = W * y; W2y = W * Wy
        e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
        ρ, a, b = cfe_root(e0, e1 .+ D.d .* e0, e1, e2 .+ D.d .* e1)
        ρs[r] = ρ
        y0 = y .- ρ .* Wy; û = Mproj(D, y0)
        hc2 = û .^ 2 ./ D.Mii
        sc = MMf \ (û .^ 2)
        for (k, s) in ((:oracle, σ2), (:hc2, hc2), (:cjn, sc), (:cjnc, sc))
            V = vparts(D, W, y, ρ, s) + (k == :cjnc ? corr(s) : 0.0)
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
t0 = time()
for signal in (:weak, :strong), law in (:normal, :chisq), var in (:smooth, :rough)
    for kh in (7, 10, 13, 15)
        cell(; kh, law, var, signal)
    end
    println()
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
