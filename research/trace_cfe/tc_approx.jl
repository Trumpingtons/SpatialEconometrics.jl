# Remark 1 / Proposition 3 (THEORY.md): approximation error, f ∉ span B.
#
# For the coded moment g(ρ) = y₀(ρ)′A y₀(ρ), A = M(W + DM) with diag A = 0, E g(ρ₀) = f′Af exactly, and with
# r = Mf (the approximation error) and r_W = MWP_Bf (the part of the lagged fit outside span B),
#     β = f′Af = r′(W + D)r + r′r_W.
# The sketched remark kept only the first term. Here β, its two terms, V = Var g(ρ₀) (exact, at the true σ²),
# c = β/√V and J = −E g′(ρ₀) are computed exactly; the predictions are a ρ̂ bias of β/J and a coverage of
# P(|Z + c| ≤ 1.96). They are checked against simulation (R = 1000) of the TC-CFE at its feasible root with
# the HC2 V̂. Design: n = 900, kNN W (k = 6), tensor hat basis k×k plus one covariate x, ρ₀ = 0.4, smooth
# variances, normal errors, f = a·sin(2πu)cos(πv) + x, which the hat basis cannot represent exactly. The
# amplitude a = 1 and a = 4: β grows like a², √V like a when the linear part dominates, so c grows with a.
# Power check: the first term alone (the old remark) predicts a different bias where r′r_W matters.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function cell(; n = 900, kh = 5, amp = 1.0, R = 1000, ρ0 = 0.4, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    D = tc_design(hcat(hatbasis(u, v, kh), x), W)
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    r = Mproj(D, f); Pf = f .- r
    rW = Mproj(D, D.Wd * Pf)
    t1 = dot(r, D.Wd * r) + dot(r, D.d .* r); t2 = dot(r, rW)
    β = dot(f, D.A * f)
    bt = (D.A + D.A') * f
    V = 2 * dot(σ2, D.As2 * σ2) + sum(bt .^ 2 .* σ2)
    Gm = D.Wd / (I - ρ0 * D.Wd)
    J = 2 * (dot(Gm * f, D.As * f) + tr(Gm' * D.As * Diagonal(σ2)))
    c = β / sqrt(V)
    pcov = cdf(Normal(), 1.96 - c) - cdf(Normal(), -1.96 - c)
    Fac = lu(sparse(I - ρ0 * W))
    ρs = zeros(R); cov = 0
    for k in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n))
        Wy = W * y; W2y = W * Wy
        e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
        ρ, a, b = cfe_root(e0, e1 .+ D.d .* e0, e1, e2 .+ D.d .* e1)
        ρs[k] = ρ
        Vh = tc_varg(D, W, y, ρ).robust
        cov += Vh > 0 && abs(ρ - ρ0) <= 1.96 * sqrt(Vh) / abs(b - 2ρ * a)
    end
    bias = mean(ρs) - ρ0; se = std(ρs) / sqrt(R)
    @printf("amp %.0f K %3d ‖r‖∞ %.4f ‖r_W‖∞ %.4f | β %+8.3f = r′(W+D)r %+8.3f + r′r_W %+8.3f | c = β/√V %+.3f | ρ̂ bias: predicted β/J %+.4f, first term only %+.4f, simulated %+.4f (z vs pred %5.1f) sd %.4f | coverage: predicted %.3f simulated %.3f\n",
            amp, D.K, maximum(abs.(r)), maximum(abs.(rW)), β, t1, t2, c, β / J, t1 / J, bias, (bias - β / J) / se, std(ρs), pcov, cov / R)
end

t = @elapsed cell(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. R = 1000; MC SE of a coverage near 0.95: 0.007.\n")
t0 = time()
for amp in (1.0, 4.0)
    for kh in (3, 4, 5, 7, 10, 13)
        cell(; kh, amp)
    end
    println()
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
