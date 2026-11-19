# Proposition 4(c) (THEORY.md): the HC2 V̂ with a penalised smoother, sᵢ = ûᵢ²/(M²)ᵢᵢ.
#
# At ρ₀, û = z + r with z = Mε and r = Mf, the shrinkage. With mᵢ = (M²)ᵢᵢ, s⁰ = z²/m (r removed, infeasible)
# and δᵢ = (2rᵢzᵢ + rᵢ²)/mᵢ, the feasible s = s⁰ + δ. Both versions use the same â = Hy₀ (H = (A + A′)S;
# mean μ = (A + A′)Sf = b − (A + A′)r, η = Hε), so their gap comes from s alone, and under normal errors
# its mean is exact, with Γ = MΣM, Ω = As∘As, N = H∘H:
#     E[V̂(s) − V̂(s⁰)] = 2[2Σ Ωᵢⱼ Γᵢᵢ rⱼ²/(mᵢmⱼ) + Σ Ωᵢⱼ (4rᵢrⱼΓᵢⱼ + rᵢ²rⱼ²)/(mᵢmⱼ)]
#                     + Σᵢ (4rᵢμᵢ(HΣM)ᵢᵢ + rᵢ²(μᵢ² + (HΣH′)ᵢᵢ))/mᵢ
#                     − Σᵢₖ Nᵢₖ [rₖ²Γᵢᵢ + rᵢ²Γₖₖ + 4rᵢrₖΓᵢₖ + rᵢ²rₖ²]/(mᵢmₖ).
# Check: mean V̂/V with the feasible s, against the same with r removed from s, and the gap against the
# exact value (z statistic). Power check: the first-order shift with b in place of μ,
#     Δ₁ = 4σ²′Ω(r²/m) + Σ bᵢ²rᵢ²/mᵢ, must miss where (A + A′)r is not small.
# Separately, â estimates the shrunk signal μ, not b, so even with r removed from s, V̂ falls short of V by
#     Δ_b = Σᵢ (μᵢ² − bᵢ²)σᵢ² ≤ 0 (printed as 1 + Δ_b/V, next to the r-removed mean).
# Same design as tc_smoother.jl, at ρ₀ (no root). R = 500.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function cell(; n = 900, kh = 15, λ = 1.0, amp = 1.0, R = 500, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]; Wd = Matrix(W)
    x = randn(rng, n)
    B = hcat(hatbasis(u, v, kh), x)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ωh = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    Ωp = zeros(kh^2 + 1, kh^2 + 1); Ωp[1:kh^2, 1:kh^2] = Ωh
    S = B * ((B' * B + λ * Ωp) \ B'); S = (S + S') / 2
    M = I - S; MW = M * Wd
    d = (M .* M) \ (-diag(MW))
    A = MW + M * Diagonal(d) * M
    As = (A + A') / 2; As2 = As .^ 2; Ab = A + A'; H = Ab * S
    m = diag(M * M)
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    b = Ab * f; r = M * f; br = Ab * r
    V = 2 * dot(σ2, As2 * σ2) + sum(b .^ 2 .* σ2)
    Δ1 = 4 * dot(σ2, As2 * (r .^ 2 ./ m)) + sum(b .^ 2 .* r .^ 2 ./ m)
    Γ = M * Diagonal(σ2) * M; gd = diag(Γ); μ = b .- br; N = H .^ 2
    HΣM = diag(H * Diagonal(σ2) * M); HΣH = vec(sum((H .^ 2) .* σ2', dims = 2))
    ρm = r ./ m; r2m = r .^ 2 ./ m; gm = gd ./ m
    quad = 2 * (2 * dot(gm, As2 * r2m) + 4 * dot(ρm, (As2 .* Γ) * ρm) + dot(r2m, As2 * r2m))
    lin = sum((4 .* r .* μ .* HΣM .+ r .^ 2 .* (μ .^ 2 .+ HΣH)) ./ m)
    nse = dot(gm, N * r2m) + dot(r2m, N * gm) + 4 * dot(ρm, (N .* Γ) * ρm) + dot(r2m, N * r2m)
    Δx = quad + lin - nse
    Δb = sum((μ .^ 2 .- b .^ 2) .* σ2)
    vh(y0, s) = (â = H * y0; noise = vec(sum((H .^ 2) .* s', dims = 2));
                 2 * dot(s, As2 * s) + sum((â .^ 2 .- noise) .* s))
    q1 = zeros(R); q0 = zeros(R)
    for k in 1:R
        ε = sqrt.(σ2) .* randn(rng, n)
        y0 = f .+ ε                                   # at ρ₀
        z = M * ε
        q1[k] = vh(y0, (z .+ r) .^ 2 ./ m) / V       # feasible: û = z + r
        q0[k] = vh(y0, z .^ 2 ./ m) / V              # r removed from s (infeasible)
    end
    g = q1 .- q0; se = std(g) / sqrt(R)
    @printf("amp %.0f λ %7.3f tr S %6.1f ‖r‖²/n %.4f max|r| %.3f | mean V̂/V: feasible %.3f, r removed %.3f (1 + Δ_b/V %.3f) | gap %.4f ± %.4f, exact %.4f (z %5.1f), first order with b %.4f\n",
            amp, λ, tr(S), sum(abs2, r) / n, maximum(abs.(r)), mean(q1), mean(q0), 1 + Δb / V, mean(g), se, Δx / V, (mean(g) - Δx / V) / se, Δ1 / V)
end

cell(; R = 5); println("(warm-up line above)\n")
t0 = time()
for amp in (1.0, 4.0)
    for λ in (1e-3, 0.1, 1.0, 10.0, 100.0)
        cell(; λ, amp)
    end
    println()
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
