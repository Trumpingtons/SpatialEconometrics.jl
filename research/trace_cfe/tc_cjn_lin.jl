# Theory step 2: the exact bias of CJN's linear part, and a full plug-in correction of CJN's V̂.
#
# At ρ₀ with f ∈ span B: â = b + η, η = Hε, H = MWP_B, and the noise term is noiseᵢ = Σₖ Hᵢₖ² sₖ, with
# s = Λû², Λ = (M∘M)⁻¹. Using E[XY²]-type identities for linear forms in independent ε and Λ(M∘M) = I:
#     E[ηᵢsᵢ] = Hᵢᵢμ₃ᵢ,
#     E[ηᵢ²sᵢ] = σᵢ² Σₗ Hᵢₗ²σₗ² + 2Σₖ Λᵢₖ(HΣM)ᵢₖ² + Hᵢᵢ²κᵢ,
#     E[noiseᵢsᵢ] = σᵢ² Σₖ Hᵢₖ²σₖ² + 2[(H∘H)Λ((MΣM)∘(MΣM))Λ]ᵢᵢ + Hᵢᵢ²κᵢ,
# so the fourth cumulants cancel and the linear part's bias is
#     2Σᵢ bᵢHᵢᵢμ₃ᵢ + 2Σᵢₖ Λᵢₖ(HΣM)ᵢₖ² − 2Σᵢ[(H∘H)Λ((MΣM)∘(MΣM))Λ]ᵢᵢ        (skew, cross, noise terms).
# Checks: (1) the predicted ratio against the simulated mean of CJN's linear part, by z; (2) each term's size;
# (3) the corrected V̂: CJN minus the plug-in quadratic bias (tc_cjn_bias.jl) minus the plug-in cross and
# noise terms, with Σ replaced by Diag(s). The skew term is left out of the correction, because μ₃ᵢ has no
# usable one-observation estimate. Design: tc_vhat_lo_hat.jl's hat basis, both signal strengths, smooth and
# rough variances, normal and χ²₃ errors, R = 1000, at ρ₀.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

MSM(D, s) = (SQ = s .* D.Q; QS = D.Q' * SQ; Diagonal(s) .- D.Q * SQ' .- SQ * D.Q' .+ D.Q * (QS * D.Q'))
HSM(D, s) = (GS = D.G * (D.Q' .* s'); GS .- (GS * D.Q) * D.Q')     # H Diag(s) M with H = G Q′

function cell(; n = 900, kh = 10, R = 1000, law = :normal, var = :rough, signal = :weak, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    D = tc_design(hcat(hatbasis(u, v, kh), x), W)
    f = D.Q * (D.Q' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    σ2 = var == :homo ? ones(n) : var == :smooth ? exp.(3 .* (u .+ v .- 1)) : exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    Ω = D.As2
    Vq = 2 * dot(σ2, Ω * σ2)
    if signal == :strong
        b0 = Mproj(D, D.Wd * f); f .*= sqrt(9 * Vq / sum(b0 .^ 2 .* σ2))
    end
    b = Mproj(D, D.Wd * f)
    Vl = sum(b .^ 2 .* σ2)
    Λ = inv(D.M .* D.M)
    T = Λ * Ω * Λ
    H = D.G * D.Q'
    Hd = diag(H)
    A1 = (H .^ 2) * Λ
    μ3 = law == :normal ? zeros(n) : sqrt(8 / 3) .* σ2 .^ 1.5
    quadb(s) = 4 * sum(T .* MSM(D, s) .^ 2)
    crossb(s) = 2 * sum(Λ .* HSM(D, s) .^ 2)
    noiseb(s) = (X = MSM(D, s) .^ 2; 2 * sum((A1 * X) .* Λ'))        # Σᵢ [(H∘H)Λ X Λ]ᵢᵢ
    skew = 2 * sum(b .* Hd .* μ3); cr = crossb(σ2); nz = noiseb(σ2)
    predl = skew + cr - nz; predq = quadb(σ2)
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    lin = zeros(R); tot = zeros(R); totc = zeros(R)
    for r in 1:R
        ε = sqrt.(σ2) .* draw()
        û = Mproj(D, ε)
        a = b .+ Mproj(D, D.Wd * (ε .- û))                    # â = MW P_B y₀ = b + η
        s = Λ * (û .^ 2)
        noise = vec(sum((D.G * (D.Q' * (s .* D.Q))) .* D.G, dims = 2))
        q = 2 * dot(s, Ω * s)
        lin[r] = sum((a .^ 2 .- noise) .* s)
        tot[r] = q + lin[r]
        totc[r] = tot[r] - quadb(s) - crossb(s) + noiseb(s)
    end
    V = Vq + Vl
    zt(x, t) = (mean(x) - t) / (std(x) / sqrt(R))
    @printf("%-6s %-6s %-6s K %3d K/n %.3f maxh %.3f | lin: predicted %.3f simulated %.3f (z vs pred %5.1f) [skew %+.3f cross %+.3f noise %+.3f of Vl] | total: cjn %.3f (z %5.1f) corrected %.3f (z %5.1f) sd/V %.3f %.3f\n",
            law, var, signal, D.K, D.K / n, maximum(1 .- D.Mii), 1 + predl / Vl, mean(lin) / Vl, zt(lin, Vl + predl),
            skew / Vl, cr / Vl, -nz / Vl, mean(tot) / V, zt(tot, V), mean(totc) / V, zt(totc, V), std(tot) / V, std(totc) / V)
end

t = @elapsed cell(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. R = 1000; ratios over the true linear part Vl or the true V.\n")
for signal in (:weak, :strong), law in (:normal, :chisq), var in (:smooth, :rough)
    for kh in (7, 10, 13, 15)
        cell(; kh, law, var, signal)
    end
    println()
end
