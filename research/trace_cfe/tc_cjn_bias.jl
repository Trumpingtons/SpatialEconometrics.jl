# Theory step 2: the exact bias of CJN's quadratic part, and a plug-in correction.
#
# With s = Λû², Λ = (M∘M)⁻¹, E s = σ² exactly (at ρ₀, f ∈ span B), so E[2s′Ωs] − 2σ²′Ωσ² = 2 tr(Ω Cov s),
# Ω = Cs∘Cs. Cov s = Λ Cov(û²) Λ, and Cov(ûᵢ², ûⱼ²) = 2(MΣM)ᵢⱼ² + Σₖ Mᵢₖ²Mⱼₖ²κₖ (κ the fourth cumulants).
# The cumulant part gives Λ(M∘M)Diag(κ)(M∘M)Λ = Diag(κ), and Ω has a zero diagonal, so it drops out:
#     bias = 4 tr(Ω Λ [(MΣM)∘(MΣM)] Λ)   for any error law.
# Checks: (1) the predicted ratio 1 + bias/Vq against the simulated mean of CJN's quadratic part, by z;
# (2) normal and χ²₃ errors must give the same prediction and agree with it (κ ≠ 0 for χ²₃, so this tests the
# cancellation); (3) the plug-in correction, subtracting 4 tr(ΩΛ[(MSM)∘(MSM)]Λ) with S = Diag(s), as a new
# estimator. Design: tc_vhat_lo_hat.jl's (n = 900, kNN W, tensor hat basis plus one covariate), R = 1000, at ρ₀.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

MSM(D, s) = (SQ = s .* D.Q; QS = D.Q' * SQ; Diagonal(s) .- D.Q * SQ' .- SQ * D.Q' .+ D.Q * (QS * D.Q'))

function cell(; n = 900, kh = 10, R = 1000, law = :normal, var = :rough, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    D = tc_design(hcat(hatbasis(u, v, kh), x), W)
    σ2 = var == :homo ? ones(n) : var == :smooth ? exp.(3 .* (u .+ v .- 1)) : exp.(randn(rng, n))
    σ2 ./= mean(σ2)
    Ω = D.As2
    Λ = inv(D.M .* D.M)
    T = Λ * Ω * Λ                                   # tr(ΩΛXΛ) = Σᵢⱼ Tᵢⱼ Xᵢⱼ for symmetric X
    Vq = 2 * dot(σ2, Ω * σ2)
    pred = 4 * sum(T .* MSM(D, σ2) .^ 2)
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    q = zeros(R); qc = zeros(R)
    for r in 1:R
        û = Mproj(D, sqrt.(σ2) .* draw())
        s = Λ * (û .^ 2)
        q[r] = 2 * dot(s, Ω * s)
        qc[r] = q[r] - 4 * sum(T .* MSM(D, s) .^ 2)
    end
    zt(x, t) = (mean(x) - t) / (std(x) / sqrt(R))
    @printf("%-6s %-6s K %3d K/n %.3f maxh %.3f | predicted %.3f simulated %.3f (z vs predicted %5.1f) | corrected %.3f (z %5.1f) sd/Vq: cjn %.3f corrected %.3f\n",
            law, var, D.K, D.K / n, maximum(1 .- D.Mii), 1 + pred / Vq, mean(q) / Vq, zt(q, Vq + pred),
            mean(qc) / Vq, zt(qc, Vq), std(q) / Vq, std(qc) / Vq)
end

t = @elapsed cell(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. Ratios are over the true quadratic part Vq; R = 1000.\n")
for law in (:normal, :chisq), var in (:homo, :smooth, :rough)
    for kh in (7, 10, 13, 15)
        cell(; kh, law, var)
    end
    println()
end
