# Coverage of ρ̂ when the errors have no fourth moment.
#
# tc_clt_moments.jl showed that the standardised moment g/√V approaches normality slowly under t₃ and
# Pareto(3) errors. This script asks what that does to inference on ρ. Design: tc_clt.jl's hat basis,
# K ≈ 2√n, kNN W (k = 6), f in span(B), spatially smooth variances σᵢ² = exp(3(uᵢ + vᵢ − 1)) normalised
# to mean 1, y = (I − ρ₀W)⁻¹(f + ε), ρ₀ = 0.4. The estimator is Theorem 1's: Jin & Lee's (2012, §3.2)
# first step, the root of y₀(ρ)′C y₀(ρ) = 0 with C = (W′ − Δ)M and Δ = Diag(W′M)Diag(M)⁻¹ (zero diagonal).
# Three intervals ρ̂ ± 1.96·se, all with the delta-method denominator |b − 2ρ̂a| from the data:
#   robust : se² from the plug-in robust variance of g (tc_jinlee.jl's robust_var), what an analyst would use;
#   oracle : se² from the exact V = 2σ²′(Cs∘Cs)σ² + Σ bᵢ²σᵢ², b = C′f — isolates the CLT from V̂;
#   homo   : the homoskedastic plug-in, which must under-cover with these variances (the power check).
# Error laws: normal and standardised χ²₃ (all moments finite) as references; standardised t₃ and centred,
# standardised Pareto with tail index 3 (no fourth moment; A1 holds for any δ < 1).
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function draw(rng, law, n)                         # unit-variance draws
    law == :normal && return randn(rng, n)
    law == :chisq && return (vec(sum(randn(rng, n, 3) .^ 2, dims = 2)) .- 3) ./ sqrt(6)
    if law == :t3                                  # t₃ = Z/√(χ²₃/3), variance 3
        z = randn(rng, n); c = vec(sum(randn(rng, n, 3) .^ 2, dims = 2))
        return z ./ sqrt.(c ./ 3) ./ sqrt(3)
    end
    x = rand(rng, n) .^ (-1 / 3)                   # Pareto(x_m = 1, α = 3): mean 3/2, variance 3/4
    (x .- 1.5) ./ sqrt(0.75)
end

function cell(; n, law, ρ0 = 0.4, R = 1000, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = knnW(P, 6); Wd = Matrix(W)
    k = round(Int, sqrt(2sqrt(n)))
    B = hatbasis(u, v, k)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; K = size(Q, 2)
    M = I - Q * Q'; Mii = diag(M)
    δ = diag(Wd' * M) ./ Mii
    Ct = Wd .- Diagonal(δ)                         # (W′ − Δ)′ = W − Δ
    C = Ct' * M                                    # (W′ − Δ)M, zero diagonal
    S = C + C'; S2 = (S ./ 2) .^ 2; H = S * Q
    f = Q * (Q' * (sin.(2π .* u) .* cos.(π .* v))); f .*= sqrt(n) / norm(f)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    b = C' * f
    V = 2 * dot(σ2, S2 * σ2) + sum(b .^ 2 .* σ2)  # exact variance of g at ρ₀
    Fac = lu(sparse(I - ρ0 * W))
    est = zeros(R); cov = zeros(Bool, R, 3); ser = zeros(R)
    for r in 1:R
        ε = sqrt.(σ2) .* draw(rng, law, n)
        y = Fac \ (f .+ ε); Wy = W * y
        e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
        ρ, a, bb = cfe_root(e0, Ct * y, e1, Ct * Wy)
        den = abs(bb - 2ρ * a)
        y0 = y .- ρ .* Wy
        uu = y0 .- Q * (Q' * y0); s2 = uu .^ 2 ./ Mii
        â = S * (y0 .- uu)
        noise = vec(sum((H * (Q' * (s2 .* Q))) .* H, dims = 2))
        Vrob = 2 * dot(s2, S2 * s2) + sum((â .^ 2 .- noise) .* s2)
        sb = mean(s2)
        Vhom = 2 * sb^2 * sum(S2) + sb * dot(â, â)
        se = (sqrt(max(Vrob, 0)), sqrt(V), sqrt(Vhom)) ./ den
        est[r] = ρ; ser[r] = se[1]
        for j in 1:3; cov[r, j] = abs(ρ - ρ0) <= 1.96 * se[j]; end
    end
    c = vec(mean(cov, dims = 1))
    @printf("%-7s n %5d K %4d | bias %+.4f sd %.4f | robust se/sd %.3f | coverage: robust %.3f  oracle %.3f  homo %.3f\n",
            law, n, K, mean(est) - ρ0, std(est), median(ser) / std(est), c[1], c[2], c[3])
end

t = @elapsed cell(; n = 400, law = :t3, R = 20)
println("warm-up (20 replications at n = 400) $(round(t, digits = 1)) s")
t = @elapsed cell(; n = 2500, law = :t3, R = 20)
println("20 replications at n = 2500: $(round(t, digits = 1)) s. Cells: 4 laws × 4 sizes, R = 1000 each.")
println("Monte Carlo SE of a coverage near 0.95 at R = 1000: 0.007. se/sd uses the median se.\n")
t0 = time()
for law in (:normal, :chisq, :t3, :pareto)
    for n in (400, 900, 1600, 2500); cell(; n, law); end
    println()
end
println("total $(round(time() - t0, digits = 1)) s")
