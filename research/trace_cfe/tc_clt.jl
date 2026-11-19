# Theory step 1, numerical side: the conditions of a CLT for the zero-diagonal moment when M is dense.
#
# At the true ρ, with f in span(B), the moment is g = b′ε + ε′Cε with C = PM, diag(C) = 0, b = MP′f,
# for Jin & Lee's first-step P = W′ − Δ, Δ = Diag(W′M) Diag(M)⁻¹. Its variance is exact for independent ε
# with any third and fourth moments:  V = 2σ′(Cs∘Cs)σ + Σ bᵢ²σᵢ²,  σ = the vector of variances.
#
# The martingale-CLT argument (THEORY.md) needs, as n grows:
#   (a) ‖Cs‖_op bounded and ‖Cs‖_F²/n bounded below, so ‖Cs‖_op/‖Cs‖_F → 0;
#   (b) the same for the strictly lower-triangular part L of Cs, up to a log n factor: ‖L‖_op/‖Cs‖_F → 0;
#   (c) Lindeberg for the linear part: max bᵢ²/‖b‖² → 0;
# and predicts |Δᵢᵢ| ≤ ‖W eᵢ‖ √hᵢᵢ/(1 − hᵢᵢ), so Δ → 0 when the largest leverage max hᵢᵢ → 0.
# For each design this prints the quantities, then the distribution of z = g/√V over R draws of
# heteroskedastic, standardised χ²₃ errors. The power check is a W of two complete groups, where
# ‖Cs‖_op/‖Cs‖_F does not vanish, so the CLT should fail.
include(joinpath(@__DIR__, "tc_lib.jl"))

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function groupW(n, G, rng)
    g = rand(rng, 1:G, n); I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for i in 1:n
        nb = [j for j in 1:n if g[j] == g[i] && j != i]
        append!(I_, fill(i, length(nb))); append!(J_, nb); append!(V_, fill(1 / length(nb), length(nb)))
    end
    sparse(I_, J_, V_, n, n)
end

skew(z) = mean(((z .- mean(z)) ./ std(z)) .^ 3)
exkurt(z) = mean(((z .- mean(z)) ./ std(z)) .^ 4) - 3
function ks(z)                                     # Kolmogorov–Smirnov distance to N(0, 1)
    s = sort(z); n = length(s)
    Φ(x) = 0.5 * (1 + erf_(x / sqrt(2)))
    maximum(max(abs(i / n - Φ(s[i])), abs((i - 1) / n - Φ(s[i]))) for i in 1:n)
end
function erf_(x)                                   # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592)t * exp(-x^2)
    sign(x) * y
end

function design(; n, basis, k, Wkind = :knn, R = 2000, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = Wkind == :knn ? knnW(P, 6) : groupW(n, 2, rng)
    B = basis == :hat ? hatbasis(u, v, k) : surfbasis(u, v, k)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; K = size(Q, 2)
    h = vec(sum(Q .^ 2, dims = 2)); M = I - Q * Q'; Wd = Matrix(W)
    δ = diag(Wd' * M) ./ diag(M)
    Pm = Wd' .- Diagonal(δ); C = Pm * M; Cs = (C + C') / 2
    f = Q * (Q' * (sin.(2π .* u) .* cos.(π .* v)))
    f .*= sqrt(n) / norm(f)
    b = M * (Pm' * f)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    V = 2 * dot(σ2, (Cs .^ 2) * σ2) + sum(b .^ 2 .* σ2)
    L = tril(Cs, -1)
    opC = opnorm(Cs); FC = norm(Cs); opL = opnorm(L)
    bound = maximum(vec(sqrt.(sum(Wd .^ 2, dims = 1))) .* sqrt.(h) ./ (1 .- h))
    z = zeros(R)
    for r in 1:R
        e = randn(rng, n, 3); ε = sqrt.(σ2) .* (vec(sum(e .^ 2, dims = 2)) .- 3) ./ sqrt(6)
        z[r] = (dot(b, ε) + dot(ε, C * ε)) / sqrt(V)
    end
    r3(x) = round(x, digits = 3)
    println(rpad("$(Wkind) $(basis) n=$n K=$K", 26),
        " K/n $(r3(K/n)) maxh $(r3(maximum(h))) | max|Δ| $(r3(maximum(abs.(δ)))) ≤ $(r3(bound)) | ‖Cs‖op $(r3(opC)) ‖Cs‖F²/n $(r3(FC^2/n)) op/F $(r3(opC/FC)) ‖L‖op/F $(r3(opL/FC)) | Lind_b $(r3(maximum(b .^ 2) / sum(b .^ 2))) | z: mean $(r3(mean(z))) sd $(r3(std(z))) skew $(r3(skew(z))) exk $(r3(exkurt(z))) size $(r3(mean(abs.(z) .> 1.96))) KS $(r3(ks(z)))")
end

design(; n = 400, basis = :hat, k = 4, R = 50)   # warm-up
println("Monte Carlo SEs at R = 2000: mean 0.022, sd 0.016, skew 0.055, excess kurtosis 0.11, size 0.005; KS 95% critical value 0.030\n")
println("Local (hat) basis, K ≈ 2√n: max leverage → 0")
for n in (400, 900, 1600, 2500); design(; n, basis = :hat, k = round(Int, sqrt(2sqrt(n)))); end
println("\nLocal (hat) basis, K ≈ 0.15 n: K/n fixed, max leverage does not vanish")
for n in (400, 900, 1600, 2500); design(; n, basis = :hat, k = round(Int, sqrt(0.15n))); end
println("\nGlobal polynomial (Chebyshev) basis, K ≈ 2√n: boundary leverage")
for n in (400, 900, 1600, 2500); design(; n, basis = :cheb, k = round(Int, (sqrt(1 + 16sqrt(n)) - 3) / 2)); end
println("\nPower check: W of two complete groups, hat basis K ≈ 2√n (‖Cs‖op/‖Cs‖F should not vanish)")
for n in (400, 900, 1600); design(; n, basis = :hat, k = round(Int, sqrt(2sqrt(n))), Wkind = :groups); end
