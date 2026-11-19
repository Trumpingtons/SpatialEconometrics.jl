# Theorem 1 with 2 + δ moments: the standardised moment z = g/√V when the errors have no fourth moment.
#
# The designs are tc_clt.jl's (same seed, same W, basis, f and variances; Jin & Lee's P = W′ − Δ, C = PM),
# and only the error law changes. Both laws have finite moments of order below 3 and an infinite fourth:
#   :t3     Student t with 3 degrees of freedom, standardised (symmetric);
#   :pareto Pareto with tail index 3, centred and standardised (skewed to the right).
# THEORY.md's Theorem 1 now needs only supᵢ E|εᵢ|^{2+δ} < ∞ for some δ > 0, here any δ < 1. Its proof
# bounds the error in the normal approximation by powers of ‖Cs‖_op/‖Cs‖_F and maxᵢ bᵢ²/V, with exponents
# δ/2 and p − 1 = δ/2, so the approach to normality should be slower than with χ²₃ errors (tc_clt.jl).
# The power check keeps the t₃ errors but uses tc_clt.jl's W of two complete groups, where ‖Cs‖_op/‖Cs‖_F
# does not vanish: there z must stay away from normal.
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
function erf_(x)                                   # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592)t * exp(-x^2)
    sign(x) * y
end
function ks(z)                                     # Kolmogorov–Smirnov distance to N(0, 1)
    s = sort(z); n = length(s)
    Φ(x) = 0.5 * (1 + erf_(x / sqrt(2)))
    maximum(max(abs(i / n - Φ(s[i])), abs((i - 1) / n - Φ(s[i]))) for i in 1:n)
end

# unit-variance draws with an infinite fourth moment
function draw(rng, law, n)
    if law == :t3                                  # t₃ = Z/√(χ²₃/3), variance 3
        z = randn(rng, n); c = vec(sum(randn(rng, n, 3) .^ 2, dims = 2))
        return z ./ sqrt.(c ./ 3) ./ sqrt(3)
    else                                           # Pareto(x_m = 1, α = 3): mean 3/2, variance 3/4
        x = rand(rng, n) .^ (-1 / 3)
        return (x .- 1.5) ./ sqrt(0.75)
    end
end

function design(; n, k, law, Wkind = :knn, R = 2000, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = Wkind == :knn ? knnW(P, 6) : groupW(n, 2, rng)
    B = hatbasis(u, v, k)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; K = size(Q, 2)
    M = I - Q * Q'; Wd = Matrix(W)
    δ = diag(Wd' * M) ./ diag(M)
    Pm = Wd' .- Diagonal(δ); C = Pm * M; Cs = (C + C') / 2
    f = Q * (Q' * (sin.(2π .* u) .* cos.(π .* v)))
    f .*= sqrt(n) / norm(f)
    b = M * (Pm' * f)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    V = 2 * dot(σ2, (Cs .^ 2) * σ2) + sum(b .^ 2 .* σ2)
    opF = opnorm(Cs) / norm(Cs)
    z = zeros(R)
    for r in 1:R
        ε = sqrt.(σ2) .* draw(rng, law, n)
        z[r] = (dot(b, ε) + dot(ε, C * ε)) / sqrt(V)
    end
    r3(x) = round(x, digits = 3)
    println(rpad("$(Wkind) $(law) n=$n K=$K", 28),
        " ‖Cs‖op/‖Cs‖F $(r3(opF)) | maxb²/V $(r3(maximum(b .^ 2 .* σ2) / V)) | z: mean $(r3(mean(z))) sd $(r3(std(z))) skew $(r3(skew(z))) exk $(r3(exkurt(z))) size $(r3(mean(abs.(z) .> 1.96))) KS $(r3(ks(z)))")
end

t0 = time()
design(; n = 400, k = 4, law = :t3, R = 50)        # warm-up
println("warm-up done. Monte Carlo SEs at R = 2000: mean 0.022, sd 0.016 (normal case), size 0.005; KS 95% critical value 0.030.")
println("Sample skewness and kurtosis are reported, but with an infinite fourth moment their own sampling error is large.\n")
for law in (:t3, :pareto)
    println("Errors $(law); local (hat) basis, K ≈ 2√n")
    for n in (400, 900, 1600, 2500, 4900); design(; n, k = round(Int, sqrt(2sqrt(n))), law); end
    println()
end
println("Power check: t₃ errors, W of two complete groups, hat basis K ≈ 2√n (‖Cs‖op/‖Cs‖F does not vanish)")
for n in (400, 900, 1600); design(; n, k = round(Int, sqrt(2sqrt(n))), law = :t3, Wkind = :groups); end
println("\ntotal $(round(time() - t0, digits = 1)) s")
