# The Corollary's root selection, measured. g(ρ) = c − bρ + aρ² has two roots. The Corollary
# (THEORY.md) shows that with probability → 1 the discriminant is positive, the consistent root is the
# one where g′(ρ̂) = −(b − 2ρ̂a) has the sign of −J, J = −E g′(ρ₀) = 2[f′G′Cs f + tr(G′CsΣ)], and the other
# root stays a distance of order J/a away. The code (tc_lib.jl's cfe_root) instead keeps the root in
# (−1, 1) nearest the OLS ratio e₁′e₀/e₁′e₁, which is not consistent, so the proof does not cover it.
# For tc_coverage_tails.jl's design (hat basis, K ≈ 2√n, smooth variances, ρ₀ = 0.4), per cell:
#   J/n (exact), the share of replications with a negative discriminant, the share where cfe_root's
#   choice differs from the sign rule's, and the distance from ρ₀ of the other root (median and 1% quantile).
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
function draw(rng, law, n)
    law == :normal && return randn(rng, n)
    z = randn(rng, n); c = vec(sum(randn(rng, n, 3) .^ 2, dims = 2))   # :t3, unit variance
    z ./ sqrt.(c ./ 3) ./ sqrt(3)
end

function cell(; n, law, ρ0 = 0.4, R = 1000, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = knnW(P, 6); Wd = Matrix(W)
    k = round(Int, sqrt(2sqrt(n)))
    Q = Matrix(qr(hatbasis(u, v, k)).Q); Q = Q[:, 1:rank(Q)]
    M = I - Q * Q'; Mii = diag(M)
    δ = diag(Wd' * M) ./ Mii
    Ct = Wd .- Diagonal(δ); C = Ct' * M; Cs = (C + C') / 2
    f = Q * (Q' * (sin.(2π .* u) .* cos.(π .* v))); f .*= sqrt(n) / norm(f)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    G = Wd / (I - ρ0 * Wd)
    J = 2 * (dot(G * f, Cs * f) + tr(G' * Cs * Diagonal(σ2)))
    Fac = lu(sparse(I - ρ0 * W))
    negD = 0; differ = 0; other = Float64[]
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* draw(rng, law, n)); Wy = W * y
        e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
        ρc, a, b = cfe_root(e0, Ct * y, e1, Ct * Wy)
        c = dot(e0, Ct * y)
        D = b^2 - 4a * c
        if D < 0; negD += 1; continue; end
        rts = [(b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a)]
        gp = [-(b - 2r_ * a) for r_ in rts]          # g′ at each root: ∓√D
        pick = findfirst(x -> sign(x) == -sign(J), gp)
        ρs = rts[pick]
        differ += abs(ρc - ρs) > 1e-10
        push!(other, abs(rts[3 - pick] - ρ0))
    end
    @printf("%-6s n %5d | J/n %+.3f | D < 0: %.3f | cfe_root ≠ sign rule: %.3f | other root |ρ − ρ₀|: median %.2f, 1%% %.2f\n",
            law, n, J / n, negD / R, differ / R, median(other), quantile(other, 0.01))
end

cell(; n = 400, law = :t3, R = 10)                 # warm-up
println("R = 1000 per cell\n")
for law in (:normal, :t3), n in (400, 900, 1600, 2500)
    cell(; n, law)
end
