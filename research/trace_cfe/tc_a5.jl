# Proposition 5 (THEORY.md): a primitive condition for A5, liminf ‖Cs‖_F²/n > 0.
#
# For the exact-diagonal moment A = MW + MDM, M Cs M = M(Ws + D)M with Ws = (W + W′)/2, and ‖MXM‖_F ≤ ‖X‖_F,
# MXM, P_B X M, M X P_B and P_B X P_B are orthogonal in the Frobenius inner product, and ‖P_B X‖_F² ≤ K‖X‖_op², so
#     ‖Cs‖_F² ≥ ‖MXM‖_F² ≥ ‖Ws‖_F² − 2K (‖W‖_op + d̄)²,   d̄ = maxᵢ |dᵢ|,
# using ‖Ws + D‖_F² = ‖Ws‖_F² + ‖D‖_F² (diag Ws = 0). For a non-negative row-standardised W with at most m
# neighbours per row, ‖Ws‖_F² ≥ ‖W‖_F²/2 ≥ n/(2m). And if maxᵢ hᵢᵢ = h̄ < 1/2, M∘M is diagonally dominant and
#     d̄ ≤ maxᵢ‖Weᵢ‖ √h̄ / ((1 − h̄)(1 − 2h̄)).
# This script computes, for the designs of tc_vhat.jl (Chebyshev surface) and tc_vhat_lo_hat.jl (hat basis),
# the actual ‖Cs‖_F²/n against the bound in its three forms: with the measured ‖Ws‖_F and d̄, with
# n/(2m) for ‖Ws‖_F², and with the h̄ < 1/2 bound for d̄ where it applies. No simulation: all are exact.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function row(name, B, W, n)
    D = tc_design(B, W)
    Wd = D.Wd
    Ws = (Wd + Wd') / 2
    cs = sum(D.As2) / n
    opW = opnorm(Wd); d̄ = maximum(abs.(D.d)); h̄ = maximum(1 .- D.Mii)
    m = maximum(vec(sum(Wd .> 0, dims = 2)))
    colmax = maximum(sqrt.(vec(sum(Wd .^ 2, dims = 1))))
    lb(wsF, dd) = max(wsF^2 / n - 2 * (D.K / n) * (opW + dd)^2, 0.0)
    b1 = lb(norm(Ws), d̄)
    b2 = lb(sqrt(n / (2m)), d̄)
    dbd = h̄ < 0.5 ? colmax * sqrt(h̄) / ((1 - h̄) * (1 - 2h̄)) : NaN
    b3 = h̄ < 0.5 ? lb(sqrt(n / (2m)), dbd) : NaN
    kstar = norm(Ws)^2 / n / (2 * (opW + d̄)^2)               # K/n below which bound 1 is positive
    @printf("%-10s K %3d K/n %.3f maxh %.3f | ‖Cs‖F²/n %.4f | ‖Ws‖F²/n %.4f (≥ 1/(2m) = %.4f) ‖W‖op %.3f d̄ %.3f (bound %s) | lower bound: measured %.4f, with 1/(2m) %.4f, with d̄ bound %s | bound positive for K/n < %.4f\n",
            name, D.K, D.K / n, h̄, cs, norm(Ws)^2 / n, 1 / (2m), opW, d̄, isnan(dbd) ? "n/a, h̄ ≥ 1/2" : @sprintf("%.3f", dbd),
            b1, b2, isnan(b3) ? "n/a" : @sprintf("%.4f", b3), kstar)
end

n = 900
rng = MersenneTwister(5)
P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
x = randn(rng, n)
for dsurf in (2, 3, 4, 6, 8, 12, 17, 21, 25)
    row("chebyshev", hcat(surfbasis(u, v, dsurf), x), W, n)
end
println()
for kh in (2, 3, 4, 5, 7, 10, 13, 15)
    row("hat", hcat(hatbasis(u, v, kh), x), W, n)
end
