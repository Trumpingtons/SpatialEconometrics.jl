# Remark 3 / Proposition 4 (THEORY.md): a penalised smoother in place of the projection.
#
# M = I − S with S = B(B′B + λΩ)⁻¹B′ symmetric with eigenvalues in [0, 1] (a tensor P-spline: a 15 × 15 hat
# basis with a second-difference penalty, plus an unpenalised covariate x). M is not idempotent, so:
#   - the moment is g(ρ) = y₀′Ay₀ with A = MW + MDM, (M∘M)d = −diag(MW), so diag A = 0 exactly;
#   - Mf ≠ 0 even when f is smooth, so E g(ρ₀) = β = f′Af (Proposition 3 with r = Mf, the shrinkage bias);
#   - E ûᵢ² = σ²(M²)ᵢᵢ under homoskedasticity, so the HC2 plug-in becomes sᵢ = ûᵢ²/(M²)ᵢᵢ, and the noise term
#     is diag(HΣ̂H′) with H = (A + A′)S, the general form of the projection's G(Q′Σ̂Q)G′.
# Checks: (1) Theorem 1: z₀ = (g(ρ₀) − β)/√V has mean 0, sd 1 and a small KS distance from N(0, 1);
# (2) the predicted bias β/J and coverage P(|Z + c| ≤ 1.96), c = β/√V, against simulation, for the oracle V
# and the general V̂. Power checks: the projection formulas applied to a smoother (sᵢ = ûᵢ²/Mᵢᵢ), and the
# code's shortcut moment (My₀)′(MWy₀ + D My₀) = y₀′(M²W + MDM)y₀, whose diagonal is not zero when M² ≠ M.
# Design: n = 900, kNN W, f = a·sin(2πu)cos(πv) + x with a = 1 and 4, smooth variances, normal errors,
# ρ₀ = 0.4, R = 1000, λ from 10⁻³ to 10².
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function ks(z)
    z = sort(z); m = length(z); F = cdf.(Normal(), z)
    maximum(max.((1:m) ./ m .- F, F .- (0:m-1) ./ m))
end

function cell(; n = 900, kh = 15, λ = 1.0, amp = 1.0, R = 1000, ρ0 = 0.4, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]; Wd = Matrix(W)
    x = randn(rng, n)
    Bh = hatbasis(u, v, kh); B = hcat(Bh, x)
    D2 = diff(Matrix(1.0I, kh, kh), dims = 1) |> m -> diff(m, dims = 1)
    Ωh = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    Ω = zeros(kh^2 + 1, kh^2 + 1); Ω[1:kh^2, 1:kh^2] = Ωh
    S = B * ((B' * B + λ * Ω) \ B'); S = (S + S') / 2
    M = I - S
    MW = M * Wd
    d = (M .* M) \ (-diag(MW))
    A = MW + M * Diagonal(d) * M
    As = (A + A') / 2; As2 = As .^ 2; Ab = A + A'
    H = Ab * S
    M2d = diag(M * M); Mii = diag(M)
    Acode = M * MW + M * Diagonal(d) * M                  # y₀′(M²W + MDM)y₀, the idempotent shortcut
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    β = dot(f, A * f); bt = Ab * f
    V = 2 * dot(σ2, As2 * σ2) + sum(bt .^ 2 .* σ2)
    G = Wd / (I - ρ0 * Wd)
    J = 2 * (dot(G * f, As * f) + tr(G' * As * Diagonal(σ2)))
    c = β / sqrt(V); pcov = cdf(Normal(), 1.96 - c) - cdf(Normal(), -1.96 - c)
    βcode = dot(f, Acode * f) + tr(Acode * Diagonal(σ2))
    Fac = lu(sparse(I - ρ0 * W))
    z0 = zeros(R); ρs = zeros(R); ρc = zeros(R)
    cov = Dict(k => 0 for k in (:oracle, :general, :projection, :code)); neg = 0
    vh(y0, s) = (â = H * y0; noise = vec(sum((H .^ 2) .* s', dims = 2));
                 2 * dot(s, As2 * s) + sum((â .^ 2 .- noise) .* s))
    for r in 1:R
        ε = sqrt.(σ2) .* randn(rng, n)
        y = Fac \ (f .+ ε); Wy = W * y
        z0[r] = (dot(f .+ ε, A * (f .+ ε)) - β) / sqrt(V)
        ρ, a, b = cfe_root(y, A * y, Wy, A * Wy); ρs[r] = ρ
        gp = abs(b - 2ρ * a)
        y0 = y .- ρ .* Wy; û = M * y0
        cov[:oracle] += abs(ρ - ρ0) <= 1.96 * sqrt(V) / gp
        Vg = vh(y0, û .^ 2 ./ M2d); neg += Vg <= 0
        cov[:general] += Vg > 0 && abs(ρ - ρ0) <= 1.96 * sqrt(Vg) / gp
        Vp = vh(y0, û .^ 2 ./ Mii)
        cov[:projection] += Vp > 0 && abs(ρ - ρ0) <= 1.96 * sqrt(Vp) / gp
        ρk, ak, bk = cfe_root(y, Acode * y, Wy, Acode * Wy); ρc[r] = ρk
        cov[:code] += Vg > 0 && abs(ρk - ρ0) <= 1.96 * sqrt(Vg) / abs(bk - 2ρk * ak)
    end
    bias = mean(ρs) - ρ0; se = std(ρs) / sqrt(R)
    @printf("amp %.0f λ %7.3f df %6.1f maxS %.3f ‖As‖op %.3f max|d| %.3f | Thm 1: mean %+.3f sd %.3f KS %.3f | β %+8.3f c %+.3f | bias: pred %+.4f sim %+.4f (z %5.1f) sd %.4f | coverage: pred %.3f oracle %.3f general %.3f%s projection %.3f | code moment: E g %+.2f bias %+.4f coverage %.3f\n",
            amp, λ, tr(S), maximum(diag(S)), opnorm(As), maximum(abs.(d)), mean(z0), std(z0), ks(z0), β, c,
            β / J, bias, (bias - β / J) / se, std(ρs), pcov, cov[:oracle] / R, cov[:general] / R,
            neg > 0 ? "(neg $neg)" : "", cov[:projection] / R, βcode, mean(ρc) - ρ0, cov[:code] / R)
end

t = @elapsed cell(; R = 5)
println("warm-up (the line above, 5 replications) $(round(t, digits = 1)) s. R = 1000: MC SE 0.007 for a coverage near 0.95; KS critical value 0.043 at 5%.\n")
t0 = time()
for amp in (1.0, 4.0)
    for λ in (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
        cell(; λ, amp)
    end
    println()
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
