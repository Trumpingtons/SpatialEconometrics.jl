# Proposition 1 (THEORY.md): Jin & Lee's two-step REsd has the same limit as the infeasible estimator.
#
# REsd: ρ̃ is the first step (C₁ = (W′ − Δ)M); then ρ̂ is the root of y₀(ρ)′C(ρ̃)y₀(ρ) with
# C(r) = (G(r)′ − Δ(r))M, G(r) = W(I − rW)⁻¹, Δ(r) = Diag(G(r)′M)Diag(M)⁻¹. The infeasible ρ̂* uses C(ρ₀).
# The proposition says √n(ρ̂ − ρ̂*) → 0 in probability, because C(r)f = 0 and diag C(r) = 0 for every r
# make the plug-in's first-order effect have mean exactly zero. So sd(ρ̂ − ρ̂*)/sd(ρ̂*) should fall like
# n^{−1/2}. Power check: a fixed wrong plug-in, C(ρ₀ + 0.3). It is still a valid moment (any fixed r is),
# so it stays unbiased, but it is a different estimator, and its distance from ρ̂* must NOT shrink
# relative to sd(ρ̂*). Design: tc_coverage_tails.jl's (hat basis, K ≈ 2√n, smooth variances, ρ₀ = 0.4),
# normal errors, R = 500. Coverage uses the robust variance written for a general zero-diagonal C
# (tc_jinlee.jl) and the code's root rule.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

# C = (A − Δ)M with Δ = Diag(AM)Diag(M)⁻¹, and Ct = (A − Δ)′, for A = W′ or G(r)′
function moment_mats(A, M, Mii)
    AM = A * M
    δ = diag(AM) ./ Mii
    (; Ct = A' .- Diagonal(δ), C = AM .- δ .* M)
end

function fitC(mm, y, Wy, Q, Mii)
    e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
    ρ, a, b = cfe_root(e0, mm.Ct * y, e1, mm.Ct * Wy)
    y0 = y .- ρ .* Wy
    u = y0 .- Q * (Q' * y0); s2 = u .^ 2 ./ Mii
    S = mm.C + mm.C'; H = S * Q
    â = S * (y0 .- u)
    noise = vec(sum((H * (Q' * (s2 .* Q))) .* H, dims = 2))
    V = 2 * dot(s2, ((S ./ 2) .^ 2) * s2) + sum((â .^ 2 .- noise) .* s2)
    (ρ, sqrt(max(V, 0)) / abs(b - 2ρ * a))
end

Gof(Wd, r) = Wd / (I - r * Wd)

function cell(; n, ρ0 = 0.4, R = 500, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = knnW(P, 6); Wd = Matrix(W)
    k = round(Int, sqrt(2sqrt(n)))
    Q = Matrix(qr(hatbasis(u, v, k)).Q); Q = Q[:, 1:rank(Q)]
    M = I - Q * Q'; Mii = diag(M)
    f = Q * (Q' * (sin.(2π .* u) .* cos.(π .* v))); f .*= sqrt(n) / norm(f)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    m1 = moment_mats(Matrix(Wd'), M, Mii)               # first step
    mstar = moment_mats(Matrix(Gof(Wd, ρ0)'), M, Mii)    # infeasible C(ρ₀)
    mbad = moment_mats(Matrix(Gof(Wd, ρ0 + 0.3)'), M, Mii)
    orth = maximum(abs.(mstar.C * f)), maximum(abs.(diag(mstar.C)))
    Fac = lu(sparse(I - ρ0 * W))
    out = zeros(R, 6)                                  # ρ̃, ρ̂, se(ρ̂), ρ̂*, se(ρ̂*), ρ̂_bad
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        ρt, _ = fitC(m1, y, Wy, Q, Mii)
        ρh, seh = fitC(moment_mats(Matrix(Gof(Wd, ρt)'), M, Mii), y, Wy, Q, Mii)
        ρs, ses = fitC(mstar, y, Wy, Q, Mii)
        ρb, _ = fitC(mbad, y, Wy, Q, Mii)
        out[r, :] = [ρt, ρh, seh, ρs, ses, ρb]
    end
    sds = std(out[:, 4])
    covh = mean(abs.(out[:, 2] .- ρ0) .<= 1.96 .* out[:, 3])
    covs = mean(abs.(out[:, 4] .- ρ0) .<= 1.96 .* out[:, 5])
    @printf("n %5d K %4d | max|C*f| %.1e max|diag C*| %.1e | sd: ρ̃ %.4f ρ̂ %.4f ρ̂* %.4f | sd(ρ̂ − ρ̂*)/sd(ρ̂*) %.4f  ×√n %.3f | bad plug-in: sd(ρ̂_bad − ρ̂*)/sd(ρ̂*) %.4f bias %+.4f | coverage ρ̂ %.3f ρ̂* %.3f\n",
            n, size(Q, 2), orth..., std(out[:, 1]), std(out[:, 2]), sds,
            std(out[:, 2] .- out[:, 4]) / sds, sqrt(n) * std(out[:, 2] .- out[:, 4]) / sds,
            std(out[:, 6] .- out[:, 4]) / sds, mean(out[:, 6]) - ρ0, covh, covs)
end

t = @elapsed cell(; n = 400, R = 10)
println("warm-up: 10 replications at n = 400 in $(round(t, digits = 1)) s")
t = @elapsed cell(; n = 2500, R = 5)
println("5 replications at n = 2500 in $(round(t, digits = 1)) s. R = 500 per cell; MC SE of a coverage near 0.95: 0.010.\n")
t0 = time()
for n in (400, 900, 1600, 2500); cell(; n); end
println("\ntotal $(round(time() - t0, digits = 1)) s")
