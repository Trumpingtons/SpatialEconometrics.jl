# Jin & Lee (2012, §3.2) against the exact-diagonal TC-CFE, in the concurvity design of tc_concurvity.jl.
#
# Jin & Lee's heteroskedasticity-robust root estimator solves y0'P M y0 = 0, with y0 = y − ρWy and
# P = A − Δ M, where Δ = Diag(AM) Diag(M)⁻¹. Then C = P M = (A − Δ) M has a zero diagonal.
#   JL-W:  A = W′ (their initial estimator, §3.2 p. 451)
#   JL-G:  A = G(ρ̂)′, G = W (I − ρ̂W)⁻¹, with ρ̂ = JL-W (their REsd, two-step)
# Correction, 2026-11-19: the first version of this script used A = W and A = G(ρ̂). The primes were lost
# in the pdftotext extraction and are visible on the page images. With A = W the linear instrument is
# M(W′ − Δ)f in place of MWf, which with an asymmetric kNN W cost 25–40% of variance (tc_misalign.jl, part D).
# Every estimator gets the same robust variance, written for a general zero-diagonal C:
#   Var g = 2σ'(Cs∘Cs)σ + Σ (âᵢ² − noiseᵢ) σᵢ²,  â = (C + C') P_B y0,  noise = diag(H (Q'ΣQ) H'), H = (C + C')Q
# For C = MW + MDM this reduces to tc_varg (checked below on the first replication).
include(joinpath(@__DIR__, "tc_lib.jl"))

# moment (e0 − ρe1)'(v0 − ρv1) with e = M y-terms and v = (A − Δ)' y-terms; C = (A − Δ)M
function jl_fit(D, A, y, Wy)
    AM = A * D.M
    δ = diag(AM) ./ D.Mii
    Ct = A' .- Diagonal(δ)                         # (A − Δ)'
    e0, e1 = Mproj(D, y), Mproj(D, Wy)
    ρ, a, b = cfe_root(e0, Ct * y, e1, Ct * Wy)
    C = AM .- δ .* D.M                             # (A − Δ) M, zero diagonal
    (; ρ, a, b, C)
end

function robust_var(D, C, W, y, ρ)
    y0 = y .- ρ .* (W * y)
    u = Mproj(D, y0); s2 = u .^ 2 ./ D.Mii
    S = C + C'
    a = S * (y0 .- u)                              # (C + C') P_B y0
    H = S * D.Q
    noise = vec(sum((H * (D.Q' * (s2 .* D.Q))) .* H, dims = 2))
    2 * dot(s2, ((S ./ 2) .^ 2) * s2) + sum((a .^ 2 .- noise) .* s2)
end

se_of(V, ρ, a, b) = sqrt(max(V, 0)) / abs(b - 2ρ * a)

function compare(; n = 900, ω = 0.1, ρ0 = 0.4, het = false, R = 500, seed = 21, dsurf = 12, dx = 8, check = false)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    g = sin.(π .* u) .+ cos.(1.5π .* v)
    x2 = (1 - ω) .* (g .- mean(g)) ./ std(g) .+ ω .* randn(rng, n)
    x1 = randn(rng, n)
    s = 0.8 .* sin.(2π .* u) .* cos.(π .* v)
    f = s .+ 1.2 .* sin.(1.5 .* x2) .+ x1
    σ = het ? exp.(1.5 .* (u .+ v .- 1)) : ones(n)
    B = hcat(surfbasis(u, v, dsurf), unibasis(x2, dx), x1)
    D = tc_design(B, W)
    Wd = D.Wd
    Fac = lu(sparse(I - ρ0 * W))
    out = zeros(R, 9)                              # (ρ, se) for TC, JL-W, JL-G; then coverage ×3
    for r in 1:R
        y = Fac \ (f .+ σ .* randn(rng, n)); Wy = W * y
        tc = fit_all(D, W, y)
        jw = jl_fit(D, Matrix(Wd'), y, Wy)
        sew = se_of(robust_var(D, jw.C, W, y, jw.ρ), jw.ρ, jw.a, jw.b)
        Gh = Wd / (I - jw.ρ * Wd)
        jg = jl_fit(D, Matrix(Gh'), y, Wy)
        seg = se_of(robust_var(D, jg.C, W, y, jg.ρ), jg.ρ, jg.a, jg.b)
        if check && r == 1                         # the general variance reproduces tc_varg for C = A
            Vg = robust_var(D, D.A, W, y, tc.ρd); Vt = tc_varg(D, W, y, tc.ρd).robust
            println("  check: general variance / tc_varg = $(Vg / Vt);  max|diag C| JL-W = $(maximum(abs.(diag(jw.C)))), JL-G = $(maximum(abs.(diag(jg.C))))")
        end
        out[r, 1:6] = [tc.ρd, tc.se_d, jw.ρ, sew, jg.ρ, seg]
    end
    for k in 1:3
        out[:, 6 + k] = abs.(out[:, 2k - 1] .- ρ0) .<= 1.96 .* out[:, 2k]
    end
    f4(x) = round(x, digits = 4)
    cell(k, name) = "$name $(f4(mean(out[:, 2k-1]) - ρ0)) sd $(f4(std(out[:, 2k-1]))) se $(f4(mean(out[:, 2k]))) cov $(round(mean(out[:, 6+k]), digits = 3))"
    println("ω=$ω ρ0=$ρ0 het=$het K=$(D.K) | ", cell(1, "TC exact"), " | ", cell(2, "JL-W"), " | ", cell(3, "JL-G"))
end

t = @elapsed compare(; R = 5, check = true)
println("warm-up (5 replications) in $(round(t, digits = 1)) s\n")
for het in (false, true), ω in (1.0, 0.3, 0.1, 0.0), ρ0 in (0.0, 0.4)
    compare(; ω, ρ0, het)
end
println("\nLarger basis (dsurf = 18):")
for het in (false, true), ρ0 in (0.0, 0.4)
    compare(; ω = 0.1, ρ0, het, dsurf = 18)
end
