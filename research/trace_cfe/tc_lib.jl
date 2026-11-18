# Shared code: the plain CFE and the trace-corrected CFE (TC-CFE) with a projection first stage.
using LinearAlgebra, Random, Statistics, SparseArrays, NearestNeighbors

function knnW(P, k)
    n = size(P, 1); tree = KDTree(permutedims(P))
    idx, _ = knn(tree, permutedims(P), k + 1, true)
    I = Int[]; J = Int[]
    for i in 1:n, j in idx[i][2:end]; push!(I, i); push!(J, j); end
    sparse(I, J, fill(1.0 / k, length(I)), n, n)
end
cheb(x, m) = cos(m * acos(clamp(2x - 1, -1, 1)))
surfbasis(u, v, d) = hcat([cheb.(u, i) .* cheb.(v, j) for i in 0:d for j in 0:d-i]...)
unibasis(x, d) = (z = (x .- minimum(x)) ./ (maximum(x) - minimum(x)); hcat([cheb.(z, m) for m in 1:d]...))

# root of c - bρ + aρ² = 0 for the moment (e0 - ρe1)'(f1 - ρf2); keep the admissible root nearest the OLS ratio
function cfe_root(e0, f1, e1, f2)
    a = dot(e1, f2); b = dot(e0, f2) + dot(e1, f1); c = dot(e0, f1)
    D = b^2 - 4a * c; ols = dot(e1, e0) / dot(e1, e1)
    D < 0 && return (clamp(b / (2a), -0.99, 0.99), a, b)
    r = filter(x -> -1 < x < 1, [(b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a)])
    isempty(r) && return (clamp(ols, -0.99, 0.99), a, b)
    (r[argmin(abs.(r .- ols))], a, b)
end

"""Fixed-design objects: Q (orthonormal basis of span B), the dense M, A = MW + MDM with diag(A) = 0,
its symmetric part squared elementwise, and the corrections τ (pooled) and d (exact diagonal)."""
function tc_design(B, W)
    n = size(B, 1)
    F = qr(B); Q = Matrix(F.Q)[:, 1:rank(B)]; K = size(Q, 2)
    M = I - Q * Q'
    Wd = Matrix(W)
    MW = M * Wd
    τ = tr(Q' * (Wd * Q)) / (n - K)
    d = (M .* M) \ (-diag(MW))                    # makes diag(MW + M D M) = 0 exactly
    A = MW + M * Diagonal(d) * M
    As = (A + A') / 2
    G = M * (Wd * Q)                              # Var(â) = G (Q'ΣQ) G'
    (; Q, M, K, τ, d, A, As, As2 = As .^ 2, Mii = diag(M), Wd, G)
end

Mproj(D, z) = z .- D.Q * (D.Q' * z)

"""Plain CFE with Smirnov's closed-form variance σ̂²/(b − 2ρ̂a), the pooled-τ TC-CFE, and the exact-diagonal
TC-CFE with its heteroskedasticity-robust variance."""
function fit_all(D, W, y)
    n = length(y)
    Wy = W * y; W2y = W * Wy
    e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
    ρp, ap, bp = cfe_root(e0, e1, e1, e2)
    up = e0 .- ρp .* e1
    se_p = sqrt((dot(up, up) / n) / (bp - 2ρp * ap))
    ρτ, _, _ = cfe_root(e0, e1 .+ D.τ .* e0, e1, e2 .+ D.τ .* e1)
    ρd, ad, bd = cfe_root(e0, e1 .+ D.d .* e0, e1, e2 .+ D.d .* e1)
    vg = tc_varg(D, W, y, ρd)
    se_d = sqrt(vg.robust) / abs(bd - 2ρd * ad)
    (; ρp, se_p, ρτ, ρd, se_d)
end

"""Variance of the corrected moment at ρ: robust (σ̂ᵢ² = ûᵢ²/Mᵢᵢ) and homoskedastic (σ̄²)."""
function tc_varg(D, W, y, ρ)
    y0 = y .- ρ .* (W * y)
    u = Mproj(D, y0)
    s2 = u .^ 2 ./ D.Mii
    fhat = y0 .- u                                 # P_B y0
    a = Mproj(D, W * fhat)                         # M W f̂
    GC = D.G * (D.Q' * (s2 .* D.Q))
    noise = vec(sum(GC .* D.G, dims = 2))          # diag of the estimated Var(â)
    robust = 2 * dot(s2, D.As2 * s2) + sum((a .^ 2 .- noise) .* s2)
    sb = mean(s2)
    homo = 2 * sb^2 * sum(D.As2) + sb * dot(a, a)
    (; robust, homo, g = dot(u, Mproj(D, W * y0) .+ D.d .* u))
end
