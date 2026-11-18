# Quick check: does the trace correction remove the free-surface bias of the CFE?
using LinearAlgebra, Random, Statistics, SparseArrays, NearestNeighbors

function knnW(P, k)
    n = size(P, 1); tree = KDTree(permutedims(P))
    idx, _ = knn(tree, permutedims(P), k + 1, true)
    I = Int[]; J = Int[]
    for i in 1:n, j in idx[i][2:end]; push!(I, i); push!(J, j); end
    sparse(I, J, fill(1.0 / k, length(I)), n, n)
end

# tensor Chebyshev basis in (u, v) up to total degree d
cheb(x, m) = cos(m * acos(clamp(2x - 1, -1, 1)))
basis(u, v, d) = hcat([cheb.(u, i) .* cheb.(v, j) for i in 0:d for j in 0:d-i]...)

function cfe_root(e0, f1, e1, f2)
    a = dot(e1, f2); b = dot(e0, f2) + dot(e1, f1); c = dot(e0, f1)
    D = b^2 - 4a * c
    ols = dot(e1, e0) / dot(e1, e1)
    if D < 0; return clamp(b / (2a), -0.99, 0.99); end
    r = [(b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a)]
    r = filter(x -> -1 < x < 1, r)
    isempty(r) && return clamp(ols, -0.99, 0.99)
    r[argmin(abs.(r .- ols))]
end

function run(; n = 900, d = 12, ρ0 = 0.0, R = 400, seed = 1, surf = 1.0)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6)
    u, v = P[:, 1], P[:, 2]
    s = surf .* (sin.(2π .* u) .* cos.(π .* v) .+ 0.5 .* (u .- 0.5) .^ 2)
    B0 = basis(u, v, d)
    A = I - ρ0 * W
    out = zeros(R, 2); K = 0
    for r in 1:R
        x = randn(rng, n)
        B = hcat(B0, x); Q = Matrix(qr(B).Q)[:, 1:rank(B)]; K = size(Q, 2)
        y = A \ (s .+ x .+ randn(rng, n))
        Wy = W * y; W2y = W * Wy
        M(z) = z .- Q * (Q' * z)
        e0, e1, e2 = M(y), M(Wy), M(W2y)
        out[r, 1] = cfe_root(e0, e1, e1, e2)
        τ = tr(Q' * (W * Q)) / (n - K)            # tr(P_B W)/(n-K)
        out[r, 2] = cfe_root(e0, e1 .+ τ .* e0, e1, e2 .+ τ .* e1)
    end
    (K = K, plain_bias = mean(out[:, 1]) - ρ0, plain_sd = std(out[:, 1]),
     corr_bias = mean(out[:, 2]) - ρ0, corr_sd = std(out[:, 2]))
end

for d in (6, 12, 18), ρ0 in (0.0, 0.4)
    r = run(; d, ρ0)
    println("d=$d K=$(r.K) ρ0=$ρ0  plain bias $(round(r.plain_bias, digits=4)) sd $(round(r.plain_sd, digits=4))  |  corrected bias $(round(r.corr_bias, digits=4)) sd $(round(r.corr_sd, digits=4))")
end
