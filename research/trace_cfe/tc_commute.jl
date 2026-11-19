# The two zero-diagonal corrections coincide when W and the basis commute (torus + Fourier),
# and differ when they do not (torus + Chebyshev). tr(Cs²) is the quadratic variance term.
using LinearAlgebra
m = 20; n = m^2
idx(i, j) = mod(i - 1, m) * m + mod(j - 1, m) + 1
W = zeros(n, n)
for i in 1:m, j in 1:m, (a, b) in ((1,0),(-1,0),(0,1),(0,-1)); W[idx(i,j), idx(i+a,j+b)] = 0.25; end
u = [(i - 1) / m for i in 1:m for j in 1:m]; v = [(j - 1) / m for i in 1:m for j in 1:m]
four(p, q) = (cos.(2π .* (p .* u .+ q .* v)), sin.(2π .* (p .* u .+ q .* v)))
function compare(B, label)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; M = I - Q * Q'
    d = (M .* M) \ (-diag(M * W)); ours = M * W + M * Diagonal(d) * M
    δ = diag(W * M) ./ diag(M); jl = (W .- Diagonal(δ)) * M
    Cs(C) = (C + C') / 2
    println(rpad(label, 34), "K=$(size(Q,2))  ‖MW−WM‖=$(round(norm(M*W - W*M), sigdigits=3))  ‖ours−JL‖=$(round(norm(ours - jl), sigdigits=3))  tr(Cs²) ours/JL=$(round(tr(Cs(ours)^2)/tr(Cs(jl)^2), digits=3))")
end
F = hcat(ones(n), [hcat(four(p, q)...) for p in 0:4 for q in -4:4 if (p > 0 || q > 0)]...)
compare(F, "torus + Fourier basis (commutes)")
cheb(x, k) = cos(k * acos(clamp(2x - 1, -1, 1)))
C = hcat([cheb.(u, i) .* cheb.(v, j) for i in 0:8 for j in 0:8-i]...)
compare(C, "torus + Chebyshev basis")
