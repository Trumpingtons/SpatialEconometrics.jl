# Theory step 1, numerically: where does the two-sided correction's efficiency gain come from?
#
# Model: ρ₀ = 0, y = f + ε, ε ~ N(0, σ²I) with σ = 1, f exactly in span(B). Both corrections give a moment
# g(ρ) = y0′C y0 with y0 = y − ρWy and diag(C) = 0:
#   ours: C = MW + MDM, (M∘M)d = −diag(MW)          Jin & Lee: C = (W − Δ)M, Δ = Diag(WM)Diag(M)⁻¹
# With Cs = (C + C′)/2, the delta-method variance of ρ̂ at ρ₀ = 0 is
#   Var ρ̂ = [Vq + Vl] / J²,  Vq = 2σ⁴ tr(Cs²),  Vl = 4σ² ‖Cs f‖²,
#   J = −E g′(ρ₀) = 2 f′W′Cs f + 2σ² tr(W′Cs) = Js + Jn.
# Part A checks this formula against simulation. Part B follows a path from the commuting case
# (torus rook W, Fourier basis: the two coincide) towards a basis unrelated to W, B(θ) = cos θ F + sin θ G,
# and reports each piece as a ratio ours / Jin & Lee. Part C evaluates it in tc_concurvity.jl's design.
include(joinpath(@__DIR__, "tc_lib.jl"))

function torusW(m)
    n = m^2; idx(i, j) = mod(i - 1, m) * m + mod(j - 1, m) + 1
    I_ = Int[]; J_ = Int[]
    for i in 1:m, j in 1:m, (a, b) in ((1, 0), (-1, 0), (0, 1), (0, -1)); push!(I_, idx(i, j)); push!(J_, idx(i + a, j + b)); end
    sparse(I_, J_, fill(0.25, length(I_)), n, n)
end
function fourier(m, r)                             # all modes with |p|, |q| ≤ r, as real columns
    u = [(i - 1) / m for i in 1:m for j in 1:m]; v = [(j - 1) / m for i in 1:m for j in 1:m]
    cols = [ones(m^2)]
    for p in 0:r, q in -r:r
        (p > 0 || q > 0) || continue
        push!(cols, cos.(2π .* (p .* u .+ q .* v))); push!(cols, sin.(2π .* (p .* u .+ q .* v)))
    end
    hcat(cols...)
end
orth(B) = (F = qr(B); Matrix(F.Q)[:, 1:rank(B)])

function corrections(Q, W)
    M = I - Q * Q'; Wd = Matrix(W)
    d = (M .* M) \ (-diag(M * Wd)); ours = M * Wd + M * Diagonal(d) * M
    δ = diag(Wd * M) ./ diag(M); jl = (Wd .- Diagonal(δ)) * M
    (; ours, jl, Wd)
end

function pieces(C, Wd, f; σ = 1.0)
    Cs = (C + C') / 2
    Vq = 2σ^4 * sum(abs2, Cs); Vl = 4σ^2 * sum(abs2, Cs * f)
    Js = 2 * dot(Wd * f, Cs * f); Jn = 2σ^2 * sum(Wd .* Cs)       # tr(W′Cs) = Σ Wᵢⱼ Csᵢⱼ
    (; Vq, Vl, Js, Jn, V = (Vq + Vl) / (Js + Jn)^2)
end

function root(C, y, Wy)                            # g(ρ) = c − bρ + aρ², admissible root nearest the OLS ratio
    a = dot(Wy, C * Wy); b = dot(Wy, (C + C') * y); c = dot(y, C * y)
    D = b^2 - 4a * c; ols = dot(Wy, y) / dot(Wy, Wy)
    D < 0 && return clamp(b / (2a), -0.99, 0.99)
    r = filter(x -> -1 < x < 1, [(b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a)])
    isempty(r) ? clamp(ols, -0.99, 0.99) : r[argmin(abs.(r .- ols))]
end

function simulate(C1, C2, W, f; R = 4000, seed = 5)
    rng = MersenneTwister(seed); n = length(f); out = zeros(R, 2)
    for r in 1:R
        y = f .+ randn(rng, n); Wy = W * y
        out[r, :] = [root(C1, y, Wy), root(C2, y, Wy)]
    end
    out
end

r3(x) = round(x, digits = 3)

# shared setup: torus m = 30 (n = 900); F = Fourier modes |p|,|q| ≤ r; G = random columns orthogonal to F
m = 30; n = m^2; W = torusW(m)
rng = MersenneTwister(1)
function path_basis(r, θ; rng = MersenneTwister(1))
    F = orth(fourier(m, r)); K = size(F, 2)
    G = randn(rng, n, K); G = orth(G .- F * (F' * G)) # orthonormal, orthogonal to span F
    orth(cos(θ) .* F .+ sin(θ) .* G), K
end
coefvec(K; rng = MersenneTwister(2)) = (c = randn(rng, K); c ./ norm(c) .* sqrt(n))   # ‖f‖²/n = 1

println("A. Formula against simulation (R = 4000; the sd of a simulated sd is about 1.1% of it)")
for (r, θ) in ((4, π / 4), (4, π / 2), (6, π / 4))
    Q, K = path_basis(r, θ); f = Q * coefvec(size(Q, 2))
    c = corrections(Q, W); po = pieces(c.ours, c.Wd, f); pj = pieces(c.jl, c.Wd, f)
    s = simulate(c.ours, c.jl, W, f)
    println("  K=$(size(Q,2)) θ=$(r3(θ)): ours sd sim $(r3(std(s[:,1]))) vs formula $(r3(sqrt(po.V))) | JL sd sim $(r3(std(s[:,2]))) vs formula $(r3(sqrt(pj.V))) | bias ours $(r3(mean(s[:,1]))) JL $(r3(mean(s[:,2])))")
end

println("\nB. Misalignment path: ratios ours / Jin & Lee (1 = no difference)")
println("  columns: Var ρ̂ | Vq (quadratic) | Vl (linear) | J (derivative) | Js (signal part) | Jn (noise part) | shares of ours: Vl/V_g, Js/J")
for r in (2, 4, 6), θ in (0.0, π / 12, π / 6, π / 4, π / 3, π / 2)
    Q, K = path_basis(r, θ); f = Q * coefvec(size(Q, 2))
    c = corrections(Q, W); po = pieces(c.ours, c.Wd, f); pj = pieces(c.jl, c.Wd, f)
    J(p) = p.Js + p.Jn
    println("  K=$(lpad(size(Q,2),3)) K/n=$(r3(size(Q,2)/n)) θ=$(rpad(r3(θ),5)) | $(r3(po.V/pj.V)) | $(r3(po.Vq/pj.Vq)) | $(r3(po.Vl/pj.Vl)) | $(r3(J(po)/J(pj))) | $(r3(po.Js/pj.Js)) | $(r3(po.Jn/pj.Jn)) | $(r3(po.Vl/(po.Vq+po.Vl))), $(r3(po.Js/J(po)))")
end

println("\nC. tc_concurvity.jl's design at ρ₀ = 0, constant variance (f projected on span B); simulated sd ratios were 0.84 (ω = 1) and 0.76 (ω = 0.1)")
for ω in (1.0, 0.1)
    local rng = MersenneTwister(21)
    P = rand(rng, 900, 2); Wk = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    g = sin.(π .* u) .+ cos.(1.5π .* v)
    x2 = (1 - ω) .* (g .- mean(g)) ./ std(g) .+ ω .* randn(rng, 900); x1 = randn(rng, 900)
    f0 = 0.8 .* sin.(2π .* u) .* cos.(π .* v) .+ 1.2 .* sin.(1.5 .* x2) .+ x1
    Q = orth(hcat(surfbasis(u, v, 12), unibasis(x2, 8), x1)); f = Q * (Q' * f0)
    c = corrections(Q, Wk); po = pieces(c.ours, c.Wd, f); pj = pieces(c.jl, c.Wd, f)
    J(p) = p.Js + p.Jn
    println("  ω=$ω K=$(size(Q,2)): sd ratio $(r3(sqrt(po.V/pj.V))) | Var $(r3(po.V/pj.V)) | Vq $(r3(po.Vq/pj.Vq)) | Vl $(r3(po.Vl/pj.Vl)) | J $(r3(J(po)/J(pj))) | Js $(r3(po.Js/pj.Js)) | Jn $(r3(po.Jn/pj.Jn))")
end

# D. The hypothesis from C: the gain is in the linear (signal) part, and comes from the ASYMMETRY of W, not from
# misalignment. Ours has linear instrument MWf (so Js = ‖MWf‖², the best linear instrument by Cauchy–Schwarz);
# Jin & Lee with A = W has M(W′ − Δ)f. Prediction: Jin & Lee with A = W′, or a symmetric W, removes most of the gain.
println("
D. Asymmetry test, same design as C (ω = 0.1): Var ratio ours / alternative, and the instrument alignment")
let ω = 0.1, rng = MersenneTwister(21)
    P = rand(rng, 900, 2); Wk = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    g = sin.(π .* u) .+ cos.(1.5π .* v)
    x2 = (1 - ω) .* (g .- mean(g)) ./ std(g) .+ ω .* randn(rng, 900); x1 = randn(rng, 900)
    f0 = 0.8 .* sin.(2π .* u) .* cos.(π .* v) .+ 1.2 .* sin.(1.5 .* x2) .+ x1
    Q = orth(hcat(surfbasis(u, v, 12), unibasis(x2, 8), x1)); f = Q * (Q' * f0)
    jlA(Wd, A, M) = (δ = diag(A * M) ./ diag(M); (A .- Diagonal(δ)) * M)
    for (label, Wm) in (("kNN W (asymmetric)", Wk), ("symmetrised (W + W′)/2", (Wk + Wk') ./ 2))
        c = corrections(Q, Wm); M = I - Q * Q'
        po = pieces(c.ours, c.Wd, f)
        for (alt, C) in (("JL, A = W", c.jl), ("JL, A = W′", jlA(c.Wd, Matrix(c.Wd'), M)))
            pa = pieces(C, c.Wd, f)
            println("  $(rpad(label, 24)) vs $(rpad(alt, 11)): Var $(r3(po.V/pa.V)) | Vq $(r3(po.Vq/pa.Vq)) | Vl $(r3(po.Vl/pa.Vl)) | Js $(r3(po.Js/pa.Js))")
        end
        a = M * (c.Wd * f); b = M * (c.Wd' * f)
        println("    cos(MWf, MW′f) = $(r3(dot(a, b) / (norm(a) * norm(b))))")
    end
end
