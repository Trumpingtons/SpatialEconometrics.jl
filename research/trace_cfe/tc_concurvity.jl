# Concurvity: x2 is itself a smooth function of location, and the fitted model has free smooths of both
# the coordinates s(u, v) and x2. f lies only near the span of the basis (approximation error present).
include(joinpath(@__DIR__, "tc_lib.jl"))

function concurvity(; n = 900, ω = 0.1, ρ0 = 0.4, het = false, R = 500, seed = 21, dsurf = 12, dx = 8)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    g = sin.(π .* u) .+ cos.(1.5π .* v)                         # smooth spatial field
    x2 = (1 - ω) .* (g .- mean(g)) ./ std(g) .+ ω .* randn(rng, n)  # ω = 1: no concurvity
    x1 = randn(rng, n)
    s = 0.8 .* sin.(2π .* u) .* cos.(π .* v)
    f = s .+ 1.2 .* sin.(1.5 .* x2) .+ x1
    σ = het ? exp.(1.5 .* (u .+ v .- 1)) : ones(n)
    Bfree = hcat(surfbasis(u, v, dsurf), unibasis(x2, dx), x1)
    Bor = hcat(ones(n), s, sin.(1.5 .* x2), x1)                  # oracle: the true components, linear
    D = tc_design(Bfree, W); Do = tc_design(Bor, W)
    Fac = lu(sparse(I - ρ0 * W))
    out = zeros(R, 6)
    for r in 1:R
        y = Fac \ (f .+ σ .* randn(rng, n))
        fo = fit_all(Do, W, y); ff = fit_all(D, W, y)
        out[r, :] = [fo.ρp, ff.ρp, ff.ρτ, ff.ρd, ff.se_d, abs(ff.ρd - ρ0) <= 1.96 * ff.se_d]
    end
    b(j) = round(mean(out[:, j]) - ρ0, digits = 4); sd(j) = round(std(out[:, j]), digits = 4)
    println("ω=$ω ρ0=$ρ0 het=$het K=$(D.K) | oracle $(b(1)) ($(sd(1))) | free plain $(b(2)) ($(sd(2))) | pooled τ $(b(3)) ($(sd(3))) | TC exact $(b(4)) ($(sd(4))), se $(round(mean(out[:, 5]), digits = 4)), cov $(round(mean(out[:, 6]), digits = 3))")
end

concurvity(; R = 5)
for het in (false, true), ω in (1.0, 0.3, 0.1, 0.0), ρ0 in (0.0, 0.4)
    concurvity(; ω, ρ0, het)
end
