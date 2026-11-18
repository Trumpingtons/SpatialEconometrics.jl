# Null-distribution check for the exact-diagonal TC-CFE. Fixed design, errors redrawn R times.
include(joinpath(@__DIR__, "tc_lib.jl"))

function null_check(; n = 900, dsurf = 12, ρ0 = 0.4, R = 2000, law = :normal, het = true, seed = 11)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hcat(surfbasis(u, v, dsurf), x)
    D = tc_design(B, W)
    s = sin.(2π .* u) .* cos.(π .* v)
    f = D.Q * (D.Q' * s) .+ x                      # f exactly in span(B), so the test is exact
    σ = het ? exp.(1.5 .* (u .+ v .- 1)) : ones(n)
    a0 = Mproj(D, W * f)                           # M W f
    Vexact = 2 * dot(σ .^ 2, D.As2 * σ .^ 2) + sum(a0 .^ 2 .* σ .^ 2)
    Fac = lu(sparse(I - ρ0 * W))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    g = zeros(R); Vr = zeros(R); Vh = zeros(R); ρd = zeros(R); sed = zeros(R); ρp = zeros(R); sep = zeros(R)
    for r in 1:R
        y = Fac \ (f .+ σ .* draw())
        vg = tc_varg(D, W, y, ρ0)
        g[r] = vg.g; Vr[r] = vg.robust; Vh[r] = vg.homo
        fa = fit_all(D, W, y)
        ρd[r] = fa.ρd; sed[r] = fa.se_d; ρp[r] = fa.ρp; sep[r] = fa.se_p
    end
    z(x, μ) = (mean(x) - μ) / (std(x) / sqrt(length(x)))
    gc2 = (g .- mean(g)) .^ 2
    cov(ρ, se) = mean(abs.(ρ .- ρ0) .<= 1.96 .* se)
    println("law=$law het=$het ρ0=$ρ0 K=$(D.K) R=$R")
    println("  mean g:            z = $(round(z(g, 0), digits = 2))")
    println("  var g vs exact:    ratio $(round(var(g) / Vexact, digits = 3)), z = $(round(z(gc2, Vexact), digits = 2))")
    println("  E[robust V̂] / V:   ratio $(round(mean(Vr) / Vexact, digits = 3)), z = $(round(z(Vr, Vexact), digits = 2))")
    println("  E[homo V̂] / V:     ratio $(round(mean(Vh) / Vexact, digits = 3)), z = $(round(z(Vh, Vexact), digits = 2))   (power check)")
    println("  TC-CFE: bias $(round(mean(ρd) - ρ0, digits = 4)), sd $(round(std(ρd), digits = 4)), mean se $(round(mean(sed), digits = 4)), coverage $(round(cov(ρd, sed), digits = 3))")
    println("  plain CFE: bias $(round(mean(ρp) - ρ0, digits = 4)), sd $(round(std(ρp), digits = 4)), mean se $(round(mean(sep), digits = 4)), coverage $(round(cov(ρp, sep), digits = 3))")
end

t = @elapsed null_check(; R = 20)
println("warm-up done in $(round(t, digits = 1)) s\n")
for law in (:normal, :chisq), het in (true, false), ρ0 in (0.0, 0.4)
    null_check(; law, het, ρ0)
end
