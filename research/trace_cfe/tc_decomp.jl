include(joinpath(@__DIR__, "tc_lib.jl"))
function decomp(; n = 900, dsurf = 12, R = 2000, het = true, seed = 11)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n); B = hcat(surfbasis(u, v, dsurf), x); D = tc_design(B, W)
    f = D.Q * (D.Q' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    σ = het ? exp.(1.5 .* (u .+ v .- 1)) : ones(n); σ2 = σ .^ 2
    a0 = Mproj(D, W * f)
    Vq = 2 * dot(σ2, D.As2 * σ2); Va = sum(a0 .^ 2 .* σ2)
    # analytic expectation of the a-term plug-in noise: Var(â) = G (Q'ΣQ) G', G = M W Q
    G = D.M * (D.Wd * D.Q); Vahat = G * (D.Q' * (σ2 .* D.Q)) * G'
    Va_noise = sum(diag(Vahat) .* σ2)
    q = zeros(R); la = zeros(R)
    for r in 1:R
        y0 = f .+ σ .* randn(rng, n)
        uu = Mproj(D, y0); s2 = uu .^ 2 ./ D.Mii
        a = Mproj(D, W * (y0 .- uu))
        q[r] = 2 * dot(s2, D.As2 * s2); la[r] = sum(a .^ 2 .* s2)
    end
    println("het=$het  Vq share $(round(Vq/(Vq+Va), digits=3))")
    println("  quadratic: E[q̂]/Vq = $(round(mean(q)/Vq, digits=4))")
    println("  linear:    E[â'Σ̂â]/Va = $(round(mean(la)/Va, digits=4));  predicted (Va + noise)/Va = $(round((Va+Va_noise)/Va, digits=4))")
    println("  total ratio $(round((mean(q)+mean(la))/(Vq+Va), digits=4))")
end
decomp(het = true); decomp(het = false)
