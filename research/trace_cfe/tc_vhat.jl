# Theory step 2, numerical side: is the robust variance V̂ of the exact-diagonal moment unbiased when K/n
# does not vanish?  THEORY.md conjectures that the plug-in σ̂ᵢ² = ûᵢ²/Mᵢᵢ is biased by about hᵢᵢ under
# heteroskedasticity, so that E V̂/V stays away from 1 when K/n → c > 0.
#
# Fixed design (W, B, f, σ), errors redrawn R times, everything evaluated at the true ρ (so that only the
# variance estimator is under test, not ρ̂). The moment is g = ε′Aε + (MWf)′ε, A = MW + MDM, diag(A) = 0,
# with exact variance  V = 2σ²′(As∘As)σ² + Σ bᵢ²σᵢ²,  b = MWf.
#
# Variants of V̂ = 2s′(As∘As)s + Σ(âᵢ² − noiseᵢ)sᵢ², which differ only in the variances s plugged in:
#   oracle : s = σ², the true variances. Exactly unbiased, so it checks the code and the noise term.
#   HC2    : s = û²/Mᵢᵢ, the current estimator (tc_lib.jl, tc_varg).
#   CJN    : s = (M∘M)⁻¹û², Cattaneo, Jansson & Newey (2018). E s = σ² exactly, so its linear part is
#            unbiased; its quadratic part still carries Σ cᵢⱼ² Cov(sᵢ, sⱼ).
#   CJN-G  : CJN with that covariance removed under normality: Cov(ûᵢ², ûⱼ²) = 2(MΣM)ᵢⱼ², with MΣM
#            estimated by M Diag(s) M.
# The power check is the homoskedastic V̂ (σ̄² everywhere), which must fail under heteroskedasticity.
include(joinpath(@__DIR__, "tc_lib.jl"))

function vhat_check(; n = 900, dsurf = 12, R = 1000, law = :normal, var = :smooth, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hcat(surfbasis(u, v, dsurf), x)
    D = tc_design(B, W)
    h = 1 .- D.Mii
    s = sin.(2π .* u) .* cos.(π .* v)
    f = D.Q * (D.Q' * s) .+ x
    σ2 = var == :homo ? ones(n) :
         var == :smooth ? exp.(3 .* (u .+ v .- 1)) :
         exp.(randn(rng, n))                       # :rough, unrelated to location
    σ2 ./= mean(σ2)
    b = Mproj(D, D.Wd * f)
    V = 2 * dot(σ2, D.As2 * σ2) + sum(b .^ 2 .* σ2)
    MM = D.M .* D.M
    MMf = lu(MM)
    MMi = inv(MM)
    T = MMi * D.As2 * MMi                          # bias of the quadratic part = 2 Σᵢⱼ Tᵢⱼ Cov(ûᵢ², ûⱼ²)
    H2 = D.G                                       # â = b + G Q′ε, so Var(â) = G(Q′ΣQ)G′
    Asum = sum(D.As2)
    function vh(s, a)
        noise = vec(sum((H2 * (D.Q' * (s .* D.Q))) .* H2, dims = 2))
        2 * dot(s, D.As2 * s) + sum((a .^ 2 .- noise) .* s)
    end
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    out = Dict(k => zeros(R) for k in (:oracle, :hc2, :cjn, :cjng, :homo, :g))
    for r in 1:R
        ε = sqrt.(σ2) .* draw()
        y0 = f .+ ε                                # y − ρ₀Wy
        û = Mproj(D, y0)
        a = Mproj(D, D.Wd * (y0 .- û))             # M W P_B y0
        out[:g][r] = dot(û, Mproj(D, D.Wd * y0) .+ D.d .* û)
        out[:oracle][r] = vh(σ2, a)
        s_hc2 = û .^ 2 ./ D.Mii
        out[:hc2][r] = vh(s_hc2, a)
        s_cjn = MMf \ (û .^ 2)
        vc = vh(s_cjn, a)
        out[:cjn][r] = vc
        # Gaussian: Cov(û²) = 2(MΣM)∘(MΣM), and Cov(s) = (M∘M)⁻¹ Cov(û²) (M∘M)⁻¹
        QS = D.Q' * (s_cjn .* D.Q)                 # M Diag(s) M = Diag(s) − QQ′S − SQQ′ + Q(Q′SQ)Q′
        SQ = s_cjn .* D.Q
        MSM = Diagonal(s_cjn) .- D.Q * SQ' .- SQ * D.Q' .+ D.Q * (QS * D.Q')
        out[:cjng][r] = vc - 2 * sum(T .* (2 .* MSM .^ 2))
        sb = mean(s_hc2)
        out[:homo][r] = 2 * sb^2 * Asum + sb * dot(a, a)
    end
    zt(x) = (mean(x) - V) / (std(x) / sqrt(R))
    r3(x) = round(x, digits = 3)
    g2 = out[:g] .^ 2
    print(rpad("$law $var K=$(D.K)", 22), " K/n $(r3(D.K / n)) maxh $(r3(maximum(h))) | Var g/V $(r3(mean(g2) / V)) (z $(r3((mean(g2) - V) / (std(g2) / sqrt(R)))))")
    for k in (:oracle, :hc2, :cjn, :cjng, :homo)
        print(" | $k $(r3(mean(out[k]) / V)) (z $(round(zt(out[k]), digits = 1)))")
    end
    println()
end

t = @elapsed vhat_check(; R = 5)
println("warm-up $(round(t, digits = 1)) s. Each cell: E V̂/V with its z against V (R = 1000; |z| > 3 is a real bias).\n")
for law in (:normal, :chisq), var in (:homo, :smooth, :rough)
    for dsurf in (8, 12, 17, 21, 25)
        vhat_check(; dsurf, law, var)
    end
    println()
end
