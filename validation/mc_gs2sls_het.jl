# Heteroskedasticity-robust GS2SLS: `fit_gs2sls_panel(...; errors =
# :heteroskedastic)` against the default homoskedastic path (AUDIT finding 36).
#
# The SARAR model y = ρWy + Xβ + γ·en + u, u = λWu + ε, with one endogenous
# regressor `en` (instrumented by two excluded instruments) and ε = s ∘ e,
# where e is correlated with en's first-stage error. W is the row-standardised
# binary distance band on an irregular map, a quarter of whose points are
# packed into the north-east corner, so the neighbour count varies from 2
# to over 30. The error sd s is constant, or proportional to the neighbour
# count (scaled to mean 1), the design under which Arraiz et al. (2010) and
# Kelejian & Prucha (2010) find maximum likelihood biased.
#
# A. Cross-section: `NoFE()`, T_total = 2 (the fit drops the first period),
#    an intercept, 300 and 1,200 points, λ ∈ {0.4, 0.8}, ρ = 0.3. This is
#    the case Kelejian & Prucha's theory covers.
# B. Panel, T_total = 6, 300 points: unit effects (`IndivFE()`), and unit
#    and period effects (`TwoWayFE()`). The panel version is this package's
#    extension: the moments are pooled over periods on within-transformed
#    data and all covariances are cluster sums by unit.
#
# Run: julia --project=. validation/mc_gs2sls_het.jl   (MC_REPS=50 for a pilot)
# Output: validation/mc_gs2sls_het_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror

const R = parse(Int, get(ENV, "MC_REPS", "2000"))
const Z95 = 1.959963984540054
const OUT = joinpath(@__DIR__, "mc_gs2sls_het_output.txt")
io = open(OUT, "w")
say(s) = (println(s); println(io, s); flush(io))
t0 = time()
say("# Heteroskedasticity-robust GS2SLS against the homoskedastic path — Monte Carlo ($R replications per cell)")
say("Julia $(VERSION). Coverage is of nominal 95% intervals; its Monte Carlo sd at R = $R is $(round(sqrt(0.95 * 0.05 / R), digits = 4)). `hom` is the default `errors = :homoskedastic` (Kelejian–Prucha 1998 moments, Sargan test); `het` is `errors = :heteroskedastic` (Kelejian–Prucha 2010 moments, cluster-robust covariance, Hansen's J). `overid` is the rejection rate at 5% of the over-identification test, whose null is true throughout. β = (1.2, −0.6), γ = 0.8.")
say("")

function make_map(npts, band)
    rg = MersenneTwister(1)
    pts = rand(rg, npts, 2); q = npts ÷ 4
    pts[1:q, :] .= 0.6 .+ 0.4 .* pts[1:q, :]
    Wb = distance_weights(pts, band; decay = :power, alpha = 0.0, row_standardize = false)
    pts = pts[vec(sum(Wb .!= 0, dims = 2)) .> 0, :]
    Wb = distance_weights(pts, band; decay = :power, alpha = 0.0, row_standardize = false)
    nb = vec(sum(Wb .!= 0, dims = 2))
    return row_standardize(Wb), nb
end

function cell(W, nb, sdlab, λ, ρ, Tt, fe, intercept, seed)
    N = size(W, 1); NT = N * Tt
    sd = sdlab == "constant" ? ones(N) : nb ./ mean(nb)
    Sl = lu(sparse(1.0I, N, N) - ρ .* W); Bl = lu(sparse(1.0I, N, N) - λ .* W)
    truth = intercept ? [ρ, 1.0, 1.2, -0.6, 0.8, λ] : [ρ, 1.2, -0.6, 0.8, λ]
    jl = length(truth)
    res = Dict(e => (cov = zeros(Int, jl), lam = Float64[], rho = Float64[], over = Ref(0),
                     fail = Ref(0)) for e in (:homoskedastic, :heteroskedastic))
    rg = MersenneTwister(seed)
    for _ in 1:R
        X = randn(rg, NT, 2); Z = randn(rg, NT, 2); v = randn(rg, NT)
        en = Z * [0.5, -0.3] .+ 0.3 .* v
        μ = fe isa NoFE ? zeros(N) : randn(rg, N)
        ξ = fe isa TwoWayFE ? randn(rg, Tt) : zeros(Tt)
        y = zeros(NT)
        for t in 1:Tt
            id = ((t - 1) * N + 1):(t * N)
            ε = sd .* (0.6 .* randn(rg, N) .+ 0.5 .* v[id])
            y[id] = Sl \ ((intercept ? 1.0 : 0.0) .+ μ .+ ξ[t] .+ X[id, :] * [1.2, -0.6] .+
                          0.8 .* en[id] .+ (Bl \ ε))
        end
        Xf = intercept ? hcat(ones(NT), X) : X
        for e in (:homoskedastic, :heteroskedastic)
            r = res[e]
            m = try
                fit_gs2sls_panel(fe, W, Xf, reshape(en, :, 1), Z, y, N, Tt; errors = e)
            catch err
                err isa InterruptException && rethrow()
                r.fail[] += 1; continue
            end
            b = coef(m); s = stderror(m)
            r.cov .+= abs.(b .- truth) .<= Z95 .* s
            push!(r.lam, m.lambda); push!(r.rho, m.rho)
            r.over[] += m.sargan_pval < 0.05
        end
    end
    return res, truth
end

function report(lab, res, truth, λ, intercept)
    for e in (:homoskedastic, :heteroskedastic)
        r = res[e]; n = length(r.lam)
        c = r.cov ./ n
        cβ = intercept ? c[3] : c[2]
        say(@sprintf("| %s | %s | %.4f | %.4f | %.3f | %.4f | %.3f | %.3f | %.3f | %.3f | %d |",
                     lab, e === :homoskedastic ? "hom" : "het", mean(r.lam),
                     sqrt(mean((r.lam .- λ) .^ 2)), c[end], mean(r.rho), c[1], cβ, c[end - 1],
                     r.over[] / n, r.fail[]))
    end
end

hdr = "| N, λ, error sd | fit | mean λ̂ | rmse λ̂ | cov λ | mean ρ̂ | cov ρ | cov β₁ | cov γ | overid | failed |"
sep = "|---|---|---|---|---|---|---|---|---|---|---|"

say("## A. Cross-section (`NoFE()`, T_total = 2, intercept), ρ = 0.3")
say("")
say(hdr); say(sep)
for (npts, band) in ((300, 0.12), (1200, 0.06))
    W, nb = make_map(npts, band)
    for λ in (0.4, 0.8), sdlab in ("constant", "∝ neighbours")
        res, truth = cell(W, nb, sdlab, λ, 0.3, 2, NoFE(), true, 20261119)
        report("$(size(W, 1)), $λ, $sdlab", res, truth, λ, true)
    end
end
W, nb = make_map(1200, 0.06)
say("")
say(@sprintf("Neighbours per unit, 1,200-point map: min %d, median %d, max %d.", minimum(nb), Int(median(nb)), maximum(nb)))
W, nb = make_map(300, 0.12)
say(@sprintf("Neighbours per unit, 300-point map: min %d, median %d, max %d.", minimum(nb), Int(median(nb)), maximum(nb)))
say("")

say("## B. Panel, T_total = 6, $(size(W, 1)) units, λ = 0.4, ρ = 0.3")
say("")
say(replace(hdr, "N, λ, error sd" => "effects, error sd")); say(sep)
for (felab, fe) in (("IndivFE", IndivFE()), ("TwoWayFE", TwoWayFE())), sdlab in ("constant", "∝ neighbours")
    res, truth = cell(W, nb, sdlab, 0.4, 0.3, 6, fe, false, 20261120)
    report("$felab, $sdlab", res, truth, 0.4, false)
end
say("")
say(@sprintf("Run time %.0f s.", time() - t0))
close(io)
