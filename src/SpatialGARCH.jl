"""
    SpatialGARCH

Spatial log-GARCH for a cross-section (Sato & Matsuda 2021): spatial
dependence in the conditional **variance** rather than in the mean.

    y_i = √h_i ε_i,    log h = λW log h + ρW log y² + α1 + Zδ,

with `ε_i` i.i.d., `E ε_i² = 1`, and one row-standardised `W`. Writing
`Y = log y²`, `U = log ε²`, `c = E U`, `V = U − c` and `R(x) = I − xW`, the
model is a SARMA(1,1) in `Y`:

    R(λ + ρ) Y = Xβ + R(λ) V,    X = [1, Z],  β = (α + (1 − λ)c, δ).

Sato & Matsuda estimate it in two steps: Gaussian QML of the SARMA for
`ψ = (β, σ², λ, ρ)` (their Theorems 3.1–3.2: consistent and asymptotically
normal under a sandwich covariance), then `α` in closed form from the
`E ε² = 1` normalisation (Theorem 3.3: consistent, no distribution theory).

The score covariance `Ω` behind the sandwich is not given explicitly in the
paper or its supplement. It is derived here: at the true `ψ` every score
component is linear-quadratic in the i.i.d. `V`, so `Ω` has an exact closed
form in `σ²` and the third and fourth moments of `V`
(`SpatialUtils._lq_cov`). The test suite checks it against simulation.

The covariate-distance `W` that Fülle & Otto (2024) pair with this model is
[`covariate_weights`](@ref SpatialEconometrics.SpatialWeights.covariate_weights), in `SpatialWeights`.
"""
module SpatialGARCH

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
import StatsAPI
import ..SpatialUtils: check_design_rank, _print_coef_table, _logdet_eig, _lq_cov

export SpatialLogGARCHModel, fit_spatial_log_garch

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

"""
    SpatialLogGARCHModel

A fitted spatial log-GARCH, returned by [`fit_spatial_log_garch`](@ref).

Fields:
- `coef`, `vcov`, `coefnames`: `(δ, λ, ρ)` and their sandwich covariance
  `Ĥ⁻¹Ω̂Ĥ⁻¹`, which allows any distribution of `ε` with `E ε² = 1` and
  finite fourth moment of `log ε²`.
- `vcov_gaussian`: `−Ĥ⁻¹` for the same parameters. It is valid only if
  `log ε²` were normal. It never is when `ε` is normal: `log χ²₁` has
  excess kurtosis 4. It is kept for comparison, not for inference.
- `alpha`: the volatility intercept from step 2. It has **no standard
  error**: Sato & Matsuda prove it consistent but give no distribution.
- `beta0`: the intercept of the `log y²` equation, `α + (1 − λ)c`, a
  nuisance that mixes `α` with `c = E log ε²`. Its variance is in
  `vcov_full`.
- `vcov_full`, `fullnames`: the sandwich for all of `(β, σ², λ, ρ)`.
- `sigma2`, `mu3`, `mu4`: the variance and the third and fourth central
  moments of `V = log ε² − c`, from the step-1 residuals.
- `log_volatility`: the fitted `log ĥ`.
- `loglik`: the Gaussian quasi log-likelihood of the SARMA at the estimate.
- `iterations`, `gradient_norm`: the Newton iterations, and the final
  `max|∂ℓ/∂θ| / n` for `θ = (λ, λ + ρ)`.
"""
struct SpatialLogGARCHModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    vcov_gaussian::Matrix{T}
    coefnames::Vector{String}
    lambda::T
    rho::T
    delta::Vector{T}
    alpha::T
    beta0::T
    sigma2::T
    mu3::T
    mu4::T
    vcov_full::Matrix{T}
    fullnames::Vector{String}
    log_volatility::Vector{T}
    residuals::Vector{T}
    loglik::T
    n::Int
    iterations::Int
    gradient_norm::T
end

StatsAPI.coef(m::SpatialLogGARCHModel) = m.coef
StatsAPI.vcov(m::SpatialLogGARCHModel) = m.vcov
StatsAPI.stderror(m::SpatialLogGARCHModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.coefnames(m::SpatialLogGARCHModel) = m.coefnames
StatsAPI.nobs(m::SpatialLogGARCHModel) = m.n
StatsAPI.loglikelihood(m::SpatialLogGARCHModel) = m.loglik
StatsAPI.residuals(m::SpatialLogGARCHModel) = m.residuals
StatsAPI.dof_residual(m::SpatialLogGARCHModel) = m.n - length(m.fullnames)

function Base.show(io::IO, m::SpatialLogGARCHModel)
    println(io, "Spatial log-GARCH (Sato & Matsuda 2021), two-step QML")
    println(io, "══════════════════════════════════════════════════════")
    @printf(io, "  n = %d,  quasi log-lik = %.2f\n", m.n, m.loglik)
    @printf(io, "  λ + ρ = %.4f  (a stationary solution needs |λ + ρ| < 1: %s)\n",
            m.lambda + m.rho, abs(m.lambda + m.rho) < 1 ? "yes" : "NO")
    @printf(io, "  α = %.4f  (step 2; consistent, no standard error)\n", m.alpha)
    @printf(io, "  log ε²: variance %.3f, skewness %.3f, excess kurtosis %.3f\n",
            m.sigma2, m.mu3 / m.sigma2^1.5, m.mu4 / m.sigma2^2 - 3)
    println(io, "  Standard errors: sandwich Ĥ⁻¹Ω̂Ĥ⁻¹")
    println(io)
    _print_coef_table(io, m.coefnames, m.coef, StatsAPI.stderror(m))
end

# ─────────────────────────────────────────────────────────────────────────────
# Data, likelihood, score and Hessian
#
# Work in θ = (λ, a), a = λ + ρ, so that R(a)Y = Xβ + R(λ)V. With
# F = R(λ), Ỹ = F⁻¹R(a)Y, X̃ = F⁻¹X, Ỹ_W = F⁻¹WY, G = F⁻¹W and V = Ỹ − X̃β:
#
#   ℓ   = −n/2 log(2πσ²) − V'V/(2σ²) + log|R(a)| − log|R(λ)|
#   S_β = X̃'V/σ²
#   S_σ² = −n/(2σ²) + V'V/(2σ⁴)
#   S_λ = −V'GV/σ² + tr G
#   S_a = Ỹ_W'V/σ² − tr G_a,     G_a = W R(a)⁻¹,
#
# using ∂V/∂β = −X̃, ∂V/∂a = −Ỹ_W, ∂V/∂λ = GV, ∂X̃/∂λ = GX̃,
# ∂Ỹ_W/∂λ = GỸ_W and ∂G/∂λ = G². The traces come from the eigenvalues.
# ─────────────────────────────────────────────────────────────────────────────

struct _GData
    Y::Vector{Float64}
    WY::Vector{Float64}
    X::Matrix{Float64}
    W::SparseMatrixCSC{Float64, Int}
    μ::Vector{ComplexF64}
    n::Int
    p::Int
end

_logdetR(d::_GData, x) = _logdet_eig(x, real.(d.μ), imag.(d.μ))
_trG(d::_GData, x)  = real(sum(μ / (1 - x * μ) for μ in d.μ))
_trG2(d::_GData, x) = real(sum((μ / (1 - x * μ))^2 for μ in d.μ))

_Rλ(d::_GData, λ) = lu(sparse(1.0I, d.n, d.n) - λ * d.W)

# Everything the likelihood and its derivatives need at θ, with (β, σ²)
# either given or profiled out.
function _pieces(d::_GData, λ, a; β = nothing, σ2 = nothing)
    F = _Rλ(d, λ)
    Xt = F \ d.X
    YWt = F \ d.WY
    Yt = (F \ d.Y) .- a .* YWt
    βv = β === nothing ? Xt \ Yt : β
    V = Yt .- Xt * βv
    s2 = σ2 === nothing ? dot(V, V) / d.n : σ2
    return (; F, Xt, YWt, Yt, β = βv, V, σ2 = s2)
end

function _loglik(d::_GData, λ, a, P)
    n = d.n
    return -n / 2 * log(2π * P.σ2) - dot(P.V, P.V) / (2P.σ2) +
           _logdetR(d, a) - _logdetR(d, λ)
end

# Analytic score and Hessian of ℓ in ψ = (β, σ², λ, a), at the pieces P.
function _score_hessian(d::_GData, λ, a, P)
    n, p = d.n, d.p
    F, Xt, YWt, V, s2 = P.F, P.Xt, P.YWt, P.V, P.σ2
    GV = F \ (d.W * V)
    GXt = F \ (d.W * Xt)
    GYWt = F \ (d.W * YWt)
    GGV = F \ (d.W * GV)
    VGV = dot(V, GV)
    s4 = s2^2

    g = zeros(p + 3)
    g[1:p] = Xt' * V ./ s2
    g[p+1] = -n / (2s2) + dot(V, V) / (2s4)
    g[p+2] = -VGV / s2 + _trG(d, λ)
    g[p+3] = dot(YWt, V) / s2 - _trG(d, a)

    H = zeros(p + 3, p + 3)
    iβ = 1:p; iσ = p + 1; iλ = p + 2; ia = p + 3
    H[iβ, iβ] = -(Xt' * Xt) ./ s2
    H[iβ, iσ] = -(Xt' * V) ./ s4
    H[iβ, ia] = -(Xt' * YWt) ./ s2
    H[iβ, iλ] = (GXt' * V .+ Xt' * GV) ./ s2
    H[iσ, iσ] = n / (2s4) - dot(V, V) / (s4 * s2)
    H[iσ, ia] = -dot(YWt, V) / s4
    H[iσ, iλ] = VGV / s4
    H[ia, ia] = -dot(YWt, YWt) / s2 - _trG2(d, a)
    H[ia, iλ] = (dot(GYWt, V) + dot(YWt, GV)) / s2
    H[iλ, iλ] = -(dot(GV, GV) + 2dot(V, GGV)) / s2 + _trG2(d, λ)
    for i in 1:p+3, j in 1:i-1
        H[j, i] == 0 && (H[j, i] = H[i, j])
        H[i, j] = H[j, i]
    end
    return g, H
end

# ─────────────────────────────────────────────────────────────────────────────
# The score as linear-quadratic forms in V, and its exact covariance
#
# At the true ψ, Y = R(a)⁻¹(Xβ + R(λ)V), so Ỹ_W = d̄ + HV with
# d̄ = F⁻¹G_aXβ and H = F⁻¹G_aF (tr H = tr G_a). Then
#
#   S_β_k = X̃_k'V/σ²,               S_σ² = V'V/(2σ⁴) − n/(2σ²),
#   S_λ   = −V'sym(G)V/σ² + tr G,   S_a  = d̄'V/σ² + V'sym(H)V/σ² − tr H,
#
# each of the form a'V + V'AV − σ² tr A.
# ─────────────────────────────────────────────────────────────────────────────

_sym(M) = (M .+ M') ./ 2

function _lq_forms(d::_GData, λ, a, β, σ2)
    n, p = d.n, d.p
    F = _Rλ(d, λ)
    Wd = Matrix(d.W)
    Rλ = sparse(1.0I, n, n) - λ * d.W
    Ga = lu(sparse(1.0I, n, n) - a * d.W) \ Wd          # R(a)⁻¹W = W R(a)⁻¹
    G = F \ Wd
    Hm = F \ (Ga * Rλ)
    Xt = F \ d.X
    dbar = F \ (Ga * (d.X * β))
    z = zeros(n)
    forms = Any[]
    for k in 1:p
        push!(forms, (nothing, Xt[:, k] ./ σ2))
    end
    push!(forms, (Matrix((1 / (2σ2^2)) * I, n, n), z))
    push!(forms, (-_sym(G) ./ σ2, z))
    push!(forms, (_sym(Hm) ./ σ2, dbar ./ σ2))
    return forms
end

"""
    _score_cov(d, λ, a, β, σ2, μ3, μ4)

Exact `Var(S(ψ))` at `ψ`, for i.i.d. `V` with variance `σ2` and central
moments `μ3`, `μ4`, in the order `(β, σ², λ, a)`.
"""
_score_cov(d::_GData, λ, a, β, σ2, μ3, μ4) = _lq_cov(_lq_forms(d, λ, a, β, σ2), σ2, μ3, μ4)

# ─────────────────────────────────────────────────────────────────────────────
# Estimation
# ─────────────────────────────────────────────────────────────────────────────

function _concentrated(d::_GData, λ, a)
    P = _pieces(d, λ, a)
    return _loglik(d, λ, a, P), P
end

function _newton(d::_GData, λ, a, lo, hi; tol, maxiter)
    inside(x) = lo < x < hi
    ll, P = _concentrated(d, λ, a)
    p = d.p
    iφ = 1:p+1; iθ = p+2:p+3
    for it in 1:maxiter
        g, H = _score_hessian(d, λ, a, P)
        gθ = g[iθ]
        gnorm = maximum(abs, gθ) / d.n
        gnorm < tol && return λ, a, ll, P, it - 1, gnorm
        Hc = H[iθ, iθ] - H[iθ, iφ] * (H[iφ, iφ] \ H[iφ, iθ])
        step = isposdef(-Hc) ? -(Hc \ gθ) : gθ ./ (d.n * maximum(abs, gθ) / d.n + 1e-12) .* 0.1
        t = 1.0
        accepted = false
        for _ in 1:60
            λn, an = λ + t * step[1], a + t * step[2]
            if inside(λn) && inside(an)
                lln, Pn = _concentrated(d, λn, an)
                if lln >= ll - 1e-10 * abs(ll)
                    λ, a, ll, P = λn, an, lln, Pn
                    accepted = true
                    break
                end
            end
            t /= 2
        end
        if !accepted
            g2, _ = _score_hessian(d, λ, a, P)
            gn = maximum(abs, g2[iθ]) / d.n
            gn < tol && return λ, a, ll, P, it, gn
            throw(_search_error("the Newton search stalled", λ, a, lo, hi, gn, tol))
        end
    end
    g, _ = _score_hessian(d, λ, a, P)
    gn = maximum(abs, g[iθ]) / d.n
    gn < tol && return λ, a, ll, P, maxiter, gn
    throw(_search_error("no convergence in $maxiter Newton iterations", λ, a, lo, hi, gn, tol))
end

# The error a failed search throws. When the iterates end beyond the outermost
# grid point, the search has run to the edge of `bounds`; the message then says
# why that happens (the likelihood is unbounded as λ → 1, see the docstring).
function _search_error(what, λ, a, lo, hi, gn, tol)
    head = "fit_spatial_log_garch: $what at λ = $(round(λ, digits = 4)), " *
           "λ + ρ = $(round(a, digits = 4)) with max|∂ℓ/∂θ|/n = " *
           "$(round(gn, sigdigits = 3)) > tol = $tol."
    edge = (hi - lo) / 40
    if min(λ - lo, hi - λ, a - lo, hi - a) < edge
        return ErrorException(head * " The search ran to the edge of `bounds`: " *
            "this sample has no interior maximum. With a row-standardised W and an " *
            "intercept, the quasi log-likelihood rises without bound as λ → 1, so an " *
            "interior local maximum is the only estimate there is. This happens most " *
            "when λ + ρ is near 1 and n is small. See the docstring.")
    end
    return ErrorException(head * " The quasi likelihood may be flat in λ: see the " *
        "identification note in the docstring.")
end

"""
    fit_spatial_log_garch(y, W; Z = nothing, varnames = nothing,
                          eigenvalues = nothing, bounds = (-0.99, 0.99),
                          tol = 1e-8, maxiter = 100) -> SpatialLogGARCHModel

Fit the spatial log-GARCH of Sato & Matsuda (2021) to a cross-section `y`:

    y_i = √h_i ε_i,    log h = λW log h + ρW log y² + α1 + Zδ.

`y` is mean-zero data: returns, or the residuals of a mean model (Sato &
Matsuda fit SAR residuals). `W` must be row-standardised (every row sums to
1); the theory covers one `W` in both places. `Z` holds optional exogenous
variables, without an intercept.

**Step 1** maximises the Gaussian quasi log-likelihood of the SARMA(1,1)
form `R(λ+ρ) log y² = Xβ + R(λ)V` over `θ = (λ, λ + ρ)`, with `β` and `σ²`
profiled out: a 21 × 21 grid inside `bounds`, then Newton steps with step
halving until `max|∂ℓ/∂θ|/n < tol`. The fit throws if that is not reached;
it never returns a point that does not solve the first-order conditions.
`log|R(x)|` and its traces come from the eigenvalues of `W`, computed once
(`O(n³)`) or passed in with `eigenvalues`.

**Step 2** sets `α̂ = (1 − λ̂) log(n⁻¹ Σ exp C_i)`, with
`C = R(λ̂)⁻¹(R(λ̂ + ρ̂) log y² − Zδ̂)` (eq. 11). It has no standard error.

**Standard errors** are the sandwich `Ĥ⁻¹Ω̂Ĥ⁻¹`. `Ĥ` is the analytic Hessian
and `Ω̂` the exact score covariance at the estimate, with the variance, third
and fourth moments of `log ε²` taken from the residuals. The paper does not
state `Ω` explicitly; it is derived here and checked by simulation in the
test suite.

!!! warning "No interior maximum near λ = 1"
    Because `W1 = 1`, the intercept absorbs the part of the residual that
    `R(λ)⁻¹` inflates as `λ → 1`, so `σ̂²(λ)` stays bounded while
    `−log|R(λ)|` grows like `−log(1 − λ)`. The quasi log-likelihood is
    therefore unbounded above as `λ → 1`, in every sample (for a connected
    `W`). The estimate is the interior local maximum. When a sample has none,
    the search runs to the edge of `bounds` and the fit throws. In the
    package's Monte Carlo this happens in 36–39% of samples with
    `(λ, ρ) = (0.9, 0.05)` and `n = 100` (rook `W`), and in none at `n = 400`.

!!! warning "Identification of λ"
    The mean of `log y²` depends on `λ` and `ρ` only through `λ + ρ`. `λ` is
    identified separately only by the moving-average part of the error,
    `R(λ)V`, so it is estimated less precisely than `λ + ρ`, and with an
    intercept alone (no `Z`) it rests on that part entirely. Sato & Matsuda
    assume identification (their Assumption 3.8) without stating this. The
    package's Monte Carlo measures what it costs; see the manual page.

!!! note "Zeros"
    `log y²` is undefined at `y_i = 0`, so the fit throws if any `y_i` is
    zero. Rounded returns often have exact zeros; the model does not cover
    them.

# Example
```julia
W = make_rook_weights(20)                       # 400 units
m = fit_spatial_log_garch(y, W; Z = z)
m.lambda, m.rho, m.alpha, stderror(m)
```
"""
function fit_spatial_log_garch(
    y::AbstractVector{<:Real}, W::AbstractSparseMatrix{<:Real};
    Z::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
    varnames::Union{Nothing, Vector{String}} = nothing,
    eigenvalues::Union{Nothing, AbstractVector{<:Number}} = nothing,
    bounds::Tuple{Real, Real} = (-0.99, 0.99),
    tol::Real = 1e-8,
    maxiter::Int = 100,
)
    n = length(y)
    size(W) == (n, n) || throw(DimensionMismatch("W is $(size(W)), y has length $n"))
    nz = count(iszero, y)
    nz > 0 && throw(ArgumentError("fit_spatial_log_garch: $nz of the $n values of y are " *
        "exactly zero, where log y² is undefined. The model does not cover zeros."))
    any(!isfinite, y) && throw(ArgumentError("fit_spatial_log_garch: y has non-finite values."))
    rs = vec(sum(W, dims = 2))
    maximum(abs, rs .- 1) < 1e-8 || throw(ArgumentError("fit_spatial_log_garch: W must be " *
        "row-standardised (every row summing to 1); the largest deviation is " *
        "$(round(maximum(abs, rs .- 1), sigdigits = 3)). Use row_standardize(W)."))
    lo, hi = Float64(bounds[1]), Float64(bounds[2])
    lo < 0 < hi || throw(ArgumentError("bounds must contain 0"))

    Zm = Z === nothing ? zeros(n, 0) : Float64.(Z isa AbstractVector ? reshape(Z, :, 1) : Z)
    size(Zm, 1) == n || throw(DimensionMismatch("Z has $(size(Zm, 1)) rows, y has $n"))
    q = size(Zm, 2)
    znames = varnames === nothing ? ["z$k" for k in 1:q] : varnames
    length(znames) == q || throw(ArgumentError("varnames has $(length(znames)) names for $q columns of Z"))
    X = hcat(ones(n), Zm)
    check_design_rank(X, vcat(["(Intercept)"], znames), "fit_spatial_log_garch";
                      context = "`Z` must not contain its own intercept, since one is added.")

    Ws = SparseMatrixCSC{Float64, Int}(W)
    μ = eigenvalues === nothing ? ComplexF64.(eigvals(Matrix(Ws))) : ComplexF64.(eigenvalues)
    length(μ) == n || throw(ArgumentError("eigenvalues has length $(length(μ)), expected $n"))
    Y = log.(Float64.(y) .^ 2)
    d = _GData(Y, Ws * Y, X, Ws, μ, n, q + 1)

    # Grid in (λ, a), then Newton.
    grid = range(lo + (hi - lo) / 40, hi - (hi - lo) / 40, length = 21)
    best = (-Inf, 0.0, 0.0)
    for λg in grid, ag in grid
        ll, _ = _concentrated(d, λg, ag)
        ll > best[1] && (best = (ll, λg, ag))
    end
    λ, a, ll, P, iters, gnorm = _newton(d, best[2], best[3], lo, hi;
                                        tol = Float64(tol), maxiter = maxiter)

    # Sandwich at ψ̂.
    V = P.V
    σ2 = P.σ2
    μ3 = mean(V .^ 3)
    μ4 = mean(V .^ 4)
    _, H = _score_hessian(d, λ, a, P)
    Ω = _score_cov(d, λ, a, P.β, σ2, μ3, μ4)
    Hi = inv(H)
    Vψ = Hi * Ω * Hi
    Vg = -Hi
    p = d.p
    J = Matrix(1.0I, p + 3, p + 3)
    J[p+3, p+2] = -1.0                     # ρ = a − λ
    Vψ = J * Vψ * J'
    Vg = J * Vg * J'
    Vψ = (Vψ + Vψ') / 2; Vg = (Vg + Vg') / 2

    # Step 2 (eq. 11) and the fitted log-volatility.
    ρ = a - λ
    δ = P.β[2:end]
    F = P.F
    C = F \ (Y .- a .* d.WY .- Zm * δ)
    Cmax = maximum(C)
    α = (1 - λ) * (Cmax + log(mean(exp.(C .- Cmax))))
    logh = F \ (ρ .* d.WY .+ α .+ Zm * δ)

    idx = vcat(2:p, p + 2, p + 3)
    names = vcat(znames, ["λ", "ρ"])
    fullnames = vcat(["β0 (log y² intercept)"], znames, ["σ²", "λ", "ρ"])
    return SpatialLogGARCHModel{Float64}(
        vcat(δ, λ, ρ), Vψ[idx, idx], Vg[idx, idx], names, λ, ρ, δ, α, P.β[1],
        σ2, μ3, μ4, Vψ, fullnames, logh, V, ll, n, iters, gnorm)
end

end # module
