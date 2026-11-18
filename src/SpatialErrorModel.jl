module SpatialErrorModel

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, g_traces,
       brent_maximize, panel_spatial_lag!, panel_spatial_lag,
       within_transform!, within_transform, effective_dims, has_indiv_fe, has_time_fe,
       check_design_rank, _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox
const FixedEffectsSpec = SpatialUtils.FixedEffectsSpec
const NoFE = SpatialUtils.NoFE
const IndivFE = SpatialUtils.IndivFE
const TimeFE = SpatialUtils.TimeFE
const TwoWayFE = SpatialUtils.TwoWayFE

export SpatialErrorRegressionModel, SpatialErrorPanelModel,
       fit_spatial_error

# ─── Cross-section SEM result ─────────────────────────────────────────

"""
    SpatialErrorRegressionModel

A fitted cross-sectional spatial error model, returned by
[`fit_spatial_error`](@ref). Implements the `StatsAPI.RegressionModel`
interface.

Carries `coef` and `vcov` for the regression parameters, the spatial error
parameter `lambda` with its standard error, `sigma2`, `loglik`, the weight
matrix, residuals and `coefnames`.

!!! note
    Unlike `SpatialRegressionModel`, `coef` **is** interpretable
    directly: dependence sits in the error, so there is no feedback through
    `y` and no need for `spatial_effects`. What the model buys you
    is correct standard errors, not different point estimates.
"""
struct SpatialErrorRegressionModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    lambda::T
    lambda_se::T
    sigma2::T
    loglik::T
    W::M
    X::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

# ─── Panel SEM result ─────────────────────────────────────────────────

"""
    SpatialErrorPanelModel

A fitted spatial error panel, returned by the panel method of
[`fit_spatial_error`](@ref).

As [`SpatialErrorRegressionModel`](@ref), with the fixed-effects
specification and the panel dimensions `N` and `T_per` recorded alongside.
"""
struct SpatialErrorPanelModel{FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    fe_spec::FE
    N::Int
    T_periods::Int
    N_eff::Int
    T_eff::Int
    coef::Vector{T}
    vcov::Matrix{T}
    lambda::T
    lambda_se::T
    sigma2::T
    loglik::T
    W::M
    X_tilde::Matrix{T}
    y_tilde::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

# ─── Cross-section SEM estimation ─────────────────────────────────────

"""
    fit_spatial_error(W, X, y; method, varnames, lambda_bounds)
    fit_spatial_error(fe_spec, W, X, y, N, T_per; method, varnames, lambda_bounds)

Fit a spatial error model by concentrated maximum likelihood.

```math
y = X*beta + u,  u = lambda*W*u + e
```

Dependence is in the disturbance rather than in `y`. OLS stays unbiased
under this process but its standard errors are wrong, so the gain is
inferential: `beta` is interpretable as it stands, with no spillover
decomposition needed.

The likelihood is concentrated by spatially filtering `y` and `X` with
`(I - lambda*W)` at each trial `lambda`, then maximizing over `lambda`
alone.

The first method fits a cross-section; the second fits a panel, taking a
`FixedEffectsSpec` and the panel dimensions, with `X` and `y`
stacked units-fastest.

# Arguments
- `W`: `N x N` sparse weight matrix.
- `X`: covariates without an intercept — one is added.
- `y`: response.

# Keywords
- `method`: a `LogDetMethod`.
- `varnames`: covariate names.
- `lambda_bounds`: search interval for `lambda`, default
  `(-0.995, 0.995)`.
- `estimator` (cross-section only): `:ml` (default), Gaussian maximum
  likelihood, or `:gmm`, Kelejian & Prucha's (2010) generalized moments
  estimator, which stays consistent, with valid standard errors, when the
  error variance differs across units. See "Heteroskedasticity" below.

# Returns
A [`SpatialErrorRegressionModel`](@ref) or
[`SpatialErrorPanelModel`](@ref).

# Examples
```julia
m = fit_spatial_error(W, X, y; varnames = ["income", "density"])
m.lambda
coef(m)                              # directly interpretable

fit_spatial_error(TwoWayFE(), W, X, y, N, T_per)
```

Use `lm_tests` to decide between this and `fit_spatial`
before fitting either.

# Heteroskedasticity: `estimator = :gmm`

Maximum likelihood assumes every unit's error has the same variance, and
when it does not, `λ̂` can be inconsistent and the information-matrix
standard errors are wrong (Arraiz, Drukker, Kelejian & Prucha 2010).
`estimator = :gmm` is Kelejian & Prucha's (2010) estimator for exogenous
`X`. `λ` comes from the two moment conditions

    E[ε′(W′W − diag(W′W))ε] = 0,    E[ε′Wε] = 0,

which hold whatever the variances, because both matrices have a zero
diagonal (so `W` must have one). `β` is then OLS on the filtered data
(feasible GLS), and `λ` is re-estimated from those residuals, as in their
two-step procedure. The covariance of `β` is the heteroskedasticity-robust
jackknife (HC3) sandwich of the filtered regression, which allows for
high-leverage units; `λ`'s comes from their eq. (16).
With exogenous `X` the two are asymptotically independent. There is no
likelihood: `loglik` is `NaN` and `sigma2` is the mean squared filtered
residual.

The price is efficiency when the variance is in fact constant: Arraiz et
al. report root mean square errors of `λ̂` about 5–8% above ML's there, in a
model with a spatial lag of `y` as well. In this package's study of the
pure error model (`validation/mc_spatial_did_gmm.jl`, part C: 500 points on
an irregular map, a binary distance band, 2,000 replications) the gap was
at most 1%. With the error sd proportional to each unit's number of
neighbours, ML's `λ̂` was biased and its interval under-covered (0.733 for
`λ = 0.8`, coverage 0.745; 0.415 for 0.5, 0.821) where GMM's was not
(0.783, 0.950; 0.480, 0.948). `β`'s intervals covered under both.

```julia
m = fit_spatial_error(W, X, y; estimator = :gmm)
```

!!! note "Identification"
    The design has to have full column rank, and the matrix checked is the
    **within-transformed** one, since that is what gets solved. Under
    `IndivFE` a regressor that does not vary over time is swept out
    entirely, and under `TimeFE` one that does not vary across units is;
    neither coefficient can be estimated from the within variation, and
    both now throw an `ArgumentError` naming the column instead of
    returning a number. The same applies to any covariate that is a linear
    combination of the others.

    In the cross-sectional method there is no transform, and the check is
    on `[1 X]` as passed.

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

Arraiz, I., Drukker, D. M., Kelejian, H. H., & Prucha, I. R. (2010). A
spatial Cliff-Ord-type model with heteroskedastic innovations: small and
large sample results. *Journal of Regional Science*, 50(2), 592–614.
https://doi.org/10.1111/j.1467-9787.2009.00618.x

Kelejian, H. H., & Prucha, I. R. (2010). Specification and estimation of
spatial autoregressive models with autoregressive and heteroskedastic
disturbances. *Journal of Econometrics*, 157(1), 53–67.
https://doi.org/10.1016/j.jeconom.2009.10.025
"""
function fit_spatial_error(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995)),
    max_exact_n::Int=1000,
    vcov_probes::Int=250,
    rng::AbstractRNG=Random.default_rng(),
    estimator::Symbol=:ml
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert size(W) == (N, N) "Dimension mismatch between W and X"
    @assert length(y) == N "Dimension mismatch between y and X"
    estimator in (:ml, :gmm) || throw(ArgumentError(
        "unknown `estimator = :$estimator`; expected :ml (maximum likelihood) or :gmm " *
        "(Kelejian & Prucha 2010, robust to heteroskedasticity)"))

    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    X_aug = hcat(ones(T, N), X)
    coefnames_vec = vcat(["(Intercept)"], vnames, ["λ"])
    check_design_rank(X_aug, coefnames_vec, "fit_spatial_error";
                      context = """`X` must not contain its own intercept, since one is added.""")

    if estimator === :gmm
        o = _sem_gmm_het(W, X_aug, Vector{T}(y), N, 1; bounds = lambda_bounds)
        P = size(X_aug, 2)
        V = zeros(T, P + 1, P + 1)
        V[1:P, 1:P] .= o.vcov_beta
        V[P + 1, P + 1] = o.lambda_se^2
        return SpatialErrorRegressionModel{T, M}(
            vcat(o.beta, o.lambda), V, o.lambda, o.lambda_se, o.sigma2, T(NaN),
            W, Matrix{T}(X_aug), Vector{T}(y), o.residuals, coefnames_vec)
    end

    Wy = W * y
    WX = W * X_aug

    # Concentrated ML: for each λ, filter y_λ = y - λWy, X_λ = X - λWX
    # β(λ) = X_λ \ y_λ, σ²(λ) = ||y_λ - X_λ β(λ)||² / N
    qr_X = qr(X_aug)
    qr_WX = qr(WX)

    β0 = qr_X \ y
    βd = qr_X \ Wy
    Xβ0_w = qr_X \ (WX * β0)  # not needed directly

    # More efficient: precompute components for concentrated LL
    # y_λ = y - λ Wy, X_λ = X_aug - λ WX
    # β(λ) = (X_λ'X_λ)^{-1} X_λ'y_λ
    # Easier: use the Anselin (1988) trick
    # e_0 = OLS resid from y on X, e_d = OLS resid from Wy on X
    # Then resid(λ) = e_0 - λ e_d
    e0_y = y .- X_aug * (qr_X \ y)
    ed_y = Wy .- X_aug * (qr_X \ Wy)

    # But SEM filtering is different from SAR. We need:
    # y_λ = y - λWy, X_λ = X - λWX, then OLS on (X_λ, y_λ)
    # The Anselin trick: define a0 = X\y, ad = X\(Wy), b0 = X\(WX * a0), ...
    # Actually for SEM the concentrated LL approach is simpler:
    # For each λ, compute y_λ, X_λ, then β_λ = X_λ\y_λ, σ²_λ, LL_λ.
    # Since brent_maximize only evaluates ~50 times, this is fast.

    SpatialUtils._check_spatial_bounds(W, lambda_bounds, method; name = "lambda_bounds", caller = "fit_spatial_error")
    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = λ -> chebyshev_logdet(λ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = λ -> first(logabsdet(lu(I_sparse - λ * W)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))

    function concentrated_sem_ll(λ)
        y_filt = y .- λ .* Wy
        X_filt = X_aug .- λ .* WX
        β_filt = X_filt \ y_filt
        res_filt = y_filt .- X_filt * β_filt
        σ2_filt = dot(res_filt, res_filt) / N
        return const_term + logdet_fn(λ) - T(0.5) * N * log(σ2_filt)
    end

    λ_hat, max_ll = brent_maximize(concentrated_sem_ll, lambda_bounds[1], lambda_bounds[2])
    if method isa ChebyshevApprox
        # Chebyshev locates the peak; the exact log-determinant places it (AUDIT 33c).
        I_pol = sparse(one(T) * I, size(W, 1), size(W, 1))
        logdet_fn = r -> first(logabsdet(lu(I_pol - r * W)))
        λ_hat, max_ll = SpatialUtils._polish_rho(concentrated_sem_ll, λ_hat, lambda_bounds)
    end
    SpatialUtils._warn_at_bound(λ_hat, lambda_bounds; name = "lambda_bounds", caller = "fit_spatial_error")

    # Final estimates at λ_hat
    y_filt = y .- λ_hat .* Wy
    X_filt = X_aug .- λ_hat .* WX
    β_hat = X_filt \ y_filt
    res = y_filt .- X_filt * β_hat
    σ2_hat = dot(res, res) / N

    # Asymptotic variance
    vcov_mat, λ_se = _sem_asymptotic_vcov(X_filt, W, λ_hat, σ2_hat, N;
                                          max_exact_n=max_exact_n,
                                          n_probes=vcov_probes, rng=rng)
    full_coef = vcat(β_hat, λ_hat)

    return SpatialErrorRegressionModel{T, M}(
        full_coef, vcov_mat, λ_hat, λ_se, σ2_hat, max_ll,
        W, Matrix{T}(X_aug), Vector{T}(y), res, coefnames_vec
    )
end

function _sem_asymptotic_vcov(
    X_filt::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    λ::T,
    σ2::T,
    N::Int;
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    P = size(X_filt, 2)

    # G = W(I - λW)⁻¹. These three traces were estimated from 30 Hutchinson
    # probes drawn off the global RNG, which left λ's standard error moving
    # by 8% between runs on identical data while β's -- built from
    # X_filt'X_filt -- stayed put.
    tr_G, tr_G2, tr_GtG = g_traces(W, λ; max_exact_n=max_exact_n,
                                   n_probes=n_probes, rng=rng)

    # In the error model β is asymptotically independent of (λ, σ²), so the
    # information matrix is block diagonal and I_βλ stays zero.
    I_ββ = (X_filt' * X_filt) ./ σ2
    I_λλ = tr_G2 + tr_GtG
    I_λσ2 = tr_G / σ2
    I_σ2σ2 = N / (2 * σ2^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    # β and λ are asymptotically independent in SEM
    Info[P + 1, P + 1] = I_λλ
    Info[P + 1, P + 2] = I_λσ2
    Info[P + 2, P + 1] = I_λσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]
    λ_se = sqrt(max(zero(T), V_coef[P + 1, P + 1]))

    return V_coef, λ_se
end

# ─── Panel SEM estimation ─────────────────────────────────────────────

function fit_spatial_error(
    fe_spec::FE,
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_per::Int;
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995)),
    max_exact_n::Int=1000,
    vcov_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    NT, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    if fe_spec isa NoFE
        X_aug = hcat(ones(T, NT), X)
        coefnames_vec = vcat(["(Intercept)"], vnames, ["λ"])
    else
        X_aug = copy(X)
        coefnames_vec = vcat(vnames, ["λ"])
    end

    # Spatial lag of y and X for filtering
    Wy = similar(y); panel_spatial_lag!(Wy, W, y, N, T_per)
    WX = panel_spatial_lag(W, X_aug, N, T_per)

    # Within-transform
    y_tilde = within_transform!(copy(y), N, T_per, fe_spec)
    Wy_tilde = within_transform!(copy(Wy), N, T_per, fe_spec)
    X_tilde = within_transform(X_aug, N, T_per, fe_spec)
    # The transformed design is the one that gets solved, so it is the one
    # to check: a regressor the within transform absorbs leaves a column of
    # zeros here while the column the caller passed looks perfectly fine.
    check_design_rank(X_tilde, coefnames_vec, "fit_spatial_error";
                      context = """Under individual fixed effects a regressor that does not vary over time is swept out entirely, and under time fixed effects one that does not vary across units is; such a coefficient cannot be estimated from the within variation. Drop the column, or change `fe_spec`.""")
    WX_tilde = within_transform(WX, N, T_per, fe_spec)

    N_eff, T_eff = effective_dims(N, T_per, fe_spec)
    NT_eff = N_eff * T_eff

    SpatialUtils._check_spatial_bounds(W, lambda_bounds, method; name = "lambda_bounds", caller = "fit_spatial_error")
    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = λ -> chebyshev_logdet(λ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = λ -> first(logabsdet(lu(I_sparse - λ * W)))
    end

    const_term = -T(0.5) * NT_eff * (log(T(2π)) + one(T))

    function concentrated_panel_sem_ll(λ)
        y_filt = y_tilde .- λ .* Wy_tilde
        X_filt = X_tilde .- λ .* WX_tilde
        β_filt = X_filt \ y_filt
        res_filt = y_filt .- X_filt * β_filt
        σ2_filt = dot(res_filt, res_filt) / NT_eff
        ld = logdet_fn(λ)
        if has_time_fe(fe_spec)
            ld -= log(one(λ) - λ)
        end
        return const_term + T_eff * ld - T(0.5) * NT_eff * log(σ2_filt)
    end

    λ_hat, max_ll = brent_maximize(concentrated_panel_sem_ll, lambda_bounds[1], lambda_bounds[2])
    if method isa ChebyshevApprox
        # Chebyshev locates the peak; the exact log-determinant places it (AUDIT 33c).
        I_pol = sparse(one(T) * I, size(W, 1), size(W, 1))
        logdet_fn = r -> first(logabsdet(lu(I_pol - r * W)))
        λ_hat, max_ll = SpatialUtils._polish_rho(concentrated_panel_sem_ll, λ_hat, lambda_bounds)
    end
    SpatialUtils._warn_at_bound(λ_hat, lambda_bounds; name = "lambda_bounds", caller = "fit_spatial_error")

    y_filt = y_tilde .- λ_hat .* Wy_tilde
    X_filt = X_tilde .- λ_hat .* WX_tilde
    β_hat = X_filt \ y_filt
    res = y_filt .- X_filt * β_hat
    σ2_hat = dot(res, res) / NT_eff

    vcov_mat, λ_se = _sem_panel_vcov(X_filt, W, λ_hat, σ2_hat, N, T_per,
                                     N_eff, T_eff, fe_spec;
                                     max_exact_n=max_exact_n,
                                     n_probes=vcov_probes, rng=rng)
    full_coef = vcat(β_hat, λ_hat)

    return SpatialErrorPanelModel{FE, T, M}(
        fe_spec, N, T_per, N_eff, T_eff,
        full_coef, vcov_mat, λ_hat, λ_se, σ2_hat, max_ll,
        W, X_tilde, y_tilde, res, coefnames_vec
    )
end

function _sem_panel_vcov(
    X_filt::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    λ::T,
    σ2::T,
    N::Int,
    T_per::Int,
    N_eff::Int,
    T_eff::Int,
    fe_spec::FixedEffectsSpec;
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    P = size(X_filt, 2)
    NT_eff = N_eff * T_eff

    # As in the cross-section, all three traces came from 30 unseeded
    # probes. tr(G) used a second, independent probe set, and the time-FE
    # case mixed a projected estimate for two of the traces with a
    # closed-form subtraction for the third; g_traces projects G itself, so
    # all three are handled the same way.
    tr_G, tr_G2, tr_GtG = g_traces(W, λ;
                                   project_constant=has_time_fe(fe_spec),
                                   max_exact_n=max_exact_n,
                                   n_probes=n_probes, rng=rng)

    I_ββ = (X_filt' * X_filt) ./ σ2
    I_λλ = T_eff * (tr_G2 + tr_GtG)
    I_λσ2 = (T_eff * tr_G) / σ2
    I_σ2σ2 = NT_eff / (2 * σ2^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    Info[P + 1, P + 1] = I_λλ
    Info[P + 1, P + 2] = I_λσ2
    Info[P + 2, P + 1] = I_λσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]
    λ_se = sqrt(max(zero(T), V_coef[P + 1, P + 1]))

    return V_coef, λ_se
end

# ─── Heteroskedasticity-robust GM estimator (Kelejian & Prucha 2010) ──
#
# The model is y = Xβ + u, u = λWu + ε, with the ε independent and of
# unknown, possibly different, variances, and X exogenous. Kelejian &
# Prucha (2010, eq. 5) estimate λ from
#
#     E[ε′A₁ε] = 0,  A₁ = W′W − diag(W′W),        E[ε′A₂ε] = 0,  A₂ = W.
#
# Both matrices have a zero diagonal, so E[ε′Aε] = Σᵢ Aᵢᵢ σᵢ² = 0 whatever
# the σᵢ². The moments the maximum-likelihood fit rests on, and the
# Kelejian–Prucha (1998) moments in `SpatialGS2SLSPanel`, need equal σᵢ².
#
# The same code runs a panel: the rows are stacked N units at a time, W acts
# within each period, and with `fe = true` both stages work on two-way
# demeaned data. That is the package's extension, not the paper's: see the
# notes at `_gm_psi`.

# The bilinear forms behind the two moments, summed over periods:
# a′A₁b = (Wa)′(Wb) − Σᵢ dᵢaᵢbᵢ with dᵢ = (W′W)ᵢᵢ, and a′A₂b = a′Wb.
_gm_B1(a, Wa, b, Wb, dd) = dot(Wa, Wb) - sum(dd .* a .* b)
_gm_B2(a, Wb) = dot(a, Wb)

# g and G of Kelejian & Prucha's eq. (8): m(λ) = g − G[λ, λ²]′ is the sample
# moment n⁻¹ε(λ)′Aᵣε(λ) with ε(λ) = u − λWu, exactly, for every λ.
function _gm_moments(u::Vector{T}, W, N::Int, T_per::Int, dd::Vector{T}) where {T}
    n = T(length(u))
    ub = panel_spatial_lag(W, reshape(u, :, 1), N, T_per)[:, 1]
    ubb = panel_spatial_lag(W, reshape(ub, :, 1), N, T_per)[:, 1]
    g = [_gm_B1(u, ub, u, ub, dd), _gm_B2(u, ub)] ./ n
    G = [2 * _gm_B1(u, ub, ub, ubb, dd)   -_gm_B1(ub, ubb, ub, ubb, dd);
         (_gm_B2(u, ubb) + _gm_B2(ub, ub))  -_gm_B2(ub, ubb)] ./ n
    return g, G
end

_gm_m(g, G, λ) = g .- G * [λ, λ^2]

# Ψ, the covariance of √n times the moments, Kelejian & Prucha's eq. (13)
# with no first-stage term (the regressors are exogenous, so their aᵣ = 0):
#
#     ψᵣₛ = (2n)⁻¹ Σᵢⱼ Sᵣ,ᵢⱼ Sₛ,ᵢⱼ σᵢ² σⱼ²,   Sᵣ = Aᵣ + Aᵣ′.
#
# The zero diagonals are why no third or fourth moment enters. Over T
# periods the moment is Σ_{i<j} Sᵢⱼ (eᵢ·eⱼ), eᵢ unit i's T residuals, so
# σᵢ²σⱼ² becomes E[(eᵢ·eⱼ)²] = tr(CᵢCⱼ), Cᵢ the covariance of unit i's
# errors over time, estimated by (êᵢ·êⱼ)². That is robust to the serial
# correlation the unit demeaning induces, and at T = 1 it is the paper's
# diag(ε̂²) estimator exactly. Time demeaning correlates units at O(1/N),
# which this ignores.
function _gm_psi(e::Vector{T}, S1, S2, N::Int, T_per::Int) where {T}
    E = reshape(e, N, T_per)
    K(i, j) = dot(view(E, i, :), view(E, j, :))^2
    acc(S) = (s = zero(T); (I_, J_, V_) = findnz(S);
              for k in eachindex(V_); I_[k] == J_[k] && continue
                  s += V_[k] * K(I_[k], J_[k]); end; s)
    n2 = 2 * T(N * T_per)
    return [acc(S1 .* S1) acc(S1 .* S2); acc(S1 .* S2) acc(S2 .* S2)] ./ n2
end

function _gm_solve(g, G, Υ, bounds::Tuple{T, T}) where {T}
    f(λ) = (m = _gm_m(g, G, λ); dot(m, Υ * m))
    grid = range(bounds[1], bounds[2], length = 199)
    k = argmin(f.(grid))
    lo = grid[max(1, k - 1)]; hi = grid[min(length(grid), k + 1)]
    λ, _ = brent_maximize(x -> -f(x), T(lo), T(hi); tol = T(1e-10))
    return T(λ)
end

"""
    _sem_gmm_het(W, X, y, N, T_per; fe, bounds) -> NamedTuple

Kelejian & Prucha's (2010) two-step estimator for a spatial-error
regression with exogenous `X` and heteroskedasticity of unknown form:

1. OLS, then `λ` by GM, unweighted and then with the weight `Ψ̂⁻¹`;
2. OLS on the spatially filtered data (feasible GLS), then `λ` again from
   those residuals with the weight re-estimated (their steps 1a to 2b).

`β̂` comes from step 2a and `λ̂` from step 2b, as in the paper. With
exogenous regressors the covariance of `(β̂, λ̂)` is block diagonal
(their eq. 25 with `aᵣ = 0`). `β̂`'s block is the cluster-robust sandwich
of the filtered regression, by unit, in its jackknife (CR3) form (at
`T_per = 1`, HC3); `λ̂`'s variance is `(J′Ψ̂⁻¹J)⁻¹/n`, `J = G[1, 2λ̂]′`
(their eq. 16 with the efficient weight).

`W` must have a zero diagonal; rows are stacked `N` at a time. With
`fe = true` the data are two-way demeaned after filtering, and the filter is
applied to the levels, which keeps the period effects period effects when
`W` is row-standardised.
"""
function _sem_gmm_het(W::AbstractSparseMatrix{T}, X::AbstractMatrix{T}, y::AbstractVector{T},
                      N::Int, T_per::Int; fe::Bool = false,
                      bounds::Tuple{T, T} = (T(-0.995), T(0.995))) where {T<:AbstractFloat}
    n, k = size(X)
    n == N * T_per || throw(DimensionMismatch("X has $n rows; expected N*T = $(N * T_per)"))
    all(iszero, diag(W)) || throw(ArgumentError(
        "the heteroskedasticity-robust GM estimator needs `W` with a zero diagonal: its " *
        "moments are valid for any error variances only because the diagonal is zero"))
    dm(v::AbstractVector) = fe ? within_transform!(copy(v), N, T_per, TwoWayFE()) : copy(v)
    dmX(M) = fe ? within_transform(Matrix(M), N, T_per, TwoWayFE()) : Matrix(M)
    lagv(v) = panel_spatial_lag(W, reshape(v, :, 1), N, T_per)[:, 1]

    WtW = W' * W
    dd = repeat(Vector(diag(WtW)), T_per)
    A1 = WtW - spdiagm(0 => diag(WtW))
    S1 = A1 + A1'
    S2 = W + W'

    yd, Xd = dm(y), dmX(X)
    # Steps 1a-1c: OLS residuals, initial GM, efficient GM.
    ũ = yd - Xd * (Xd \ yd)
    g, G = _gm_moments(ũ, W, N, T_per, dd)
    λ1 = _gm_solve(g, G, Matrix{T}(I, 2, 2), bounds)
    Ψ1 = _gm_psi(ũ .- λ1 .* lagv(ũ), S1, S2, N, T_per)
    λ2 = _gm_solve(g, G, inv(Symmetric(Ψ1)), bounds)

    # Step 2a: feasible GLS. The filter acts on the levels, the transform after.
    ys = dm(y .- λ2 .* lagv(y))
    Xs = dmX(X .- λ2 .* panel_spatial_lag(W, Matrix(X), N, T_per))
    β = Xs \ ys
    ε = ys - Xs * β

    # Step 2b: GM again, on the FGLS residuals of the untransformed model.
    û = yd - Xd * β
    g2, G2 = _gm_moments(û, W, N, T_per, dd)
    Ψ2 = _gm_psi(û .- λ2 .* lagv(û), S1, S2, N, T_per)
    λ̂ = _gm_solve(g2, G2, inv(Symmetric(Ψ2)), bounds)
    Ψ̂ = _gm_psi(û .- λ̂ .* lagv(û), S1, S2, N, T_per)
    J = G2 * [one(T), 2λ̂]
    vλ = inv(dot(J, Symmetric(Ψ̂) \ J)) / n

    # β's covariance: the jackknife-type cluster sandwich (CR3), by unit,
    #   V = (G−1)/G · B Σᵢ Xᵢ′(I − Hᵢᵢ)⁻¹ eᵢeᵢ′(I − Hᵢᵢ)⁻¹Xᵢ B,  B = (Xs′Xs)⁻¹,
    # with Hᵢᵢ unit i's diagonal block of the hat matrix. At T = 1 it is HC3
    # times (N−1)/N. HC1 under-covered here when the treated sit in one
    # block (0.924 for `att_direct` against 0.941 for HC3, 1,000 draws):
    # their rows have high leverage, and HC1 does not adjust for it.
    bread = inv(Symmetric(Xs' * Xs))
    meat = zeros(T, k, k)
    XB = Xs * bread
    for i in 1:N
        rows = [i + (t - 1) * N for t in 1:T_per]
        Xi = Xs[rows, :]
        Hii = XB[rows, :] * Xi'
        ẽ = (Matrix{T}(I, T_per, T_per) - Hii) \ ε[rows]
        sc = Xi' * ẽ
        meat .+= sc * sc'
    end
    Vβ = Matrix(Symmetric(T(N - 1) / T(N) .* (bread * meat * bread)))

    return (beta = β, vcov_beta = Vβ, lambda = λ̂, lambda_se = sqrt(max(zero(T), vλ)),
            lambda_initial = λ2, residuals = ε, sigma2 = dot(ε, ε) / n, Psi = Ψ̂, J = J,
            X_filtered = Xs)
end

# ─── Show methods ─────────────────────────────────────────────────────

function Base.show(io::IO, m::SpatialErrorRegressionModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    gmm = isnan(m.loglik)
    println(io, gmm ? "Spatial Error Model (SEM) — GMM, robust to heteroskedasticity (Kelejian & Prucha 2010)" :
                      "Spatial Error Model (SEM) — Maximum Likelihood")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    if gmm
        @printf(io, "λ = %.4f    σ² (mean ε̂²) = %.4f    N = %d
", m.lambda, m.sigma2, length(m.y))
    else
        @printf(io, "λ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d\n", m.lambda, m.sigma2, m.loglik, length(m.y))
    end
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

function Base.show(io::IO, m::SpatialErrorPanelModel{FE}) where {FE}
    fname = FE === NoFE ? "None" : FE === IndivFE ? "Individual" : FE === TimeFE ? "Time" : "Two-Way"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial Error Panel Model (SEM) — Maximum Likelihood — FE: $fname")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "λ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d    T = %d\n", m.lambda, m.sigma2, m.loglik, m.N, m.T_periods)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── StatsAPI interface ───────────────────────────────────────────────

StatsAPI.coef(m::SpatialErrorRegressionModel) = m.coef
StatsAPI.vcov(m::SpatialErrorRegressionModel) = m.vcov
StatsAPI.stderror(m::SpatialErrorRegressionModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialErrorRegressionModel) = m.loglik
StatsAPI.nobs(m::SpatialErrorRegressionModel) = length(m.y)
StatsAPI.dof_residual(m::SpatialErrorRegressionModel) = length(m.y) - length(m.coef)
StatsAPI.residuals(m::SpatialErrorRegressionModel) = m.residuals
StatsAPI.coefnames(m::SpatialErrorRegressionModel) = m.coefnames

StatsAPI.coef(m::SpatialErrorPanelModel) = m.coef
StatsAPI.vcov(m::SpatialErrorPanelModel) = m.vcov
StatsAPI.stderror(m::SpatialErrorPanelModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialErrorPanelModel) = m.loglik
StatsAPI.nobs(m::SpatialErrorPanelModel) = m.N * m.T_periods
StatsAPI.dof_residual(m::SpatialErrorPanelModel) = (m.N_eff * m.T_eff) - length(m.coef)
StatsAPI.residuals(m::SpatialErrorPanelModel) = m.residuals
StatsAPI.coefnames(m::SpatialErrorPanelModel) = m.coefnames

end # module
