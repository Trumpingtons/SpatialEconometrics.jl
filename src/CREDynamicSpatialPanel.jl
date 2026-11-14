"""
    CREDynamicSpatialPanel

Short-panel (fixed-`T`) spatial dynamic panel with **correlated random
effects**, by Li & Yang's (2021) M-estimator:

```math
y_t = ρy_{t-1} + λ_1W_1y_t + λ_2W_2y_{t-1} + X_tβ + Zγ + μ + α_t1_n + u_t,\\quad
u_t = λ_3W_3u_t + v_t,\\quad μ = \\bar Xπ + ε,
```

for `t = 1, …, T`, with `X̄` the mean of `X_0, …, X_T`. The process started
`m` periods before the first observed period, so `y_0` is endogenous and the
conditional QMLE is inconsistent with `T` fixed. The M-estimator solves the
**adjusted quasi-score** (AQS): the conditional quasi-score with each biased
component corrected by its expectation (Lemma 2.1). The corrections depend on
the initial observations only through `m`.

This is the fixed-`T` counterpart of [`fit_dynamic_spatial_panel`](@ref SpatialEconometrics.DynamicSpatialPanelEconometrics.fit_dynamic_spatial_panel),
whose Yu, de Jong & Lee bias correction is a large-`T` device.

The covariance is a sandwich whose middle is a corrected plug-in estimate of
the variance of the adjusted score, adapted from Li, Meng & Yang (2026); it
allows non-normal, but not heteroskedastic, errors. It is not Li & Yang's
hybrid estimator.

# References

Li, L., & Yang, Z. (2021). Spatial dynamic panel data models with correlated
random effects. *Journal of Econometrics*, 221(2), 424–454.

Li, L., Meng, X., & Yang, Z. (2026). Dynamic spatial panel data models with
fixed effects: alternative approaches and application. *Spatial Economic
Analysis*, accepted.
"""
module CREDynamicSpatialPanel

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

import StatsAPI
import StatsAPI: coef, vcov, stderror, loglikelihood, nobs, coefnames, residuals
import ForwardDiff

import ..SpatialUtils: check_design_rank, _normcdf, _print_coef_table

export CREDynamicSpatialPanelModel, fit_cre_dynamic_spatial_panel

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

# Everything the likelihood and the AQS need, fixed across parameter values.
# Vectors of length nT are held as n×T matrices, column t = period t.
struct _CREData{T}
    n::Int
    Tp::Int
    Y::Matrix{T}          # y_1..y_T
    Y1::Matrix{T}         # y_0..y_{T-1}
    X::Matrix{T}          # nT × k design (1, D, X_t, 1⊗Z, 1⊗X̄)
    Xc::Vector{Matrix{T}} # the columns of X, each n×T
    W1::Matrix{T}
    W2::Union{Nothing, Matrix{T}}
    W3::Union{Nothing, Matrix{T}}
    W1Y::Matrix{T}
    W2Y1::Union{Nothing, Matrix{T}}
    m::Int
end

_k(d::_CREData) = size(d.X, 2)
# δ = (φ, ρ, λ1[, λ2][, λ3]); ψ = (β, σ², δ).
_nδ(d::_CREData) = 3 + !isnothing(d.W2) + !isnothing(d.W3)

function _unpack(d::_CREData, δ)
    φ, ρ, λ1 = δ[1], δ[2], δ[3]
    j = 4
    λ2 = zero(eltype(δ)); λ3 = zero(eltype(δ))
    if !isnothing(d.W2); λ2 = δ[j]; j += 1; end
    if !isnothing(d.W3); λ3 = δ[j]; end
    return φ, ρ, λ1, λ2, λ3
end

# ─────────────────────────────────────────────────────────────────────────────
# Matrices at a parameter value
# ─────────────────────────────────────────────────────────────────────────────

# Ω = φ(J_T⊗I) + I_T⊗(B3'B3)⁻¹, and with J̄ = J_T/T,
#     Ω⁻¹ = J̄⊗Aφ + (I − J̄)⊗Bm,  Aφ = (TφI + (B3'B3)⁻¹)⁻¹,  Bm = B3'B3,
#     log|Ω| = log|TφI + (B3'B3)⁻¹| − (T − 1) log|B3'B3|.
struct _Mats{S}
    B1::Matrix{S}
    B1inv::Matrix{S}
    B2::Matrix{S}
    Bm::Matrix{S}         # B3'B3
    Binv3::Matrix{S}      # (B3'B3)⁻¹
    Aφ::Matrix{S}
    logdetΩ::S
    logdetB1::S
end

function _mats(d::_CREData, δ)
    φ, ρ, λ1, λ2, λ3 = _unpack(d, δ)
    S = eltype(δ)
    n = d.n
    In = Matrix{S}(I, n, n)
    B1 = In .- λ1 .* d.W1
    F1 = lu(B1)
    B1inv = inv(F1)
    B2 = ρ .* In
    isnothing(d.W2) || (B2 = B2 .+ λ2 .* d.W2)
    if isnothing(d.W3)
        Bm = copy(In); Binv3 = copy(In)
    else
        B3 = In .- λ3 .* d.W3
        Bm = B3' * B3
        Binv3 = inv(cholesky(Symmetric(Bm)))
    end
    M = d.Tp * φ .* In .+ Binv3
    FM = cholesky(Symmetric(M))
    Aφ = inv(FM)
    ldB = isnothing(d.W3) ? zero(S) : logdet(cholesky(Symmetric(Bm)))
    logdetΩ = logdet(FM) - (d.Tp - 1) * ldB
    logdetB1 = first(logabsdet(F1))
    return _Mats{S}(B1, B1inv, B2, Bm, Binv3, Aφ, logdetΩ, logdetB1)
end

# Ω⁻¹ applied to an n×T matrix.
function _oinv(M::_Mats, E::AbstractMatrix)
    ē = mean(E; dims = 2)
    return M.Aφ * ē .+ M.Bm * (E .- ē)
end

_dot(A, B) = sum(A .* B)

# e(θ) = B1Y − B2Y₋₁ − Xβ, as n×T.
_resid(d::_CREData, M::_Mats, β) =
    M.B1 * d.Y .- M.B2 * d.Y1 .- reshape(d.X * β, d.n, d.Tp)

# ─────────────────────────────────────────────────────────────────────────────
# Conditional log-likelihood (2.4) and its score (2.5)
# ─────────────────────────────────────────────────────────────────────────────

function _loglik(d::_CREData, ψ)
    k = _k(d)
    β = ψ[1:k]; σ2 = ψ[k+1]; δ = ψ[k+2:end]
    M = _mats(d, δ)
    E = _resid(d, M, β)
    nT = d.n * d.Tp
    return -nT / 2 * log(2π * σ2) - M.logdetΩ / 2 + d.Tp * M.logdetB1 -
           _dot(E, _oinv(M, E)) / (2σ2)
end

# The λ3 derivative of (B3'B3)⁻¹: G = (B3'B3)⁻¹(B3'W3 + W3'B3)(B3'B3)⁻¹.
function _G3(d::_CREData, M::_Mats, λ3)
    B3 = I - λ3 .* d.W3
    return M.Binv3 * (B3' * d.W3 .+ d.W3' * B3) * M.Binv3
end

# The parts of the score shared by the CQS and the AQS: every component
# except the trace terms of ρ, λ1, λ2, which are returned separately as the
# random parts r (so CQS = r − cqs_trace, AQS = r − aqs_trace).
function _score_parts(d::_CREData, ψ)
    k = _k(d); n, Tp = d.n, d.Tp
    β = ψ[1:k]; σ2 = ψ[k+1]; δ = ψ[k+2:end]
    φ, ρ, λ1, λ2, λ3 = _unpack(d, δ)
    M = _mats(d, δ)
    E = _resid(d, M, β)
    F = _oinv(M, E)                                   # Ω⁻¹e
    S = eltype(ψ)
    nT = n * Tp
    sβ = [_dot(F, Xc) for Xc in d.Xc] ./ σ2
    sσ = _dot(E, F) / (2σ2^2) - nT / (2σ2)
    fs = vec(sum(F; dims = 2))                        # Σ_t (Ω⁻¹e)_t
    sφ = dot(fs, fs) / (2σ2) - Tp * tr(M.Aφ) / 2
    rρ = _dot(F, d.Y1) / σ2
    rλ1 = _dot(F, d.W1Y) / σ2
    out = vcat(sβ, [sσ, sφ, rρ, rλ1])
    if !isnothing(d.W2)
        push!(out, _dot(F, d.W2Y1) / σ2)
    end
    if !isnothing(d.W3)
        G = _G3(d, M, λ3)
        q = _dot(F, G * F)
        push!(out, q / (2σ2) - (_trp(M.Aφ, G) + (Tp - 1) * _trp(M.Bm, G)) / 2)
    end
    return out, M, φ
end

# Index of ρ in ψ, and of λ1, λ2.
_iρ(d::_CREData) = _k(d) + 3

function _cqs(d::_CREData, ψ)
    s, M, _ = _score_parts(d, ψ)
    s = copy(s)
    s[_iρ(d) + 1] -= d.Tp * sum(M.B1inv .* transpose(d.W1))   # λ1: −T tr(B1⁻¹W1)
    return s
end

# ─────────────────────────────────────────────────────────────────────────────
# The AQS corrections (Lemma 2.1), by block
# ─────────────────────────────────────────────────────────────────────────────

# C_t = (Σ_{i=0}^{t+m−1} 𝓑ⁱ)B1⁻¹ for t = 0..T, and D_k = 𝓑ᵏB1⁻¹(B3'B3)⁻¹
# for k = 0..T−1, with 𝓑 = B1⁻¹B2.
function _cd_blocks(M::_Mats{S}, Tp, m) where {S}
    n = size(M.B1, 1)
    𝓑 = M.B1inv * M.B2
    P = Matrix{S}(I, n, n)
    acc = zeros(S, n, n)                    # Σ_{i<j} 𝓑ⁱ
    sums = Matrix{S}[]                      # sums[j+1] = Σ_{i=0}^{j-1} 𝓑ⁱ
    push!(sums, copy(acc))
    pows = Matrix{S}[copy(P)]               # pows[k+1] = 𝓑ᵏ
    for j in 1:(Tp + m)
        acc = acc .+ P
        push!(sums, copy(acc))
        P = P * 𝓑
        j <= Tp - 1 && push!(pows, copy(P))
    end
    Cs = [sums[t + m + 1] * M.B1inv for t in 0:Tp]           # Cs[t+1] = C_t
    BB = M.B1inv * M.Binv3
    Ds = [pows[k + 1] * BB for k in 0:Tp-1]                  # Ds[k+1] = D_k
    return Cs, Ds
end

# tr[(φC + D)Ω⁻¹W] (lag = false) or tr[(φC₋₁ + D₋₁)Ω⁻¹W] (lag = true),
# from the blocks: Σ_s [Ω⁻¹]_{st} = Aφ, and an off-diagonal block of Ω⁻¹ is
# (Aφ − Bm)/T, a diagonal one (Aφ − Bm)/T + Bm. Verified against the full
# nT×nT traces in testset 42a.
function _corr_trace(M::_Mats, Cs, Ds, Wr, φ, Tp; lag::Bool)
    AW = M.Aφ * Wr
    DW = (M.Aφ .- M.Bm) * Wr
    s = zero(φ)
    for t in 1:Tp
        s += φ * tr(Cs[(lag ? t - 1 : t) + 1] * AW)
    end
    if lag
        for kk in 1:Tp-1                     # block lag t − s = kk uses D_{kk−1}
            s += (Tp - kk) / Tp * tr(Ds[kk] * DW)
        end
    else
        for kk in 0:Tp-1
            s += (Tp - kk) / Tp * tr(Ds[kk + 1] * DW)
        end
        s += Tp * tr(Ds[1] * M.Bm * Wr)
    end
    return s
end

# tr(AB) without forming AB.
_trp(A, B) = sum(A .* transpose(B))

# The same corrections as _corr_trace, without forming C_t or D_k. With
# P_i = 𝓑ⁱ and S_j = Σ_{i<j} P_i, C_t = S_{t+m}B1⁻¹ and D_k = P_kB1⁻¹(B3'B3)⁻¹,
# so every trace is a sum of tr(P_i K) for a few fixed K, each O(n²). The
# λ-free identity (B3'B3)⁻¹B3'B3 = I makes the extra diagonal term
# T·tr(D_0 Bm W) = T·tr(B1⁻¹W). Checked against _corr_trace in testset 42a.
function _aqs_traces(d::_CREData, M::_Mats{S}, φ) where {S}
    n, Tp, m = d.n, d.Tp, d.m
    𝓑 = M.B1inv * M.B2
    pows = Vector{Matrix{S}}(undef, Tp + m)          # pows[i+1] = 𝓑ⁱ
    pows[1] = Matrix{S}(I, n, n)
    for i in 2:(Tp + m); pows[i] = pows[i-1] * 𝓑; end
    AmB = M.Aφ .- M.Bm
    BB = M.B1inv * M.Binv3
    function one(Wr, lag)
        KC = M.B1inv * (M.Aφ * Wr)
        KD = BB * (AmB * Wr)
        τC = [_trp(P, KC) for P in pows]
        cum = vcat(zero(S), cumsum(τC))              # cum[j+1] = tr(S_j KC)
        s = zero(S)
        for t in 1:Tp
            s += φ * cum[(lag ? t - 1 : t) + m + 1]
        end
        if lag
            for kk in 1:Tp-1
                s += (Tp - kk) / Tp * _trp(pows[kk], KD)
            end
        else
            for kk in 0:Tp-1
                s += (Tp - kk) / Tp * _trp(pows[kk + 1], KD)
            end
            s += Tp * _trp(M.B1inv, Wr)
        end
        return s
    end
    In = Matrix{S}(I, n, n)
    t = [one(In, true), one(d.W1, false)]
    isnothing(d.W2) || push!(t, one(d.W2, true))
    return t
end

# The slow route, through the C and D blocks; kept as the reference.
function _aqs_traces_blocks(d::_CREData, M::_Mats, φ)
    Cs, Ds = _cd_blocks(M, d.Tp, d.m)
    In = Matrix{eltype(M.B1)}(I, d.n, d.n)
    t = [_corr_trace(M, Cs, Ds, In, φ, d.Tp; lag = true),
         _corr_trace(M, Cs, Ds, d.W1, φ, d.Tp; lag = false)]
    isnothing(d.W2) || push!(t, _corr_trace(M, Cs, Ds, d.W2, φ, d.Tp; lag = true))
    return t
end

# The AQS (2.11).
function _aqs(d::_CREData, ψ)
    s, M, φ = _score_parts(d, ψ)
    t = _aqs_traces(d, M, φ)
    i = _iρ(d)
    s = copy(s)
    s[i] -= t[1]
    s[i + 1] -= t[2]
    isnothing(d.W2) || (s[i + 2] -= t[3])
    return s
end

# ─────────────────────────────────────────────────────────────────────────────
# Concentration (2.12)–(2.13)
# ─────────────────────────────────────────────────────────────────────────────

function _concentrate(d::_CREData, δ)
    M = _mats(d, δ)
    R = M.B1 * d.Y .- M.B2 * d.Y1
    OX = reduce(hcat, [vec(_oinv(M, Xc)) for Xc in d.Xc])     # Ω⁻¹X, nT×k
    β = Symmetric(d.X' * OX) \ (OX' * vec(R))
    E = R .- reshape(d.X * β, d.n, d.Tp)
    σ2 = _dot(E, _oinv(M, E)) / (d.n * d.Tp)
    return β, σ2, M
end

_psi(d::_CREData, δ) = (c = _concentrate(d, δ); vcat(c[1], c[2], δ))

# Concentrated conditional log-likelihood in δ (for the CQMLE).
function _loglik_c(d::_CREData, δ)
    β, σ2, M = _concentrate(d, δ)
    nT = d.n * d.Tp
    return -nT / 2 * (log(2π * σ2) + 1) - M.logdetΩ / 2 + d.Tp * M.logdetB1
end

# The concentrated AQS (B.1): the δ-components of the AQS at (β̂(δ), σ̂²(δ)).
function _aqs_c(d::_CREData, δ)
    ψ = _psi(d, δ)
    return _aqs(d, ψ)[_k(d)+2:end]
end

# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

# Bounds on δ: φ > 0, |ρ| < 1, and each λ inside its range. A step that leaves
# them, or makes B1, B3'B3 or TφI + (B3'B3)⁻¹ singular, is halved.
function _inside(d::_CREData, δ)
    φ, ρ, λ1, λ2, λ3 = _unpack(d, δ)
    φ > 1e-8 && abs(ρ) < 1 && abs(λ1) < 1 && abs(λ2) < 1 && abs(λ3) < 1 || return false
    try
        _mats(d, δ); return true
    catch
        return false
    end
end

# Maximise the concentrated log-likelihood by damped Newton (ForwardDiff
# Hessian), falling back to gradient ascent when the Hessian is not negative
# definite.
function _cqmle(d::_CREData, δ0; max_iter = 200, tol = 1e-9)
    δ = copy(δ0)
    f(x) = _loglik_c(d, x)
    fδ = f(δ)
    for it in 1:max_iter
        g = ForwardDiff.gradient(f, δ)
        H = ForwardDiff.hessian(f, δ)
        step = try
            C = cholesky(Symmetric(-H))
            C \ g
        catch
            g ./ max(norm(g), 1.0)
        end
        t = 1.0
        accepted = false
        for _ in 1:40
            δn = δ .+ t .* step
            if _inside(d, δn)
                # As in _solve_aqs, a trial at a huge φ can make the GLS step
                # singular; it is rejected, not allowed to abort the fit.
                fn = try
                    f(δn)
                catch err
                    err isa Union{SingularException, PosDefException, DomainError} || rethrow()
                    -Inf
                end
                if isfinite(fn) && fn >= fδ - 1e-12 * abs(fδ)
                    δ, fδ, accepted = δn, fn, true
                    break
                end
            end
            t /= 2
        end
        accepted || return δ, false, it
        maximum(abs.(t .* step)) < tol && return δ, true, it
    end
    return δ, false, max_iter
end

# A consistent starting value. The AQS can have more than one root: in a
# simulated n = 225, T = 3 panel, Newton from the true δ found a root near
# it, and Newton from the conditional QMLE (φ̂ = 0 on its boundary, ρ̂ = 0.62
# against 0.3) found another at ρ = 0.56. Newton from a √n-consistent start
# converges to the consistent root with probability tending to one, so the
# start is 2SLS on first differences, which removes μ, the CRE part included:
#     Δy_t = ρΔy_{t−1} + λ1W1Δy_t + λ2W2Δy_{t−1} + ΔX_tβ + Δα_t + Δu_t,
# for t = 2..T, with instruments y_{t−2}, W1y_{t−2}, W1²y_{t−2}, W2y_{t−2},
# ΔX_t, W1ΔX_t, W1²ΔX_t, W1X_{t−1}, ΔX_{t−1}, W1ΔX_{t−1}, W2ΔX_{t−1},
# W1W2ΔX_{t−1} and period dummies (Xall holds X_0..X_T). ΔX_{t−1} and
# W2ΔX_{t−1} instrument Δy_{t−1} and W2Δy_{t−1}; without them, at T = 3 in Li &
# Yang's Table 1 design, 30% of the starts had |ρ₀ − ρ| > 0.2 and Newton
# failed from 4 in 20. Δu_t involves u_t and
# u_{t−1} only, so y_{t−2} stays valid with a spatial error. From the
# residuals at these values, λ3 starts at the maximiser of the within
# residuals' spatial-error likelihood and φ at the between/within variance
# split. The conditional likelihood is not used: it is biased for φ (mean
# start 1.6 against a true 1 in Li & Yang's Table 1 design).
function _iv_start(d::_CREData, Xall::Vector{Matrix{Float64}}, time_effects::Bool)
    n, Tp = d.n, d.Tp
    Yall = hcat(d.Y1[:, 1], d.Y)                      # y_0..y_T
    rows_y = Float64[]; Xr = Vector{Vector{Float64}}(); Zr = Vector{Vector{Float64}}()
    has2 = !isnothing(d.W2)
    for t in 2:Tp                                     # periods, y index t+1
        yt, yt1, yt2 = Yall[:, t+1], Yall[:, t], Yall[:, t-1]
        Δy = yt .- yt1; Δy1 = yt1 .- yt2
        ΔX = Xall[t+1] .- Xall[t]                     # X_t − X_{t−1}
        ΔX1 = Xall[t] .- Xall[t-1]                    # X_{t−1} − X_{t−2}
        append!(rows_y, Δy)
        dum = time_effects ? [Float64(s == t) for s in 2:Tp] : Float64[]
        R = hcat(Δy1, d.W1 * (yt .- yt1), has2 ? d.W2 * Δy1 : zeros(n, 0), ΔX,
                 time_effects ? repeat(dum', n) : zeros(n, 0))
        Q = hcat(yt2, d.W1 * yt2, d.W1 * (d.W1 * yt2), has2 ? d.W2 * yt2 : zeros(n, 0),
                 ΔX, d.W1 * ΔX, d.W1 * (d.W1 * ΔX), d.W1 * Xall[t],
                 ΔX1, d.W1 * ΔX1, has2 ? hcat(d.W2 * ΔX1, d.W1 * (d.W2 * ΔX1)) : zeros(n, 0),
                 time_effects ? repeat(dum', n) : zeros(n, 0))
        for i in 1:n
            push!(Xr, R[i, :]); push!(Zr, Q[i, :])
        end
    end
    Xm = reduce(vcat, permutedims.(Xr)); Qm = reduce(vcat, permutedims.(Zr))
    Fq = qr(Qm)
    Xh = Qm * (Fq \ Xm)                               # P_Q X
    b = (Xh' * Xm) \ (Xh' * rows_y)
    clampλ(x) = clamp(x, -0.9, 0.9)
    ρ0 = clampλ(b[1]); λ10 = clampλ(b[2]); λ20 = has2 ? clampλ(b[3]) : 0.0
    lag = vcat(ρ0, λ10, has2 ? [λ20] : Float64[])
    # Residuals at the IV values, with φ = 1 and no spatial error.
    _, _, M = _concentrate(d, vcat(1.0, lag, isnothing(d.W3) ? Float64[] : [0.0]))
    R = M.B1 * d.Y .- M.B2 * d.Y1
    β = d.X \ vec(R)
    E = R .- reshape(d.X * β, n, Tp)
    ē = vec(mean(E; dims = 2))
    Wt = E .- ē                                       # within: B3⁻¹(v_t − v̄)
    # λ3: the within residuals' spatial-error likelihood, by golden section.
    λ30 = 0.0
    if !isnothing(d.W3)
        In = Matrix{Float64}(I, n, n)
        function fλ(λ)
            B3 = In .- λ .* d.W3
            ld, sg = logabsdet(B3)
            return -n * (Tp - 1) / 2 * log(sum(abs2, B3 * Wt)) + (Tp - 1) * ld
        end
        λ30 = _golden(fλ, -0.9, 0.9)
    end
    B3 = isnothing(d.W3) ? Matrix{Float64}(I, n, n) : I - λ30 .* d.W3
    # φ from the split of the variance: E‖B3 w‖² = n(T−1)σv², and
    # b = B3ē = B3ε + v̄ has E b_i² = σε² Σ_j B3_ij² + σv²/T.
    σv2 = sum(abs2, B3 * Wt) / (n * (Tp - 1))
    b = B3 * ē
    σε2 = (mean(b .^ 2) - σv2 / Tp) / mean(sum(B3 .^ 2; dims = 2))
    φ0 = clamp(σε2 / σv2, 0.05, 20.0)
    return vcat(φ0, lag, isnothing(d.W3) ? Float64[] : [λ30])
end

# Golden-section maximiser of a unimodal f on [a, b].
function _golden(f, a, b; iters = 40)
    g = (sqrt(5) - 1) / 2
    c1 = b - g * (b - a); c2 = a + g * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in 1:iters
        if f1 >= f2
            b, c2, f2 = c2, c1, f1
            c1 = b - g * (b - a); f1 = f(c1)
        else
            a, c1, f1 = c1, c2, f2
            c2 = a + g * (b - a); f2 = f(c2)
        end
    end
    return (a + b) / 2
end

# Finite-difference Jacobians. ForwardDiff's dual numbers cannot use BLAS, so
# through dense n×n algebra its Jacobian cost about 25 evaluations of the
# function (measured at n = 225); differences on the Float64 code cost
# 2·dim. Forward differences serve the root solver, where the Jacobian's accuracy does
# not affect the root; central differences serve Σ̂, and testset 42 checks
# them against ForwardDiff.
function _jac_fwd(f, x, fx)
    J = zeros(length(fx), length(x))
    for j in eachindex(x)
        h = 1e-6 * max(1.0, abs(x[j]))
        xp = copy(x); xp[j] += h
        J[:, j] = (f(xp) .- fx) ./ h
    end
    return J
end

function _jac_central(f, x)
    f0 = f(x)
    J = zeros(length(f0), length(x))
    for j in eachindex(x)
        h = 1e-5 * max(1.0, abs(x[j]))
        xp = copy(x); xm = copy(x); xp[j] += h; xm[j] -= h
        J[:, j] = (f(xp) .- f(xm)) ./ (2h)
    end
    return J
end

# Solve the concentrated AQS by Levenberg–Marquardt on ‖S‖², in
# x = (log φ, ρ, λ…). The φ equation is solved as φ·S_φ, the score in log φ.
# S_φ itself → 0 as φ → ∞ (Ω⁻¹ loses its between part and S_φ ≈ −n/(2φ)),
# so φ = ∞ is a false asymptotic root that draws a solver in: on ‖S‖,
# Levenberg–Marquardt stopped at φ ≈ 1e9 to 1e23 in 7 of 60 samples of Li &
# Yang's Table 1 design, and Newton drifted towards it (φ → 15 to 31, λ1 → −1)
# in 3. φ·S_φ → −n/2 there, and the finite roots are unchanged. With it,
# Levenberg–Marquardt from the IV start reached, in all 60 samples, the root
# that Newton from the true δ reaches, in 8 to 13 iterations.
function _solve_aqs(d::_CREData, δ0; max_iter = 200, tol = 1e-10)
    nT = d.n * d.Tp
    toδ(x) = vcat(exp(x[1]), x[2:end])
    g(x) = (v = _aqs_c(d, toδ(x)); v[1] *= exp(x[1]); v ./ nT)
    x = vcat(log(δ0[1]), δ0[2:end])
    s = g(x); r = norm(s); μ = 1e-3
    for it in 1:max_iter
        r < tol && return toδ(x), true, it - 1, r
        J = _jac_fwd(g, x, s)
        A = J' * J; gr = J' * s
        accepted = false
        for _ in 1:30
            Δ = try
                -((A + μ * Diagonal(diag(A) .+ 1e-12)) \ gr)
            catch
                μ *= 4; continue
            end
            xn = x .+ Δ
            if _inside(d, toδ(xn))
                # A trial step can reach a δ where the GLS step for β is
                # singular: at a huge φ, Ω⁻¹ loses its between part and the
                # time-invariant columns (Z, X̄) drop out. In the Table 1
                # Monte Carlo that aborted 3 of 4500 fits. Such a trial is
                # rejected like any other step that does not reduce ‖S‖.
                sn = try
                    g(xn)
                catch err
                    err isa Union{SingularException, PosDefException, DomainError} || rethrow()
                    nothing
                end
                if !isnothing(sn) && all(isfinite, sn) && norm(sn) < r
                    x, s, r = xn, sn, norm(sn)
                    μ = max(μ / 3, 1e-12); accepted = true
                    break
                end
            end
            μ *= 4
        end
        accepted || return toδ(x), r < tol, it, r
    end
    return toδ(x), r < tol, max_iter, r
end

# ─────────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────────

_dense(W) = Matrix{Float64}(W)

function _cre_data(y, X, Z, W1, W2, W3, N, T_total, m, time_effects, vnames, znames)
    Tp = T_total - 1
    n = N
    p = size(X, 2)
    Yall = reshape(Vector{Float64}(y), n, T_total)
    Xall = Matrix{Float64}(X)
    Xt = [Xall[(t-1)*n+1:t*n, :] for t in 1:T_total]          # X_0..X_T
    Xbar = sum(Xt) ./ T_total                                  # mean over 0..T
    Xpost = reduce(vcat, Xt[2:end])                            # X_1..X_T
    cols = Vector{Vector{Float64}}([ones(n * Tp)])
    names = ["(intercept)"]
    if time_effects
        for t in 1:Tp-1
            dcol = zeros(n * Tp); dcol[(t-1)*n+1:t*n] .= 1
            push!(cols, dcol); push!(names, "time_$t")
        end
    end
    for j in 1:p
        push!(cols, Xpost[:, j]); push!(names, vnames[j])
    end
    if !isnothing(Z)
        for j in 1:size(Z, 2)
            push!(cols, repeat(Vector{Float64}(Z[:, j]), Tp)); push!(names, znames[j])
        end
    end
    for j in 1:p
        push!(cols, repeat(Xbar[:, j], Tp)); push!(names, "mean_" * vnames[j])
    end
    Xd = reduce(hcat, cols)
    check_design_rank(Xd, names, "fit_cre_dynamic_spatial_panel";
        context = "The design is (1, time dummies, X_t, Z, X̄). A regressor that " *
                  "does not vary over time duplicates its own mean X̄; put it in `Z` " *
                  "instead. `Z` must not carry an intercept.")
    Y = Yall[:, 2:end]; Y1 = Yall[:, 1:end-1]
    W1d = _dense(W1)
    W2d = isnothing(W2) ? nothing : _dense(W2)
    W3d = isnothing(W3) ? nothing : _dense(W3)
    d = _CREData{Float64}(n, Tp, Y, Y1, Xd, [reshape(Xd[:, j], n, Tp) for j in axes(Xd, 2)],
        W1d, W2d, W3d, W1d * Y, isnothing(W2d) ? nothing : W2d * Y1, m)
    return d, names, Xt
end

function _δnames(d::_CREData)
    nm = ["phi", "rho", "lambda1"]
    isnothing(d.W2) || push!(nm, "lambda2")
    isnothing(d.W3) || push!(nm, "lambda3")
    return nm
end

# ─────────────────────────────────────────────────────────────────────────────
# Variance: the corrected plug-in estimator of Γ = Var(S*(ψ₀))
# ─────────────────────────────────────────────────────────────────────────────
#
# This follows the corrected plug-in route of Li, Meng & Yang (2026) for the
# fixed-effects sibling of this model, adapted to correlated random effects.
# Write y₀ = η + C₀ε, with C₀ = (Σ_{i<m} 𝓑ⁱ)B1⁻¹. Then η collects y_{−m}, the
# pre-sample shocks and the deterministic terms, so it is independent of the
# in-sample shocks u = (ε; v_1; …; v_T). Given η, at ψ₀, each AQS component is
# u'A u + a(η)'u + c, with A and c free of η and a(η) affine in η. So
# E[S*|η] does not depend on η, and
#     Γ = E_η Var(S*|η),
#     Var(S*|η)_ij = 2Σ_kl A_i,kl A_j,kl s_k s_l + Σ_k A_i,kk A_j,kk (μ4_k − 3s_k²)
#                    + Σ_k (A_i,kk a_j,k + A_j,kk a_i,k) μ3_k + a_i's∘a_j,
# with s, μ3, μ4 the variances and moments of the shocks (those of ε for the
# first n, of v for the rest). The η in a(η) is unobserved; its plug-in uses
# û = (ε̂; V̂), with ε̂ = E[ε|e] = φΣ_t(Ω⁻¹e)_t and V̂_t = B3(e_t − ε̂). Then
# û − u = −G(ε − ε̂), G = [I; −1_T⊗B3], with E[ε − ε̂ | η] = 0 and
# Var(ε − ε̂) = σ²φ(B3'B3)⁻¹Aφ. So â_i = a_i + F_i(ε − ε̂) with F_i the linear
# map, and the plug-in of a_i's∘a_j has bias tr(F_i' diag(s) F_j Var(ε − ε̂)),
# which is subtracted. The estimator is unbiased for Γ at the true parameters
# and moments; testset 42 checks that against the brute-force Var(S*).
#
# The shock maps are dense nT × n(T+1); the quadratic forms are dense
# n(T+1) × n(T+1), one for each of σ², φ, ρ and the λs, plus one scaled copy
# at a time, and four nT × n(T+1) maps: counted, about 10.6·8·(n(T+1))²
# bytes with all three λs, or 1.4 GB at n = 1000, T = 3, before temporaries.

# Y = c_Y + L_Y u, Y₋₁ = c_Y₋₁ + L_Y₋₁ u, e = L_e u, given η.
function _shock_maps(d::_CREData, M::_Mats, B3)
    n, Tp, m = d.n, d.Tp, d.m
    ns = n * (Tp + 1)
    In = Matrix{Float64}(I, n, n)
    𝓑 = M.B1inv * M.B2
    sums = Matrix{Float64}[zeros(n, n)]                # sums[j+1] = Σ_{i<j} 𝓑ⁱ
    P = copy(In)
    for _ in 1:(Tp + m)
        push!(sums, sums[end] .+ P)
        P = P * 𝓑
    end
    C(t) = sums[t + m + 1] * M.B1inv                    # C_t, t = 0..T
    B3inv = inv(B3)
    K = Matrix{Float64}[]                              # K[k+1] = 𝓑ᵏB1⁻¹B3⁻¹
    Q = M.B1inv * B3inv
    for _ in 0:Tp-1
        push!(K, Q); Q = 𝓑 * Q
    end
    rb(t) = (t - 1) * n + 1:t * n
    vb(s) = s * n + 1:(s + 1) * n
    LY = zeros(n * Tp, ns); LY1 = zeros(n * Tp, ns); Le = zeros(n * Tp, ns)
    for t in 1:Tp
        LY[rb(t), 1:n] = C(t)
        LY1[rb(t), 1:n] = C(t - 1)
        for s in 1:t;     LY[rb(t), vb(s)] = K[t - s + 1];      end
        for s in 1:t-1;   LY1[rb(t), vb(s)] = K[t - 1 - s + 1]; end
        Le[rb(t), 1:n] = In
        Le[rb(t), vb(t)] = B3inv
    end
    return LY, LY1, Le
end

# (I_T ⊗ W) applied to an nT × q matrix.
function _blk(W, A, n, Tp)
    out = similar(A)
    for t in 1:Tp
        r = (t - 1) * n + 1:t * n
        out[r, :] = W * A[r, :]
    end
    return out
end

_B3(d::_CREData, λ3) = isnothing(d.W3) ? Matrix{Float64}(I, d.n, d.n) :
                                         Matrix{Float64}(I, d.n, d.n) .- λ3 .* d.W3

"""
    _shock_moments(d, ψ)

Moment estimates `(μ3ε, μ4ε, μ3v, μ4v)` from the residuals at `ψ`.

- `v`: from the within part `w_t = B3(e_t − ē) = v_t − v̄`. With the weights
  `c = e_t − 1/T`, `E w³ = μ3v Σc³` and `E w⁴ = μ4v Σc⁴ + 3σ⁴((Σc²)² − Σc⁴)`.
  For `T = 2`, `Σc³ = 0`, and `μ3v` comes from `E[b w²] = μ3v (T−1)/T²`.
- `ε`: from the between part `b = B3ē = B3ε + v̄`, with row sums
  `R_q = Σ_j B3_ij^q`: `E b³ = μ3ε R3 + μ3v/T²`, and
  `E b⁴ = μ4ε R4 + 3σε⁴(R2² − R4) + 6σε²σv²R2/T + μ4v/T³ + 3σv⁴(T−1)/T³`.

The residuals are at the estimate, so each moment carries an `O(1/n)` bias
from the estimation of `β`; it is not corrected.
"""
function _shock_moments(d::_CREData, ψ)
    k = _k(d); Tp = d.Tp
    β = ψ[1:k]; σ2 = ψ[k+1]; δ = ψ[k+2:end]
    φ, _, _, _, λ3 = _unpack(d, δ)
    M = _mats(d, δ)
    E = _resid(d, M, β)
    B3 = _B3(d, λ3)
    ē = vec(mean(E; dims = 2))
    w = B3 * (E .- ē)
    b = B3 * ē
    T = Tp
    c2 = (T - 1) / T; c3 = (T - 1) * (T - 2) / T^2; c4 = (T - 1) * ((T - 1)^3 + 1) / T^4
    σv2 = σ2; σε2 = φ * σ2
    μ3v = T >= 3 ? mean(w .^ 3) / c3 : mean(b .* w .^ 2) * T^2 / (T - 1)
    μ4v = (mean(w .^ 4) - 3σv2^2 * (c2^2 - c4)) / c4
    R2 = vec(sum(B3 .^ 2; dims = 2)); R3 = vec(sum(B3 .^ 3; dims = 2)); R4 = vec(sum(B3 .^ 4; dims = 2))
    μ3ε = (mean(b .^ 3) - μ3v / T^2) / mean(R3)
    rest = mean(3σε2^2 .* (R2 .^ 2 .- R4) .+ 6σε2 * σv2 / T .* R2) +
           μ4v / T^3 + 3σv2^2 * (T - 1) / T^3
    μ4ε = (mean(b .^ 4) - rest) / mean(R4)
    return (μ3ε = μ3ε, μ4ε = μ4ε, μ3v = μ3v, μ4v = μ4v)
end

"""
    _plugin_gamma(d, ψ, mom)

The corrected plug-in estimate of `Var(S*(ψ₀))`, unscaled, at `ψ` with the
shock moments `mom` (see the comment block above).
"""
function _plugin_gamma(d::_CREData, ψ, mom; correct::Bool = true)
    k = _k(d); n, Tp = d.n, d.Tp; nT = n * Tp
    β = ψ[1:k]; σ2 = ψ[k+1]; δ = ψ[k+2:end]
    φ, ρ, λ1, λ2, λ3 = _unpack(d, δ)
    M = _mats(d, δ)
    B3 = _B3(d, λ3)
    LY, LY1, Le = _shock_maps(d, M, B3)
    ns = size(Le, 2)
    Jb = fill(1 / Tp, Tp, Tp)
    Oi = kron(Jb, M.Aφ) .+ kron(Matrix{Float64}(I, Tp, Tp) .- Jb, M.Bm)
    OiLe = Oi * Le                                     # nT × ns
    # Plug-in shocks and the deterministic parts c_Y, c_Y₋₁.
    E = _resid(d, M, β)
    εh = φ .* vec(sum(_oinv(M, E); dims = 2))
    uh = vcat(εh, vec(B3 * (E .- εh)))
    cY = vec(d.Y) .- LY * uh
    cY1 = vec(d.Y1) .- LY1 * uh
    G = vcat(Matrix{Float64}(I, n, n), repeat(-B3, Tp)) # ns × n
    Vδ = σ2 * φ .* (M.Binv3 * M.Aφ)
    s2 = vcat(fill(φ * σ2, n), fill(σ2, nT))
    μ3 = vcat(fill(mom.μ3ε, n), fill(mom.μ3v, nT))
    μ4 = vcat(fill(mom.μ4ε, n), fill(mom.μ4v, nT))
    sym(A) = (A .+ A') ./ 2
    # Each component as (A, a, F): quadratic matrix, linear vector, and the
    # map of ε − ε̂ into the plug-in error of a. `nothing` marks an absent part.
    forms = Tuple{Union{Nothing, Matrix{Float64}}, Union{Nothing, Vector{Float64}},
                  Union{Nothing, Matrix{Float64}}}[]
    for j in 1:k
        push!(forms, (nothing, OiLe' * d.X[:, j] ./ σ2, nothing))
    end
    push!(forms, (sym(Le' * OiLe) ./ (2σ2^2), nothing, nothing))           # σ²
    JO = repeat(sum(reshape(OiLe, n, Tp, ns); dims = 2)[:, 1, :], Tp)    # (J_T⊗I)Ω⁻¹L_e
    push!(forms, (sym(OiLe' * JO) ./ (2σ2), nothing, nothing))            # φ
    function bilin(W, L, c)
        WL = isnothing(W) ? L : _blk(W, L, n, Tp)
        Wc = isnothing(W) ? c : vec(W * reshape(c, n, Tp))
        LG = L * G
        WLG = isnothing(W) ? LG : _blk(W, LG, n, Tp)
        return (sym(OiLe' * WL) ./ σ2, OiLe' * Wc ./ σ2, OiLe' * WLG ./ σ2)
    end
    push!(forms, bilin(nothing, LY1, cY1))                                 # ρ
    push!(forms, bilin(d.W1, LY, cY))                                      # λ1
    isnothing(d.W2) || push!(forms, bilin(d.W2, LY1, cY1))                 # λ2
    if !isnothing(d.W3)                                                    # λ3
        G3 = _G3(d, M, λ3)
        push!(forms, (sym(OiLe' * _blk(G3, OiLe, n, Tp)) ./ (2σ2), nothing, nothing))
    end
    p = length(forms)
    dA = [isnothing(f[1]) ? nothing : diag(f[1]) for f in forms]
    κ = μ4 .- 3 .* s2 .^ 2
    Γ = zeros(p, p)
    for i in 1:p
        Asi = isnothing(forms[i][1]) ? nothing : forms[i][1] .* s2 .* s2'
        for j in i:p
        Ai, ai, Fi = forms[i]; Aj, aj, Fj = forms[j]
        v = 0.0
        if !isnothing(Ai) && !isnothing(Aj)
            v += 2 * _dot(Asi, Aj) + sum(dA[i] .* dA[j] .* κ)
        end
        !isnothing(Ai) && !isnothing(aj) && (v += sum(dA[i] .* aj .* μ3))
        !isnothing(Aj) && !isnothing(ai) && (v += sum(dA[j] .* ai .* μ3))
        !isnothing(ai) && !isnothing(aj) && (v += sum(ai .* s2 .* aj))
        correct && !isnothing(Fi) && !isnothing(Fj) && (v -= _trp(Fi' * (s2 .* Fj), Vδ))
        Γ[i, j] = Γ[j, i] = v
        end
    end
    return Γ
end

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

"""
    CREDynamicSpatialPanelModel

A fit of the short-panel CRE spatial dynamic panel by
[`fit_cre_dynamic_spatial_panel`](@ref).

# Fields

- `coef`, `coefnames`: `ψ = (β, σ²_v, φ, ρ, λ₁[, λ₂][, λ₃])`, where `β`
  stacks the intercept, the time dummies, the coefficients on `X_t`, on `Z`
  and on `X̄` (Mundlak's `π`), and `φ = σ²_ε/σ²_v`.
- `vcov`: the sandwich `Ĵ⁻¹Γ̂Ĵ⁻¹'`, with `Ĵ = −∂S*/∂ψ'` and `Γ̂` the corrected
  plug-in estimate of `Var(S*)`. It is valid for non-normal errors (it uses
  the estimated third and fourth moments of `ε` and `v`), not for
  heteroskedastic ones. The construction adapts the corrected plug-in
  estimator of Li, Meng & Yang (2026) for the fixed-effects model; it is
  **not** Li & Yang's (2021) hybrid estimator, whose details are in a
  supplement this package does not hold.
- `vcov_hessian`: `Ĵ⁻¹` from the AQS Jacobian alone. **Not robust**: it is
  `Var(ψ̂)` only when the AQS is an information-equal score, which it is not
  when `m > 0`. Kept for comparison.
- `moments`: the estimated `(μ3ε, μ4ε, μ3v, μ4v)` used by `vcov`.
- `cqmle`: the conditional QMLE of `ψ`, which is inconsistent with `T`
  fixed when `y₀` is endogenous. Kept because it is the starting value and
  the comparison the paper reports.
- `m`: the assumed process start, `m` periods before period 0.
- `residuals`: the composite errors `e = ε ⊗ 1 + B₃⁻¹v`, stacked like `y`.
- `loglik`: the conditional quasi log-likelihood (2.4) **at the M-estimate**,
  which does not maximise it.
- `converged`, `iterations`, `score_norm`: the root solver's report.
- `n`, `T`, `time_effects`.
"""
struct CREDynamicSpatialPanelModel <: StatsAPI.StatisticalModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    vcov::Union{Nothing, Matrix{Float64}}
    vcov_hessian::Matrix{Float64}
    cqmle::Vector{Float64}
    start::Vector{Float64}
    m::Int
    moments::NamedTuple{(:μ3ε, :μ4ε, :μ3v, :μ4v), NTuple{4, Float64}}
    residuals::Vector{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
    score_norm::Float64
    n::Int
    T::Int
    time_effects::Bool
    has_lambda2::Bool
    has_lambda3::Bool
end

StatsAPI.coef(r::CREDynamicSpatialPanelModel) = r.coef
StatsAPI.coefnames(r::CREDynamicSpatialPanelModel) = r.coefnames
function StatsAPI.vcov(r::CREDynamicSpatialPanelModel)
    isnothing(r.vcov) && throw(ArgumentError("the robust covariance is not available"))
    return r.vcov
end
StatsAPI.stderror(r::CREDynamicSpatialPanelModel) = sqrt.(max.(diag(vcov(r)), 0))
StatsAPI.nobs(r::CREDynamicSpatialPanelModel) = r.n * r.T
StatsAPI.loglikelihood(r::CREDynamicSpatialPanelModel) = r.loglik
StatsAPI.residuals(r::CREDynamicSpatialPanelModel) = r.residuals

function Base.show(io::IO, r::CREDynamicSpatialPanelModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Short-T spatial dynamic panel, correlated random effects (Li & Yang 2021)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    keep = [i for (i, nm) in enumerate(r.coefnames) if !startswith(nm, "time_")]
    se = isnothing(r.vcov) ? sqrt.(max.(diag(r.vcov_hessian), 0)) : stderror(r)
    _print_coef_table(io, r.coefnames[keep], r.coef[keep], se[keep])
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "n = %d    T = %d    process start m = %d periods before period 0\n",
            r.n, r.T, r.m)
    r.time_effects && println(io, "time effects: yes (dummies suppressed above)")
    println(io, isnothing(r.vcov) ? "SEs: Hessian only, NOT robust" :
                                    "SEs: sandwich, corrected plug-in Var(S*) (non-normal errors)")
    println(io, "The M-estimator depends on y₀ only through m; Li & Yang report it is")
    println(io, "robust to a wrong m, but it is an assumption, not an estimate.")
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─────────────────────────────────────────────────────────────────────────────
# The fit
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_cre_dynamic_spatial_panel(y, X, Z, W1, N, T_total; m,
                                  W2 = W1, W3 = W1, time_effects = true,
                                  varnames = nothing, zvarnames = nothing,
                                  max_iter = 100, tol = 1e-10)

Li & Yang's (2021) M-estimator for the short-panel spatial dynamic panel with
correlated random effects (see [`CREDynamicSpatialPanel`](@ref) for the model).

# Data

`y` and `X` are stacked as in [`fit_dynamic_spatial_panel`](@ref SpatialEconometrics.DynamicSpatialPanelEconometrics.fit_dynamic_spatial_panel): `N` units
within each of `T_total = T + 1` periods, **period 0 first**. Period 0
supplies `y₀` and enters the Mundlak mean `X̄`; estimation uses periods
`1..T`. `Z` is an `N×q` matrix of time-invariant regressors **without an
intercept**, or `nothing`. A regressor in `X` that does not vary over time
duplicates its own mean and is refused; put it in `Z`.

# Options

- `m` (required): the process started `m ≥ 0` periods before period 0.
  `m = 0` means `y₀` is exogenous, and then the M-estimator is the
  conditional QMLE. There is no default, because the estimator depends on
  it.
- `W2`, `W3`: the space-time lag and spatial error weights; `nothing` drops
  `λ₂` or `λ₃` together with its estimating equation.
- `time_effects`: include `α_t` as `T − 1` period dummies.
- `compute_cqmle`: also compute the conditional QMLE, kept in `cqmle` (`NaN`s
  if its optimiser fails). It is inconsistent when `m > 0`; it is there for
  comparison.
- `max_iter`, `tol`: the root solver's limits, with `tol` on `‖S*‖/(nT)`.

# Solving

The adjusted score can have more than one root, so the solver starts from a
consistent estimate (2SLS on first differences) rather than from the
conditional QMLE. It is Levenberg–Marquardt in `(log φ, ρ, λ)`, with the `φ`
equation taken in `log φ` because `φ = ∞` is a false root of the raw `φ`
score. If no root is reached, the solver is run once more from the same
start with `φ = 1`; if that fails too, the fit throws.

# Returns

A [`CREDynamicSpatialPanelModel`](@ref). `vcov` is the sandwich with the
corrected plug-in `Var(S*)`; `vcov_hessian` is the inverse Jacobian alone,
which is not a valid covariance when `m > 0`.
"""
function fit_cre_dynamic_spatial_panel(
    y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
    Z::Union{Nothing, AbstractMatrix{<:Real}},
    W1::AbstractMatrix{<:Real}, N::Int, T_total::Int;
    m::Int,
    W2::Union{Nothing, AbstractMatrix{<:Real}} = W1,
    W3::Union{Nothing, AbstractMatrix{<:Real}} = W1,
    time_effects::Bool = true,
    varnames::Union{Nothing, Vector{String}} = nothing,
    zvarnames::Union{Nothing, Vector{String}} = nothing,
    max_iter::Int = 100, tol::Real = 1e-10,
    compute_cqmle::Bool = false,
)
    T_total >= 3 || throw(ArgumentError("need T_total ≥ 3 (period 0 plus at least two)"))
    m >= 0 || throw(ArgumentError("m must be ≥ 0, got $m"))
    length(y) == N * T_total || throw(DimensionMismatch("y has $(length(y)) rows, N*T_total = $(N * T_total)"))
    size(X, 1) == N * T_total || throw(DimensionMismatch("X has $(size(X, 1)) rows, N*T_total = $(N * T_total)"))
    for (nm, W) in (("W1", W1), ("W2", W2), ("W3", W3))
        isnothing(W) && continue
        size(W) == (N, N) || throw(DimensionMismatch("$nm is $(size(W)), expected ($N, $N)"))
    end
    isnothing(Z) || size(Z, 1) == N || throw(DimensionMismatch("Z has $(size(Z, 1)) rows, expected $N"))
    p = size(X, 2)
    vn = isnothing(varnames) ? ["x$i" for i in 1:p] : varnames
    zn = isnothing(Z) ? String[] : (isnothing(zvarnames) ? ["z$i" for i in 1:size(Z, 2)] : zvarnames)
    length(vn) == p || throw(ArgumentError("varnames has $(length(vn)) names for $p columns"))

    d, bnames, Xall = _cre_data(y, X, Z, W1, W2, W3, N, T_total, m, time_effects, vn, zn)
    nT = d.n * d.Tp
    δ0 = _iv_start(d, Xall, time_effects)
    δ̂, ok, iters, rn = _solve_aqs(d, δ0; max_iter = max_iter, tol = tol)
    # A poor IV start for φ can still send the solver towards φ = ∞. In the
    # Table 1 Monte Carlo, 4 of 4500 fits failed that way; three had φ₀ of 3.6
    # to 3.8. In the one examined the root existed: started at the truth, the
    # solver reached it, and from φ = 1 it reaches the same root.
    # So a failed solve is retried once, from the same start with φ = 1.
    if !ok && δ0[1] != 1
        δ1 = copy(δ0); δ1[1] = 1.0
        δ̂, ok, it2, rn = _solve_aqs(d, δ1; max_iter = max_iter, tol = tol)
        iters += it2
    end
    ok || throw(ErrorException("fit_cre_dynamic_spatial_panel: the adjusted quasi-score " *
        "was not solved (‖S*/nT‖ = " * @sprintf("%.2e", rn) * " after $iters iterations); " *
        "no estimate is returned"))
    ψ̂ = _psi(d, δ̂)
    ψc = if compute_cqmle
        # The CQMLE is a comparison only: if it fails, it is NaN, and the
        # M-estimate is still returned.
        okc, δc = try
            δc, okc, _ = _cqmle(d, δ0)
            okc, δc
        catch err
            err isa Union{SingularException, PosDefException, DomainError} || rethrow()
            false, δ0
        end
        okc ? _psi(d, δc) : fill(NaN, length(ψ̂))
    else
        Float64[]
    end
    J = _jac_central(ψ -> _aqs(d, ψ), ψ̂)
    Ji = inv(-J)
    Vh = (Ji .+ Ji') ./ 2
    mom = _shock_moments(d, ψ̂)
    Γ = _plugin_gamma(d, ψ̂, mom)
    V = Ji * Γ * Ji'
    V = (V .+ V') ./ 2
    k = _k(d)
    M = _mats(d, δ̂)
    E = _resid(d, M, ψ̂[1:k])
    names = vcat(bnames, ["sigma2_v"], _δnames(d))
    return CREDynamicSpatialPanelModel(ψ̂, names, V, Vh, ψc, δ0, m, mom, vec(E),
        _loglik(d, ψ̂), ok, iters, rn, d.n, d.Tp, time_effects,
        !isnothing(d.W2), !isnothing(d.W3))
end

end # module
