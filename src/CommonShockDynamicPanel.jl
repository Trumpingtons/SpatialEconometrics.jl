"""
    CommonShockDynamicPanel

Short-panel (fixed-`T`, large-`N`) dynamic spatial panel with **common
shocks**, by Bai & Chen's (2026) conditional Gaussian QML with spatially
enriched random loadings (SERL):

```math
y_t = ρWy_t + φy_{t-1} + X_tβ + 1_Nδ_t + Λf_t + ε_t,\\quad t = 1, …, T,
```

with `r` common factors `f_t` and loadings `λ_i = Az_i + η_i`. The controls
`z_i` stack observed regressor summaries `Q_X`, the initial outcome `y_0`, and
their spatial lags up to order `L`:
`z_i = [(Q_X)_i, (WQ_X)_i, …, (W^LQ_X)_i, (y_0)_i, …, (W^Ly_0)_i]`. No
unit-specific parameter is estimated, so `T` can stay fixed while `N` grows.
The idiosyncratic variances `σ²_t` may change over time.

The likelihood is conditional on `y_0`. It is correct only when
`E(λ_i | y_0, X, W) = Az_i` for the chosen `L` (Bai & Chen's Assumption 2).
If `y_0` comes from a long-running spatial process, a finite `L` is an
approximation; the paper's Table 4 shows `L = 0` is inconsistent in that
design, and measures `L = 1` only at `N ≤ 1000`.

The covariance is the sandwich `Ĥ⁻¹Ω̂Ĥ⁻¹`. `Ω̂` is built from the score
contributions rearranged into a martingale-difference array (Appendix D,
eq. 54), so it allows non-normal `η_i` and `ε_i`.

# References

Bai, J., & Chen, J. C. (2026). Fixed-T dynamic spatial panel model with
common shocks. arXiv:2608.22499v1 [econ.EM]. **Preprint.**
"""
module CommonShockDynamicPanel

using LinearAlgebra
using SparseArrays
using Statistics
using Printf

import StatsAPI
import StatsAPI: coef, vcov, stderror, loglikelihood, nobs, coefnames, residuals
import ForwardDiff

import ..SpatialUtils: check_design_rank, _print_coef_table

export CommonShockDynamicPanelModel, fit_common_shock_dynamic_panel

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

# Everything fixed across parameter values. Panels are N×T matrices, row i =
# unit i, column t = sample period t (1..T).
struct _CSData
    N::Int
    T::Int
    K::Int
    r::Int
    q::Int
    Y::Matrix{Float64}            # y_t, t = 1..T
    Y1::Matrix{Float64}           # y_{t−1}
    WY::Matrix{Float64}           # (W y_t)
    X::Vector{Matrix{Float64}}    # one N×T matrix per regressor
    Z::Matrix{Float64}            # N×q SERL controls, centred and scaled
    W::Matrix{Float64}            # dense, for the variance
    eig_re::Vector{Float64}       # eigenvalues of W, for log|B(ρ)|
    eig_im::Vector{Float64}
end

# Parameter layout: α = (ρ, φ, β, δ, vec A, vec F₂, vech Σ_η, σ²).
_np(d::_CSData) = 2 + d.K + d.T + d.r * d.q + (d.T - d.r) * d.r + d.r * (d.r + 1) ÷ 2 + d.T

function _unpack(d::_CSData, α::AbstractVector{S}) where {S}
    K, T, r, q = d.K, d.T, d.r, d.q
    ρ = α[1]; φ = α[2]; β = α[3:(2 + K)]
    i = 3 + K
    δ = α[i:(i + T - 1)]; i += T
    A = reshape(α[i:(i + r * q - 1)], r, q); i += r * q
    F2 = reshape(α[i:(i + (T - r) * r - 1)], T - r, r); i += (T - r) * r
    Ση = zeros(S, r, r)
    for l in 1:r, k in l:r
        Ση[k, l] = α[i]; Ση[l, k] = α[i]; i += 1
    end
    σ2 = α[i:(i + T - 1)]
    F = vcat(Matrix{S}(I, r, r), F2)
    return ρ, φ, β, δ, A, F, Ση, σ2
end

function _pack(ρ, φ, β, δ, A, F, Ση, σ2)
    r = size(F, 2)
    v = [Ση[k, l] for l in 1:r for k in l:r]
    return vcat(ρ, φ, β, δ, vec(A), vec(F[(r + 1):end, :]), v, σ2)
end

function _αnames(d::_CSData, vn, zn)
    r, T = d.r, d.T
    nm = vcat(["rho", "phi"], vn, ["delta_$t" for t in 1:T])
    append!(nm, ["A[$a,$(zn[b])]" for b in 1:d.q for a in 1:r])
    append!(nm, ["F[$t,$c]" for c in 1:r for t in (r + 1):T])
    append!(nm, ["Sigma_eta[$k,$l]" for l in 1:r for k in l:r])
    append!(nm, ["sigma2_$t" for t in 1:T])
    return nm
end

# ─────────────────────────────────────────────────────────────────────────────
# Likelihood and unit scores
# ─────────────────────────────────────────────────────────────────────────────

# log|det(I − ρW)| from the eigenvalues; a conjugate pair contributes
# log|1 − ρλ|² through real arithmetic, so ForwardDiff can differentiate it.
function _logdetB(d::_CSData, ρ)
    s = zero(ρ)
    @inbounds for j in eachindex(d.eig_re)
        a = d.eig_re[j]; b = d.eig_im[j]
        s += log((1 - ρ * a)^2 + (ρ * b)^2) / 2
    end
    return s
end
_dlogdetB(d, ρ) = ForwardDiff.derivative(x -> _logdetB(d, x), ρ)
_d2logdetB(d, ρ) = ForwardDiff.derivative(x -> _dlogdetB(d, x), ρ)

function _U(d::_CSData, ρ, φ, β)
    U = d.Y .- ρ .* d.WY .- φ .* d.Y1
    for k in 1:d.K
        U = U .- β[k] .* d.X[k]
    end
    return U
end

_E(d::_CSData, U, δ, A, F) = U .- δ' .- d.Z * (F * A)'

function _Sigma_u(F, Ση, σ2)
    Σ = F * Ση * F'
    for t in eachindex(σ2)
        Σ[t, t] += σ2[t]
    end
    return (Σ .+ Σ') ./ 2
end

# Valid parameter point: Σ_η positive definite, σ² > 0.
function _valid(d::_CSData, α)
    ρ, φ, β, δ, A, F, Ση, σ2 = _unpack(d, α)
    all(>(0), σ2) || return false
    isposdef(Symmetric(Matrix{Float64}(Ση))) || return false
    return true
end

# −(N/2)log|Σ_u| − ½ Σ_i e_i'Σ_u⁻¹e_i, eq. (17) without the Jacobian.
function _loglik_nojac(d::_CSData, α::AbstractVector{S}) where {S}
    ρ, φ, β, δ, A, F, Ση, σ2 = _unpack(d, α)
    E = _E(d, _U(d, ρ, φ, β), δ, A, F)
    Σu = _Sigma_u(F, Ση, σ2)
    C = cholesky(Symmetric(Σu))
    M = inv(C)
    return -d.N / 2 * logdet(C) - sum((E * M) .* E) / 2
end

_loglik(d::_CSData, α) = d.T * _logdetB(d, α[1]) + _loglik_nojac(d, α)

# The ordinary unit scores s_i, one row per unit, without the Jacobian term
# (−T tr(B⁻¹W)/N per unit). Columns follow the layout of α.
function _unit_scores(d::_CSData, α::AbstractVector{S}) where {S}
    N, T, K, r, q = d.N, d.T, d.K, d.r, d.q
    ρ, φ, β, δ, A, F, Ση, σ2 = _unpack(d, α)
    E = _E(d, _U(d, ρ, φ, β), δ, A, F)
    Σu = _Sigma_u(F, Ση, σ2)
    M = inv(cholesky(Symmetric(Σu)))
    M = (M .+ M') ./ 2
    ME = E * M                               # row i = (M e_i)'
    G = zeros(eltype(ME), N, _np(d))
    G[:, 1] = vec(sum(d.WY .* ME; dims = 2))
    G[:, 2] = vec(sum(d.Y1 .* ME; dims = 2))
    for k in 1:K
        G[:, 2 + k] = vec(sum(d.X[k] .* ME; dims = 2))
    end
    c = 3 + K
    G[:, c:(c + T - 1)] = ME; c += T
    FtMe = ME * F                            # row i = (F'M e_i)'
    for b in 1:q, a in 1:r
        G[:, c] = FtMe[:, a] .* d.Z[:, b]; c += 1
    end
    Coef = d.Z * A' .+ FtMe * Ση             # row i = (Az_i)' + (F'Me_i)'Σ_η
    MFS = M * F * Ση
    for cc in 1:r, t in (r + 1):T
        G[:, c] = ME[:, t] .* Coef[:, cc] .- MFS[t, cc]; c += 1
    end
    FMF = F' * M * F
    for l in 1:r, k in l:r
        v = FtMe[:, k] .* FtMe[:, l] .- FMF[k, l]
        G[:, c] = k == l ? v ./ 2 : v; c += 1
    end
    for t in 1:T
        G[:, c] = (ME[:, t] .^ 2 .- M[t, t]) ./ 2; c += 1
    end
    return G
end

# The full score ∇ℓ (sums over units), Jacobian term included.
function _score(d::_CSData, α)
    s = vec(sum(_unit_scores(d, α); dims = 1))
    s[1] += d.T * _dlogdetB(d, α[1])
    return s
end

# −∇²ℓ: the Jacobian of the analytic score plus the log-determinant's
# second derivative.
function _neg_hessian(d::_CSData, α)
    J = ForwardDiff.jacobian(a -> vec(sum(_unit_scores(d, a); dims = 1)), α)
    H = -J
    H[1, 1] -= d.T * _d2logdetB(d, α[1])
    return (H .+ H') ./ 2
end

# ─────────────────────────────────────────────────────────────────────────────
# Martingale rearrangement and the covariance (Appendix D)
# ─────────────────────────────────────────────────────────────────────────────

# With C(ρ, φ) = I_N⊗R_T(φ) − ρW⊗I_T and G = C⁻¹, block (t, s) of G is
# φ^(t−s) B⁻¹^(t−s+1) for t ≥ s. So the pair terms of the ρ score,
# P^ρ_ij = e_i'M𝒲_ij e_j with 𝒲_ij = Σ_k w_ik G_kj, are
#   P^ρ = Σ_{d=0}^{T−1} φ^d (W B⁻¹^(d+1)) ∘ Q_d,
# and those of the φ score, with ℒ_ij = L_T G_ij,
#   P^φ = Σ_{d=1}^{T−1} φ^(d−1) B⁻¹^d ∘ Q_d,
# where Q_d[i, j] = Σ_s (M e_i)_{s+d} e_{j,s}. No loop over pairs is needed.
function _pair_mats(d::_CSData, α)
    N, T = d.N, d.T
    ρ, φ, β, δ, A, F, Ση, σ2 = _unpack(d, α)
    E = _E(d, _U(d, ρ, φ, β), δ, A, F)
    M = inv(cholesky(Symmetric(_Sigma_u(F, Ση, σ2))))
    ME = E * ((M .+ M') ./ 2)
    Binv = inv(I - ρ .* d.W)
    Pρ = zeros(N, N); Pφ = zeros(N, N)
    Bk = Binv                                  # B⁻¹^(dd+1)
    Bprev = Matrix{Float64}(I, N, N)           # B⁻¹^dd
    WB1 = d.W * Binv
    for dd in 0:(T - 1)
        Q = ME[:, (dd + 1):T] * E[:, 1:(T - dd)]'
        Pρ .+= φ^dd .* (d.W * Bk) .* Q
        dd >= 1 && (Pφ .+= φ^(dd - 1) .* Bprev .* Q)
        if dd < T - 1
            Bprev = Bk
            Bk = Bk * Binv
        end
    end
    return Pρ, Pφ, WB1
end

# Lower-triangle row sums Σ_{j<i} R_ij.
function _lower_rowsum(R::AbstractMatrix)
    N = size(R, 1)
    s = zeros(N)
    @inbounds for j in 1:N, i in (j + 1):N
        s[i] += R[i, j]
    end
    return s
end

# g_i, g†_i and the pair component of Ω, as sums (not divided by N).
function _martingale(d::_CSData, α)
    Su = _unit_scores(d, α)
    Pρ, Pφ, WB1 = _pair_mats(d, α)
    Rρ = Pρ .+ Pρ'; Rφ = Pφ .+ Pφ'
    gd = copy(Su)
    gd[:, 1] .= Su[:, 1] .- vec(sum(Pρ; dims = 2)) .+ diag(Pρ) .- d.T .* diag(WB1)
    gd[:, 2] .= Su[:, 2] .- vec(sum(Pφ; dims = 2)) .+ diag(Pφ)
    g = copy(gd)
    g[:, 1] .+= _lower_rowsum(Rρ)
    g[:, 2] .+= _lower_rowsum(Rφ)
    half(A, B) = (sum(A .* B) - sum(diag(A) .* diag(B))) / 2
    Ωv = zeros(size(Su, 2), size(Su, 2))
    Ωv[1, 1] = half(Rρ, Rρ); Ωv[2, 2] = half(Rφ, Rφ)
    Ωv[1, 2] = Ωv[2, 1] = half(Rρ, Rφ)
    return (g = g, gdag = gd, Ωdag = gd' * gd, Ωv = Ωv, trWB2 = sum(WB1 .* WB1'))
end

# ─────────────────────────────────────────────────────────────────────────────
# Estimation: Algorithm 1 (block coordinate / ECME), then Newton
# ─────────────────────────────────────────────────────────────────────────────

mutable struct _State
    ρ::Float64; φ::Float64; β::Vector{Float64}; δ::Vector{Float64}
    A::Matrix{Float64}; F::Matrix{Float64}; Ση::Matrix{Float64}; σ2::Vector{Float64}
end
_α(s::_State) = _pack(s.ρ, s.φ, s.β, s.δ, s.A, s.F, s.Ση, s.σ2)

# Appendix C: per-period OLS at ρ = 0, then principal components.
function _init(d::_CSData)
    N, T, K, r = d.N, d.T, d.K, d.r
    ϑs = zeros(1 + K, T); δ0 = zeros(T)
    for t in 1:T
        R = hcat(ones(N), d.Y1[:, t], reduce(hcat, [d.X[k][:, t] for k in 1:K]; init = zeros(N, 0)))
        b = R \ d.Y[:, t]
        δ0[t] = b[1]; ϑs[:, t] = b[2:end]
    end
    ϑ = vec(mean(ϑs; dims = 2))
    φ0 = ϑ[1]; β0 = ϑ[2:end]
    Rm = _U(d, 0.0, φ0, β0) .- δ0'
    sv = svd(Rm)
    Fpc = sqrt(T) .* sv.V[:, 1:r]
    Λpc = sv.U[:, 1:r] .* (sv.S[1:r] ./ sqrt(T))'
    H = inv(Fpc[1:r, :])
    F0 = Fpc * H
    Λ0 = Λpc * inv(H)'
    Ση0 = Λ0' * Λ0 ./ N
    ε0 = Rm .- Λ0 * F0'
    σ20 = vec(sum(ε0 .^ 2; dims = 1)) ./ N
    return _State(0.0, φ0, β0, δ0, zeros(r, d.q), F0, (Ση0 .+ Ση0') ./ 2, max.(σ20, 1e-8))
end

# One inner sweep: Steps 1–5 of §4.1.
function _inner_sweep!(s::_State, d::_CSData, ZcInv)
    N, T, K, r = d.N, d.T, d.K, d.r
    U = _U(d, s.ρ, s.φ, s.β)
    # Step 1: concentrate out A and δ
    Σu = _Sigma_u(s.F, s.Ση, s.σ2); M = inv(cholesky(Symmetric(Σu)))
    ub = vec(mean(U; dims = 1)); zb = vec(mean(d.Z; dims = 1))
    Uc = U .- ub'; Zc = d.Z .- zb'
    FM = s.F' * M
    s.A = (FM * s.F) \ (FM * Uc' * Zc * ZcInv)
    s.δ = ub .- s.F * (s.A * zb)
    E = _E(d, U, s.δ, s.A, s.F)
    # Step 2: E-step for η_i
    Dinv = 1 ./ s.σ2
    V = inv(inv(Symmetric(s.Ση)) + s.F' * (Dinv .* s.F))
    V = (V .+ V') ./ 2
    Mm = E * (Dinv .* s.F) * V                  # row i = m_i'
    # Step 3: Σ_η and D_ε
    Ση = V .+ Mm' * Mm ./ N
    s.Ση = (Ση .+ Ση') ./ 2
    Rr = E .- Mm * s.F'
    for t in 1:T
        ft = s.F[t, :]
        s.σ2[t] = sum(abs2, view(Rr, :, t)) / N + dot(ft, V * ft)
    end
    # Step 4: F, then renormalise to F[1:r, :] = I
    Sm = d.Z * s.A' .+ Mm                       # row i = s_i'
    Ft = (U .- s.δ')' * Sm / (Sm' * Sm .+ N .* V)
    H = inv(Ft[1:r, :])
    Hi = inv(H)
    s.F = Ft * H; s.F[1:r, :] .= Matrix{Float64}(I, r, r)
    s.A = Hi * s.A
    Ση = Hi * s.Ση * Hi'
    s.Ση = (Ση .+ Ση') ./ 2
    # Step 5: GLS for (φ, β), centred to profile out δ
    Σu = _Sigma_u(s.F, s.Ση, s.σ2); M = inv(cholesky(Symmetric(Σu)))
    Yt = d.Y .- s.ρ .* d.WY
    target = Yt .- d.Z * (s.F * s.A)'
    target = target .- mean(target; dims = 1)
    regs = [d.Y1, d.X...]
    regs = [R .- mean(R; dims = 1) for R in regs]
    p = 1 + K
    G = zeros(p, p); h = zeros(p)
    for a in 1:p
        RM = regs[a] * M
        h[a] = sum(RM .* target)
        for b in a:p
            G[a, b] = G[b, a] = sum(RM .* regs[b])
        end
    end
    ϑ = G \ h
    s.φ = ϑ[1]; s.β = ϑ[2:end]
    return s
end

# Step 6: the conditional criterion in ρ is T log|B(ρ)| − ½(a − 2bρ + cρ²).
function _rho_step!(s::_State, d::_CSData, lo, hi)
    E0 = _E(d, _U(d, 0.0, s.φ, s.β), s.δ, s.A, s.F)
    M = inv(cholesky(Symmetric(_Sigma_u(s.F, s.Ση, s.σ2))))
    E0M = E0 * M; WYM = d.WY * M
    a = sum(E0M .* E0); b = sum(E0M .* d.WY); c = sum(WYM .* d.WY)
    f(ρ) = d.T * _logdetB(d, ρ) - (a - 2b * ρ + c * ρ^2) / 2
    s.ρ = _golden_max(f, lo, hi)
    return s
end

function _golden_max(f, a, b; tol = 1e-12, maxit = 200)
    g = (sqrt(5) - 1) / 2
    c = b - g * (b - a); dd = a + g * (b - a)
    fc = f(c); fd = f(dd)
    for _ in 1:maxit
        abs(b - a) < tol && break
        if fc > fd
            b = dd; dd = c; fd = fc; c = b - g * (b - a); fc = f(c)
        else
            a = c; c = dd; fc = fd; dd = a + g * (b - a); fd = f(dd)
        end
    end
    return (a + b) / 2
end

# Newton on the full α, with step halving on the log-likelihood.
function _newton!(α, d::_CSData, lo, hi; max_iter = 50, tol = 1e-8)
    ℓ = _loglik(d, α)
    for it in 1:max_iter
        g = _score(d, α)
        maximum(abs, g) / d.N < tol && return α, true, it - 1
        H = _neg_hessian(d, α)
        step = try
            cholesky(Symmetric(H)) \ g
        catch err
            err isa PosDefException || rethrow()
            # Not concave here: a damped step.
            μ = max(1e-6, -eigmin(Symmetric(H)) + 1e-3 * opnorm(H))
            (H + μ * I) \ g
        end
        t = 1.0; moved = false
        for _ in 1:40
            αn = α .+ t .* step
            if lo < αn[1] < hi && _valid(d, αn)
                ℓn = _loglik(d, αn)
                if isfinite(ℓn) && ℓn >= ℓ - 1e-12 * abs(ℓ)
                    α .= αn; ℓ = ℓn; moved = true
                    break
                end
            end
            t /= 2
        end
        moved || return α, maximum(abs, _score(d, α)) / d.N < tol, it
    end
    return α, maximum(abs, _score(d, α)) / d.N < tol, max_iter
end

function _estimate(d::_CSData, lo, hi; max_outer, max_inner, tol)
    s = _init(d)
    N = d.N
    ZcInv = let Zc = d.Z .- mean(d.Z; dims = 1); inv(Symmetric(Zc' * Zc)) end
    ℓ = -Inf
    outer = 0
    for o in 1:max_outer
        outer = o
        ℓin = -Inf
        for _ in 1:max_inner
            _inner_sweep!(s, d, ZcInv)
            ℓn = _loglik(d, _α(s))
            done = abs(ℓn - ℓin) / N < 1e-10
            ℓin = ℓn
            done && break
        end
        _rho_step!(s, d, lo, hi)
        ℓn = _loglik(d, _α(s))
        done = abs(ℓn - ℓ) / N < 1e-9
        ℓ = ℓn
        done && break
    end
    α = _α(s)
    α, ok, nit = _newton!(α, d, lo, hi; tol = tol)
    return α, ok, outer, nit
end

# ─────────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────────

function _standardise(C::AbstractMatrix, names)
    Z = Matrix{Float64}(C)
    for j in axes(Z, 2)
        sd = std(Z[:, j])
        sd > 0 || throw(ArgumentError("the SERL control $(names[j]) is constant across units"))
        Z[:, j] = (Z[:, j] .- mean(Z[:, j])) ./ sd
    end
    return Z
end

function _serl_controls(Q, Qnames, y0, W, L)
    blocks = Matrix{Float64}[]; nm = String[]
    P = Matrix{Float64}(Q)
    for l in 0:L
        push!(blocks, P); append!(nm, l == 0 ? Qnames : ["W$(l == 1 ? "" : l)_" * n for n in Qnames])
        P = W * P
    end
    v = Vector{Float64}(y0)
    for l in 0:L
        push!(blocks, reshape(v, :, 1)); push!(nm, l == 0 ? "y0" : "W$(l == 1 ? "" : l)_y0")
        v = W * v
    end
    return reduce(hcat, blocks), nm
end

# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

"""
    CommonShockDynamicPanelModel

A fit of the short-panel dynamic spatial panel with common shocks by
[`fit_common_shock_dynamic_panel`](@ref).

# Fields

- `coef`, `coefnames`: `θ = (ρ, φ, β)`.
- `vcov`: the `θ` block of the sandwich `Ĥ⁻¹Ω̂Ĥ⁻¹`, with the pair component
  of `Ω̂` scaled by `N/(N − p_α)` (the paper's preferred version, eq. 23).
  This block equals the profiled covariance of §3.4.
- `vcov_uncorrected`: the same without the `N/(N − p_α)` factor.
- `alpha`, `alphanames`, `vcov_alpha`: the full parameter vector
  `α = (ρ, φ, β, δ, vec A, vec F₂, vech Σ_η, σ²)` and its sandwich covariance.
- `delta`, `A`, `F`, `Sigma_eta`, `sigma2`: the nuisance estimates, with
  `F = [I_r; F₂]`. `A`'s columns follow `controlnames`; the controls are
  centred and scaled before estimation, so `A` is in standard-deviation
  units.
- `loglik`: the conditional quasi log-likelihood (17), without `2π` terms.
- `converged`, `outer_iterations`, `newton_iterations`, `score_norm`
  (the largest `|∂ℓ/∂α|/N` at the estimate).
- `N`, `T`, `r`, `L`.
"""
struct CommonShockDynamicPanelModel <: StatsAPI.StatisticalModel
    coef::Vector{Float64}
    coefnames::Vector{String}
    vcov::Matrix{Float64}
    vcov_uncorrected::Matrix{Float64}
    alpha::Vector{Float64}
    alphanames::Vector{String}
    vcov_alpha::Matrix{Float64}
    delta::Vector{Float64}
    A::Matrix{Float64}
    F::Matrix{Float64}
    Sigma_eta::Matrix{Float64}
    sigma2::Vector{Float64}
    controlnames::Vector{String}
    residuals::Matrix{Float64}
    loglik::Float64
    converged::Bool
    outer_iterations::Int
    newton_iterations::Int
    score_norm::Float64
    N::Int
    T::Int
    r::Int
    L::Int
end

StatsAPI.coef(m::CommonShockDynamicPanelModel) = m.coef
StatsAPI.coefnames(m::CommonShockDynamicPanelModel) = m.coefnames
StatsAPI.vcov(m::CommonShockDynamicPanelModel) = m.vcov
StatsAPI.stderror(m::CommonShockDynamicPanelModel) = sqrt.(max.(diag(m.vcov), 0))
StatsAPI.nobs(m::CommonShockDynamicPanelModel) = m.N * m.T
StatsAPI.loglikelihood(m::CommonShockDynamicPanelModel) = m.loglik
StatsAPI.residuals(m::CommonShockDynamicPanelModel) = m.residuals

function Base.show(io::IO, m::CommonShockDynamicPanelModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Short-T dynamic spatial panel with common shocks (Bai & Chen 2026)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "N = %d    T = %d    factors r = %d    SERL order L = %d\n", m.N, m.T, m.r, m.L)
    println(io, "SEs: sandwich, martingale score covariance, pair component × N/(N − p)")
    println(io, "The likelihood is conditional on y₀ and assumes E(λᵢ | y₀, X, W) = Azᵢ")
    println(io, "for this L; with y₀ from a long-running process, a finite L approximates.")
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─────────────────────────────────────────────────────────────────────────────
# The fit
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_common_shock_dynamic_panel(y, X, W, N, T_total; r, L = 1, Q = nothing,
                                   controls = nothing, eigenvalues = nothing,
                                   rho_bounds = (-0.995, 0.995),
                                   varnames = nothing, max_outer = 300,
                                   max_inner = 200, tol = 1e-8)

Bai & Chen's (2026) conditional QML for the short-panel dynamic spatial panel
with `r` common factors and spatially enriched random loadings (see
[`CommonShockDynamicPanel`](@ref) for the model).

# Data

`y` and `X` are stacked as in [`fit_dynamic_spatial_panel`](@ref SpatialEconometrics.DynamicSpatialPanelEconometrics.fit_dynamic_spatial_panel): `N` units
within each of `T_total = T + 1` periods, **period 0 first**. Period 0
supplies `y₀`; `X`'s period-0 rows are not used. `X` has no intercept: the
time effects `δ_t` take its place.

# Options

- `r` (required): the number of common factors. The theory takes it as
  known and gives no rule for choosing it.
- `L`: the SERL order, the highest spatial lag of `Q_X` and `y₀` among the
  controls. Bai & Chen's Table 4 finds `L = 0` badly under-covers when `y₀`
  comes from a long-running process, and `L = 1` near nominal at
  `N ≤ 1000`.
- `Q`: the `N × p` regressor summaries `Q_X`. The default is each unit's
  mean of `X` over periods `1..T`.
- `controls`: an `N × q` matrix that replaces the SERL controls `z_i`
  entirely; `L` and `Q` are then ignored.
- `eigenvalues`: the eigenvalues of `W`, if already known. Otherwise they are
  computed, at `O(N³)` cost. Pass them when fitting many panels on the same
  `W`.
- `rho_bounds`: the search interval for `ρ`; `I − ρW` must be nonsingular on
  it.
- `max_outer`, `max_inner`: the limits of the block-coordinate algorithm
  (§4); `tol`: the convergence criterion of the final Newton polish, on the
  largest `|∂ℓ/∂α|/N`.

# Algorithm

Appendix C's start (per-period OLS, then principal components), then
Algorithm 1: ECME and GLS updates given `ρ`, alternated with a scalar search
in `ρ`. Newton steps on the full parameter vector then solve the first-order
conditions to `tol`. If they are not solved, the fit throws.

# Variance

`Ĥ = −∇²ℓ(α̂)` by automatic differentiation of the analytic score. `Ω̂` is
eq. (54): the outer products of the one-unit parts of the rearranged
scores, plus the sum over all pairs of units of the pair terms' outer
products. The pair part is formed from `N × N` matrices, so the variance
costs `O(TN³)` time and about eight `N × N` matrices of memory.

# Returns

A [`CommonShockDynamicPanelModel`](@ref).
"""
function fit_common_shock_dynamic_panel(
    y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real}, N::Int, T_total::Int;
    r::Int, L::Int = 1,
    Q::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    controls::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    eigenvalues::Union{Nothing, AbstractVector{<:Number}} = nothing,
    rho_bounds::Tuple{<:Real, <:Real} = (-0.995, 0.995),
    varnames::Union{Nothing, Vector{String}} = nothing,
    controlnames::Union{Nothing, Vector{String}} = nothing,
    max_outer::Int = 300, max_inner::Int = 200, tol::Real = 1e-8,
)
    T = T_total - 1
    T >= 2 || throw(ArgumentError("need T_total ≥ 3 (period 0 plus at least two)"))
    1 <= r < T || throw(ArgumentError("need 1 ≤ r < T, got r = $r with T = $T"))
    L >= 0 || throw(ArgumentError("L must be ≥ 0, got $L"))
    length(y) == N * T_total || throw(DimensionMismatch("y has $(length(y)) rows, N*T_total = $(N * T_total)"))
    size(X, 1) == N * T_total || throw(DimensionMismatch("X has $(size(X, 1)) rows, N*T_total = $(N * T_total)"))
    size(W) == (N, N) || throw(DimensionMismatch("W is $(size(W)), expected ($N, $N)"))
    K = size(X, 2)
    vn = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    length(vn) == K || throw(ArgumentError("varnames has $(length(vn)) names for $K columns"))

    Yall = reshape(Vector{Float64}(y), N, T_total)
    Wd = Matrix{Float64}(W)
    Y = Yall[:, 2:end]; Y1 = Yall[:, 1:(end - 1)]
    WY = Wd * Y
    Xs = [reshape(Vector{Float64}(X[:, k]), N, T_total)[:, 2:end] for k in 1:K]

    Craw, cn = if isnothing(controls)
        Qm = isnothing(Q) ? reduce(hcat, [vec(mean(Xk; dims = 2)) for Xk in Xs]) : Matrix{Float64}(Q)
        size(Qm, 1) == N || throw(DimensionMismatch("Q has $(size(Qm, 1)) rows, expected $N"))
        Qn = isnothing(Q) ? ["mean_" * v for v in vn] : ["Q$j" for j in 1:size(Qm, 2)]
        _serl_controls(Qm, Qn, Yall[:, 1], Wd, L)
    else
        size(controls, 1) == N || throw(DimensionMismatch("controls has $(size(controls, 1)) rows, expected $N"))
        Matrix{Float64}(controls), ["z$j" for j in 1:size(controls, 2)]
    end
    isnothing(controlnames) || (cn = controlnames)
    Zs = _standardise(Craw, cn)
    check_design_rank(Zs, cn, "fit_common_shock_dynamic_panel";
                      context = "The SERL controls must be linearly independent.")

    ev = isnothing(eigenvalues) ? eigvals(Wd) : eigenvalues
    length(ev) == N || throw(DimensionMismatch("eigenvalues has $(length(ev)) entries, expected $N"))
    d = _CSData(N, T, K, r, size(Zs, 2), Y, Y1, WY, Xs, Zs, Wd,
                Float64.(real.(ev)), Float64.(imag.(ev)))
    lo, hi = Float64.(rho_bounds)

    α̂, ok, outer, nit = _estimate(d, lo, hi; max_outer = max_outer, max_inner = max_inner, tol = tol)
    sn = maximum(abs, _score(d, α̂)) / N
    ok || throw(ErrorException("fit_common_shock_dynamic_panel: the first-order conditions " *
        "were not solved (max |∂ℓ/∂α|/N = " * @sprintf("%.2e", sn) * "); no estimate is returned"))

    H = _neg_hessian(d, α̂)
    mg = _martingale(d, α̂)
    p = _np(d)
    Hi = inv(H)
    V0 = Hi * (mg.Ωdag .+ mg.Ωv) * Hi'
    V1 = Hi * (mg.Ωdag .+ (N / (N - p)) .* mg.Ωv) * Hi'
    V0 = (V0 .+ V0') ./ 2; V1 = (V1 .+ V1') ./ 2
    kθ = 2 + K
    ρ, φ, β, δ, A, F, Ση, σ2 = _unpack(d, α̂)
    E = _E(d, _U(d, ρ, φ, β), δ, A, F)
    return CommonShockDynamicPanelModel(α̂[1:kθ], vcat(["rho", "phi"], vn),
        V1[1:kθ, 1:kθ], V0[1:kθ, 1:kθ], α̂, _αnames(d, vn, cn), V1,
        δ, A, F, Ση, σ2, cn, E, _loglik(d, α̂), ok, outer, nit, sn, N, T, r, L)
end

end # module
