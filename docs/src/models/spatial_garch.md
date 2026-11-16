# Spatial log-GARCH

Every other model in the package puts spatial dependence in the conditional
**mean**. [`fit_spatial_log_garch`](@ref) puts it in the conditional
**variance**: a unit's volatility depends on its neighbours' volatility and
on their squared outcomes. It implements the two-step estimator of Sato &
Matsuda (2021).

## The model

For a cross-section of mean-zero data `y` (returns, or the residuals of a
mean model):

```math
y_i = \sqrt{h_i}\,ε_i, \qquad
\log h = λW\log h + ρW\log y^2 + α1_n + Zδ,
```

with `ε_i` i.i.d., `E ε_i² = 1`, and one row-standardised `W` in both places.
`Z` holds optional exogenous variables. Write `Y = log y²`,
`V = log ε² − E log ε²` and `R(x) = I − xW`. Then the model is a spatial
ARMA(1,1) in `Y`:

```math
R(λ + ρ)\,Y = Xβ + R(λ)\,V, \qquad X = [1_n, Z],\quad β = (α + (1 − λ)c,\ δ),
```

where `c = E log ε²`. A stationary solution needs `|λ + ρ| < 1`. Rook
contiguity, as in Sato & Matsuda's simulations:

```julia
W = make_rook_weights(20)                    # 400 units
m = fit_spatial_log_garch(y, W; Z = z)
m.lambda, m.rho, m.alpha, stderror(m)
m.log_volatility                             # fitted log ĥ
```

## What separates λ from ρ

The mean of `Y` depends on `λ` and `ρ` only through `λ + ρ`. `λ` enters on
its own only through the moving-average error `R(λ)V`. So `λ + ρ` is
estimated well and `λ` poorly, and their estimates are strongly negatively
correlated. With an intercept and no `Z`, the error part is all that
identifies `λ`.

Sato & Matsuda assume identification (their Assumption 3.8) and do not
state this. The Monte Carlo below measures what it costs.

## No interior maximum near λ = 1

The quasi log-likelihood has no upper bound as `λ → 1`, in every sample:
- `W` is row-standardised, so `W1 = 1` and `R(λ)1 = (1 − λ)1`;
- `R(λ)⁻¹` inflates only the part of `R(λ + ρ)Y − Xβ` along the eigenvalue
  1, and the intercept in `β` can set that part to zero, so `V` and `σ̂²`
  stay bounded;
- meanwhile `−log|R(λ)|` grows like `−log(1 − λ)`.

The estimate is therefore an interior local maximum, not the global one.
The test suite checks the divergence value against value: the concentrated
log-likelihood rises by `log 10` for each decade of `1 − λ`, and `σ̂²`
settles.

When a sample has no interior maximum, the search runs to the edge of
`bounds` and the fit throws an error that says so. It does not return the
edge point. With rook `W`, `(λ, ρ) = (0.9, 0.05)` and `n = 100`, this
happens in 36–39% of samples, and not at all at `n = 400`. Sato & Matsuda
do not mention it, and their Table 1 reports all 1000 samples.

## Estimation

**Step 1** is Gaussian quasi maximum likelihood for the SARMA form (Sato &
Matsuda eqs 7–9). `β` and `σ² = Var V` are profiled out, which leaves a
two-parameter search in `θ = (λ, λ + ρ)`:
- a 21 × 21 grid inside `bounds` (default `(−0.99, 0.99)` for both);
- then Newton steps on the analytic score and Hessian, with step halving,
  until `max|∂ℓ/∂θ|/n < tol` (default `1e-8`).

If that tolerance is not reached, the fit throws. It never returns a point
that does not solve the first-order conditions.

`log|R(x)|` and the traces in the score come from the eigenvalues of `W`.
They are computed once per fit, at `O(n³)` cost, or passed in with
`eigenvalues` when many fits share one `W`. A fit takes about 0.1 s at
`n = 100`, 0.3 s at 400 and 1.7 s at 1600.

**Step 2** recovers the volatility intercept from `E ε² = 1` (eq. 11):

```math
\hat α = (1 − \hat λ)\log\Big(n^{-1}\sum_i \exp C_i\Big), \qquad
C = R(\hat λ)^{-1}\big(R(\hat λ + \hat ρ)Y − Z\hat δ\big).
```

Sato & Matsuda prove `α̂` consistent (Theorem 3.3) but give it no
distribution, so it has **no standard error** here.

## Standard errors

`log ε²` is far from normal: with normal `ε` it is `log χ²₁`, with skewness
−2.1 and excess kurtosis 4. The Gaussian information `−Ĥ⁻¹` is therefore
wrong, and Sato & Matsuda's Theorem 3.2 gives a sandwich instead. Its middle
term, the score covariance `Ω`, is not written out in the paper or its
supplement, so it is derived here.

At the true parameters every score component is a linear-quadratic form in
the i.i.d. `V`:

```math
S_k = a_k'V + V'A_kV − σ^2\operatorname{tr}A_k .
```

For example, `S_λ = −V'GV/σ² + tr G` with `G = R(λ)⁻¹W`. Such forms have an
exact covariance in `σ²` and the third and fourth moments `μ₃`, `μ₄` of `V`:

```math
\operatorname{Cov}(S_i, S_j) = 2σ^4 \operatorname{tr}(A_iA_j)
 + (μ_4 − 3σ^4)\sum_k A_{i,kk}A_{j,kk}
 + μ_3\sum_k (A_{i,kk}a_{j,k} + A_{j,kk}a_{i,k}) + σ^2 a_i'a_j .
```

The reported `vcov` is `Ĥ⁻¹Ω̂Ĥ⁻¹`, where:
- `Ĥ` is the analytic Hessian;
- `Ω̂` is the formula above at the estimates, with `σ̂²`, `μ̂₃` and `μ̂₄`
  taken from the step-1 residuals.

`vcov_gaussian` keeps `−Ĥ⁻¹` for comparison. It is right only when `log ε²`
is normal, which happens when `ε` is log-normal.

The moments are printed by `show`, so the departure from normality is
visible in every fit.

## How it is checked

The test suite (testset 44) compares values with values:
- the quasi log-likelihood with a dense computation, and the eigenvalue
  log-determinants and traces with `logabsdet` and `tr`;
- the analytic score and Hessian with automatic differentiation, at
  arbitrary parameters;
- the linear-quadratic forms with the score at the true parameters, and the
  moment formula with a brute-force sum over all index quadruples;
- **`Var(S(ψ₀))` against the exact formula**, by simulation, under three
  error laws:
  - normal `ε`;
  - `ε² ~ Gamma(2)/2`;
  - log-normal `ε`.

  The design is held fixed, and the largest `|z|` over the 15 entries is
  1.9 on Julia 1.13 and 1.7 on 1.10. `E[Ω̂(ψ₀)]` with sample moments is
  tested the same way (largest `|z|` 0.6 and 1.2). The Gaussian information
  fails the same test with `|z|` = 30 and 29 when `ε` is normal, which
  shows the test has power;
- `α̂` and `log ĥ` recomputed densely, the sandwich rebuilt from its parts,
  and invariance to the order of the units.

## Monte Carlo

`validation/mc_spatial_log_garch.jl` runs three designs. Its output is
`validation/mc_spatial_log_garch_output.txt`. A fit that throws is counted;
the summaries are over the fits that return. `ε` is scaled so that
`E ε² = 1`:
- normal;
- `χ²₃/√15`;
- log-normal `exp(Z − 1)`.

**A. Sato & Matsuda's Table 1 design.** Rook `W` with `n = 100` and 400,
`α = 0.5`, `δ = 1`, one fixed `z`, three `(λ, ρ)`, the three laws, 1000
samples per cell.

At `n = 400` no fit fails, and bias and RMSE agree with their Table 1. With
normal `ε`, for example:

| (λ, ρ) | par | bias | RMSE | S&M bias | S&M RMSE |
|---|---|---|---|---|---|
| (.45, .45) | λ | −.012 | .097 | −.015 | .098 |
| (.45, .45) | ρ | −.000 | .073 | .002 | .072 |
| (.9, .05) | λ | .009 | .030 | .007 | .029 |
| (.05, .9) | λ | −.014 | .087 | −.011 | .080 |

Coverage of nominal 95% sandwich intervals at `n = 400`, the range over the
three laws (Sato & Matsuda report none):

| (λ, ρ) | δ | λ | ρ |
|---|---|---|---|
| (.9, .05) | .945–.947 | .878–.886 | .869–.885 |
| (.45, .45) | .940–.951 | .940–.948 | .933–.948 |
| (.05, .9) | .943–.954 | .943–.949 | .942–.959 |

What the cells show:
- **The SEs have the right scale.** Mean SE / sd is 0.95–1.04 in every
  cell at `n = 400`.
- **At `(0.9, 0.05)`, the intervals for `λ` and `ρ` cover 87–89%.** The SE
  scale is right there too, and the bias is 0.3–0.45 of an sd. With that bias
  and SE scale, a normal sampling distribution would give .92–.94 coverage,
  not the .87–.89 observed. The shortfall
  comes from the shape of the sampling distribution, not from the variance.
  This is the design closest to the edge where the likelihood is unbounded.
- **At `n = 100` with `(0.9, 0.05)`, 355, 368 and 390 of 1000 fits throw**
  (normal, `χ²₃`, log-normal), every one at the edge of `bounds`. The fits
  that return have a `λ` RMSE of .050, against Sato & Matsuda's .082 over
  all samples. At `(0.45, 0.45)` and `n = 100`, one fit in 1000 throws
  under each law, also at the edge.
- **The sandwich matters at `(0.9, 0.05)`.** There the Gaussian intervals
  for `λ` and `ρ` cover .79–.88, against .91–.96 for the sandwich at
  `n = 100`, and .84–.87 against .87–.89 at `n = 400`. In every other cell of
  A and B the two coverages differ by at most .02, in either direction, even
  though the score covariances differ widely (testset 44d). Why the gap
  stays this small for `δ`, `λ` and `ρ` was not measured.
- **At `n = 100` with `(0.05, 0.9)`, the RMSEs of `λ̂` and `ρ̂` are larger
  than Sato & Matsuda's**: .187 and .155 against .139 and .108 with normal
  `ε`. By `n = 400` they agree to within .007. The paper does not state its
  search region; the difference is not explained.
- **`α̂` errs most when `λ` is small and `ε` heavy-tailed**, as Sato & Matsuda
  find. With log-normal `ε` and `(0.05, 0.9)`, its bias is −1.09 at
  `n = 100` and −0.28 at `n = 400` (theirs: −1.01 and −0.31). With
  `(0.9, 0.05)` it is at most .02 in absolute value.

**B. Intercept only** (no `Z`, Fülle & Otto's case), `(0.45, 0.45)`, normal
`ε`, 500 samples:

| n | sd λ̂ | sd ρ̂ | sd(λ̂ + ρ̂) | corr(λ̂, ρ̂) | bias λ̂ | cov λ | cov ρ |
|---|---|---|---|---|---|---|---|
| 100 | .260 | .149 | .172 | −.78 | −.141 | .912 | .914 |
| 400 | .125 | .082 | .055 | −.94 | −.040 | .952 | .938 |
| 1600 | .061 | .040 | .026 | −.95 | −.016 | .950 | .958 |

The error part does identify `λ`. Its sd halves when `n` is multiplied by
4, and coverage is right from `n = 400`. But at `n = 1600` the sd of `λ̂`
is 2.3 times that of `λ̂ + ρ̂`. At `n = 400` its sd is .125 here, against .096
with the covariate (design A). At `n = 100`, `λ̂` is biased by a third of
its true value.

**C. A proxy `W`.**
- The data come from a 10-nearest-neighbour `W` on uniform locations `S`.
- The fit uses `covariate_weights((1 − ω)S + ωE, 10)`, with `E` standard
  normal.
- `(λ, ρ) = (0.45, 0.45)`, normal `ε`, 200 samples per cell.

"Shared" is the share of true neighbours that the proxy keeps. At `ω = 0`
it is not exactly 1, because standardising the columns changes the
distances slightly.

| n | ω | shared | bias ρ̂ | cov ρ | bias λ̂ | bias δ̂ | cov δ |
|---|---|---|---|---|---|---|---|
| 400 | 0 | .98 | .013 | .945 | −.035 | .011 | .960 |
| 400 | .01 | .88 | .004 | .895 | −.024 | −.005 | .960 |
| 400 | .03 | .64 | −.014 | .905 | −.033 | .026 | .925 |
| 400 | .1 | .20 | −.156 | .585 | −.002 | .115 | .920 |
| 1600 | 0 | .99 | .005 | .975 | −.010 | .001 | .945 |
| 1600 | .01 | .75 | −.031 | .835 | .016 | .005 | .955 |
| 1600 | .03 | .37 | −.079 | .530 | .023 | .055 | .915 |
| 1600 | .1 | .06 | −.167 | .110 | −.218 | .265 | .155 |

At a fixed noise level the proxy gets worse as `n` grows, because the
points crowd together while the noise does not shrink. `ρ̂` is then biased
towards zero while its sd stays at .04–.06, and coverage collapses: at `n = 1600` and
`ω = 0.1`, the proxy keeps 6% of the true neighbours and the `ρ` interval
covers 11% of the time. The bias reaches `δ̂` too, through the omitted
spatial structure.

## Covariate-distance `W`

When locations are unknown, Fülle & Otto (2024) build `W` from distances
between units' covariates, following Santi et al. (2021).
[`covariate_weights`](@ref) builds that matrix:
- it standardises each covariate;
- it then takes `k` nearest neighbours with equal weights.

```julia
W = covariate_weights(hcat(size_rank, sector_beta, leverage), 10)
m = fit_spatial_log_garch(returns, W)
```

A proxy `W` is a misspecified `W`. None of the three papers gives a way to
measure from the data how close it is to the true one. Santi et al.'s
measure, `cor(W_X u, W u)`, needs the true `W`. The Monte Carlo's design C
shows what a poor proxy does to `λ̂` and `ρ̂`. Read `ρ̂` from a proxy `W` as
evidence that the variance has spatial structure, not as an estimate of
how much.

Fülle & Otto also choose the covariates by forward selection on the
likelihood. That search is not in the package: nothing corrects the
inference that follows it.

## References

- Fülle, M. J., & Otto, P. (2024). Spatial GARCH models for unknown spatial
  locations – an application to financial stock returns. *Spatial Economic
  Analysis*, 19(1), 92–105.
- Santi, F., Dickson, M. M., Espa, G., Taufer, E., & Mazzitelli, A. (2021).
  Handling spatial dependence under unknown unit locations. *Spatial
  Economic Analysis*, 16(2), 194–216.
- Sato, T., & Matsuda, Y. (2021). Spatial extension of generalized
  autoregressive conditional heteroskedasticity models. *Spatial Economic
  Analysis*, 16(2), 148–160.
