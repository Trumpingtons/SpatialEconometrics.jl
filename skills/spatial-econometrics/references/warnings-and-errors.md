# Reading warnings and errors

The package's messages name the function, the cause and usually the fix.
Read them in full, and quote the relevant part to the user. The ones below
say something about the **data or the model**, not about a mistyped
argument, so they change what can be reported.

The guiding rule: **a fit that throws has no estimate to report, and a fit
that warns has an estimate whose meaning is qualified.** Don't work around
either by catching the error, widening an option until the warning goes
away, or reporting the number without the warning.

## The spatial parameter's search interval

These come from the fits that search one spatial parameter over an interval:
`fit_spatial`, `fit_spatial_error`, `fit_spatial_panel` and the others with a
`rho_bounds` or `lambda_bounds` keyword.

**Warning: "the estimate … is at the lower edge of `rho_bounds`"** (or
`lambda_bounds`). The likelihood is still rising at the bound, so `ρ̂` is
not an interior maximum and **its standard error does not apply**. The
usual cause is that the default `(−0.995, …)` cuts off admissible negative
values: for a row-standardised `W` the interval is `(1/λ_min, 1)`, and
`1/λ_min` is below −1 unless the graph is bipartite. To search further:

```julia
using LinearAlgebra
λmin = minimum(real, eigvals(Matrix(W)))      # dense: fine up to a few thousand units
m = fit_spatial(SAR(), W, X, y; method = ExactLU(), rho_bounds = (1 / λmin + 1e-3, 0.995))
```

A strongly negative `ρ̂` is unusual in applied work. Before reporting one,
check `W` (row order, a binary `W` mistaken for a standardised one) and
the specification: a misaligned `W` can produce it.

**Warning: "… at the upper edge …".** `ρ̂` runs to 1 (for a
row-standardised `W`). The model is close to a unit root in space, often
because of an omitted trend, a common factor, or a missing fixed effect
in a panel. Check `dependence_verdict` in panels and the specification;
don't report the SE.

**`ArgumentError`: "`I − ρW` is singular at ρ = 1/r(W) … W is not
row-standardised".** `W`'s spectral radius is above 1 (a binary or
distance `W`), so the default interval crosses a singularity. Either
`row_standardize(W)`, or keep `W` and pass bounds inside
`(−1/r(W), 1/r(W))`. Remember that `ρ` then has a different scale.

**`ArgumentError`: "… reaches |ρ| ≥ 1, where the Chebyshev log-determinant
is undefined".** Bounds beyond ±1 need `method = ExactLU()`.

## Spatial log-GARCH

**`ErrorException`: "fit_spatial_log_garch: … The search ran to the edge of
`bounds`: this sample has no interior maximum".** With a row-standardised
`W` and an intercept, the quasi-likelihood rises without bound as
`λ → 1`, so in some samples there is no estimate at all. It is most common
when `λ + ρ` is near 1 and `n` is small. This is a property of the model,
not a numerical failure: report that the model has no estimate for this
sample. Do not narrow `bounds` to manufacture one; an estimate on an
arbitrary bound is not a maximum-likelihood estimate.

**"… The quasi likelihood may be flat in λ".** `λ` is weakly identified,
typically without covariates `Z`. Adding a covariate that drives volatility
helps; otherwise report `λ + ρ`, which is identified much better.

## Convergence

**`fit_mgwr`: "back-fitting did not converge", "bandwidths settled at
sweep … but the back-fitting had not converged", "the bandwidths never
settled in … sweeps".** Raise `max_iter` (or loosen `tol`) and refit.
Slow convergence is expected when the covariates are spatially correlated
with each other. When the bandwidths never settle, `enp`, `hat_trace` and
`AICc` are `NaN`.

**`fit_mgwr`: "skipping the exact degrees-of-freedom accumulator because
N = … exceeds max_inference_n".** Large `N`: pass `inference =
:stochastic`, which estimates the same quantities in `O(N)` memory, or
raise `max_inference_n` if the memory is available (the message says how
much).

**`fit_mgwr`: "the smoother did not converge on at least one probe" / "the
smoother recursion did not converge" / "the accumulated smoother reproduces
the fitted values only to …".** The inference quantities (`enp`,
`hat_trace`, `AICc`, sometimes `se_local`) are unreliable or `NaN`. The
coefficient surfaces may still be usable; check `converged`.

**`fit_gravity`: "IRLS did not converge".** Check `m.converged`. Common
causes are many zero flows within a fixed-effect cell (an origin or
destination with all-zero flows has no information), or a covariate that
is perfectly predicted by the fixed effects. Do not report the
coefficients of an unconverged fit.

## Other model warnings

**`fit_sp_cce`: "the cross-sectional averages already span the time
effects, so TimeFE is redundant".** Use `IndivFE()` or `NoFE()` with CCE.

**`shift_share_design`: "exposure shares do not sum to a constant across
shocks".** The sum of exposure is then a confounder (Borusyak, Hull &
Jaravel 2022) and belongs in `controls`. The package does not add it.

**`spvar_forecast` / `spatial_forecast`: "the fitted … is not stationary".**
The forecasts diverge with the horizon. Report them only with that caveat,
or respecify.

**`check_interference` reports `identified = false`.** The exposure design
cannot separate a spillover from the direct effect with this `W`. Don't fit
`fit_spatial_did` and report a spillover; the message says why.

## Simulation

**`monte_carlo_spatial`: "… replications failed and were excluded".**
The reported bias and coverage are conditional on the fit succeeding, which
is itself a result (as with the log-GARCH edge above). Report the failure
count beside them.

**"the estimator reported … of the DGP's … parameters; the trailing rows
are NaN".** The estimator does not estimate every parameter of the DGP
(for example, a SAR fitted to SDM data). The `NaN` rows are not estimates.

## Rank and input errors

`ArgumentError`s about the design name the offending column: an intercept
passed in `X` (the fits add one), a covariate that is a linear combination
of others, or, for SDM, a `W` for which `WX` lies in the span of `X`. Drop
the named column. A `DimensionMismatch` usually means a panel's `N * T`
does not match the length of `y`, or `W` is `NT × NT` instead of `N × N`.
