# GS2SLS / SARAR Panels

For panels carrying a spatial lag in `y` *and* spatial dependence in the
disturbance, with regressors that are themselves endogenous. Maximum
likelihood is not available here, so estimation is by generalized spatial
two-stage least squares: instruments for the endogenous regressors, spatial
instruments for `Wy`, and a GMM step for the error parameter.

## Specification

```math
y_{it} = \rho W y_{it} + X_{it} \beta + u_{it},
\qquad u_{it} = \lambda W u_{it} + \varepsilon_{it}
```

Two parameters, and they mean different things: `rho` is the spatial lag
in the dependent variable, so it induces spillovers in `y`; `lambda` is
dependence among the disturbances, which affects efficiency and standard
errors but carries no spillover interpretation.

## Estimation

```julia
m = fit_gs2sls_panel(TwoWayFE(), W, X_exog, X_endog, Z_iv, y, N, T;
    exog_names = ["x1", "x2"],
    endog_names = ["tax_rate"])
# m.rho:    spatial lag parameter
# m.lambda: spatial error parameter
```

`Z_iv` holds the external instruments for the columns of `X_endog`, and must
have at least as many columns as `X_endog` for the model to be identified.

## Unequal error variances

By default `lambda` is estimated from Kelejian & Prucha's (1998) moments,
which assume every unit's errors have the same variance. If the variance
moves with a unit's position in `W`, for example with its number of
neighbours, that estimate is biased and its standard error is wrong.
`errors = :heteroskedastic` uses Kelejian & Prucha's (2010) moments instead,
which hold whatever the variances:

```julia
m = fit_gs2sls_panel(TwoWayFE(), W, X_exog, X_endog, Z_iv, y, N, T;
    errors = :heteroskedastic)
```

The coefficients' covariance is then a cluster sandwich by unit. It
includes the covariance between `lambda` and the other coefficients, and
the term that accounts for `lambda` being estimated from residuals.
The over-identification test becomes Hansen's `J`.

In a Monte Carlo with error sd proportional to the neighbour count
(`validation/mc_gs2sls_het.jl`, 2,000 replications), the default `λ̂`
averaged 0.24 for a true 0.4 in a 1,200-unit cross-section. Its interval
covered at 0.70, and in a six-period panel at 0.70–0.77. With
`errors = :heteroskedastic`, `λ̂` averaged 0.376 and covered at 0.95, and
0.946–0.947 in the panels. With constant variance the two paths agree.
In a panel, both paths cluster the coefficients' covariance by unit.

The paper's theory is for a cross-section: `NoFE()` with `T_total = 2`, since
the first period is dropped. The panel version is this package's extension
and rests on the Monte Carlo in `validation/mc_gs2sls_het.jl`.

## Formula DSL

`endo(x, z1 + z2)` marks `x` as endogenous and names its instruments, so the
exogenous block, the endogenous block and the instrument block do not have to
be assembled by hand:

```julia
m = fit_gs2sls_panel(
    @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
    table, W; panel = :region, time = :year
)
```

See [Formula Interface](../formula.md) for the spatial terms, and
[Spatial Error Models](sem.md) for the case where there is no lag in `y`.
