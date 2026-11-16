# Package extensions

Two optional extensions add plotting and Bayesian estimation. Each loads
only when its trigger packages are loaded in the same session. The core
package does not depend on Makie or Turing, and they are not installed with
it.

| Extension | Loads when | Adds methods to |
|---|---|---|
| `SpatialEconometricsMakieExt.jl` | `Makie` is loaded (through any backend, e.g. `CairoMakie`) | `plot_coefficients`, `plot_moran_scatter`, `plot_local_moran`, `plot_gwr_coefficients`, `plot_spatial_effects` |
| `SpatialEconometricsTuringExt.jl` | `Turing` and `Distributions` are both loaded | `bayesian_sar`, `bayesian_sem`, `bayesian_sdm` |

The triggers are declared under `[weakdeps]` and `[extensions]` in
`Project.toml`.

## How they are wired

The function names are declared as stubs in the parent package and
exported from it, and each extension adds methods to those stubs. An
extension cannot export functions of its own: nothing `using`s the
extension module, so a new function defined there could not be reached.

Without the trigger packages, calling a stub fails with an error that
names the package to load (testset 24 checks this).

```julia
using SpatialEconometrics, CairoMakie
fig = plot_moran_scatter(y, W)

using Turing, Distributions
res = bayesian_sar(W, X, y; n_samples = 2000)
```

## Testing

A plain `Pkg.test()` does not test the extensions. Its sandbox holds only
the test dependencies, so testsets 25 (Makie) and 26 (Turing) each record
one broken marker in place of their assertions. To run them:
- locally, install the trigger packages in your default environment and
  run `julia --project=. test/runtests.jl` (see `test/README.md`);
- in CI, the "Package extensions (Makie + Turing)" job runs them.

The Turing extension calls `chebyshev_logdet`, so a change to the
log-determinant routines needs that run too.

See `examples/18_visualization.jl` and `examples/19_bayesian.jl`, and the
manual's [Extensions](../docs/src/extensions.md) page.
