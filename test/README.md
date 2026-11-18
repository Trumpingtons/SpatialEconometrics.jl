# Test suite

The whole suite is one file, `runtests.jl`: 62 top-level testsets inside a
single outer testset, "SpatialEconometrics.jl Full Suite Verification".

| Run | Result (2026-11-18) |
|---|---|
| Julia 1.13.4, both extensions loaded | 4,139 passed, none broken |
| Julia 1.10.13, no extensions | 4,104 passed, 2 broken |

The two broken entries are markers that stand in for the Makie and Turing
testsets when those packages are absent (see below). A local run takes about
8 minutes. CI takes 20–80 minutes per job, partly because it collects
coverage.

## Running it

```bash
# what CI's main jobs run; the extensions are not loaded
julia --project=. -e 'using Pkg; Pkg.test()'

# the full suite with the extensions, if Makie and Turing are installed
# in your default environment (@v1.x)
julia --project=. test/runtests.jl
```

`Pkg.test()` builds a sandbox that holds only the test dependencies, so it
never sees Makie or Turing, even when they are installed. Only the second
form runs testsets 25 and 26. To install their triggers:

```bash
julia -e 'using Pkg; Pkg.add(["CairoMakie", "Turing", "Distributions"])'
```

Run the suite from the repository: testset 11b reads
`notebooks/data/columbus.csv` and skips six assertions if the file is not
there. So a copy of `src/` and `test/` alone gives a lower count. Testset 45
reads `skills/`, and without it errors.

## What the testsets cover

The numbers follow the order in which the features were added.

| Testsets | Area |
|---|---|
| 1–5b | Cross-section SAR/SDM, static and dynamic panels, GS2SLS, system GMM |
| 6–8 | GRAS / cross-entropy, Driscoll–Kraay and Conley HAC, the formula interface |
| 9–11d | Diagnostics, SEM, GWR and MGWR, gravity fixed effects |
| 12–18b | Weights, model selection, panel diagnostics, higher-order models, regimes, simulation, tables |
| 19–21c | Spatial probit, gravity (PPML/NB), eigenvector spatial filtering |
| 22–23 | Regressions found by the example scripts and by the full sweep |
| 24–26 | Extension entry points; the Makie and Turing extensions |
| 27–33 | Durbin score tests, `k`NN selection, identification guards, Bayesian comparison, ST-PCA, performance invariants, compact GWR |
| 34–37 | Causal inference, synthetic control, shift-share, networks |
| 38–40 | Cross-sectional dependence, spatio-temporal models, spatial VAR |
| 41–44 | Semiparametric SAR; Li & Yang; Bai & Chen; the spatial log-GARCH |
| 45 | The Claude skill's function reference against the docstrings |

A lettered testset (`4b`, `11c`, …) was added later to its area, usually by
an audit finding that `docs/AUDIT.md` records.

## What an assertion must be

Four rules apply to every new testset. The second is the standing rule
written in `docs/AUDIT.md`; the others are the project's working
practice:
- **A value is compared with an independently computed value:** a dense
  recomputation, automatic differentiation, a closed form or a brute-force
  sum. Not with the code's own output from an earlier run.
- **A variance or test size is checked by a null-distribution Monte Carlo.**
  The design is held fixed and the null is drawn many times, and the check
  needs evidence of power: a wrong formula must fail it by a wide margin.
- **A margin is measured, not assumed.** Each assertion about a random
  quantity is run on Julia 1.13 and 1.10, whose random streams differ, and
  its distance from the bound is recorded. The smallest margin in the suite
  is 1.47×, in 42c (the size-table notes in `ROADMAP.md` keep the record).
- **No `@test` sits inside a guard that a random draw can switch off.**
  Assert the implication, `@test !guard || claim`, so that a failed guard
  cannot silently remove coverage.

Structs used by the tests are defined at the top of the file, because
Julia does not allow them inside a testset.
