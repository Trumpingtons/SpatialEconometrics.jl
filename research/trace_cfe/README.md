# Trace-corrected closed-form estimator (research, not package code)

Exploratory Monte Carlo for an idea that came out of reading Geniaux (2026)
in full on 2026-11-18. ROADMAP #13 pass 2 is **on hold by decision**. Nothing
here is part of the package, exported or tested by `test/runtests.jl`.

## The idea

With a projection first stage `P_B` (the basis `B` spans the smooths of the
covariates and of the coordinates), the closed-form estimator's (CFE's)
moment at the true `ρ` has an exact mean:

    E[g_n(ρ₀)] = −σ² tr(P_B W) / n

This is the own-observation term that Geniaux removes by cross-fitting (his
Remark 7). It is O(K/n), so it is first-order once K is comparable to √n,
and a free surface makes K large. Two corrections remove it in closed form,
and each keeps `ρ̂` as the root of one quadratic:

- **Pooled:** in the moment's second factor, use `e₁ + τe₀` and `e₂ + τe₁`,
  with `τ = tr(P_B W)/(n − K)`. This is the same as replacing `W` by `W + τI`
  in the instrument, and it makes the quadratic form trace-zero (Lee 2007).
- **Exact diagonal:** use a vector `d` in place of `τ`, where `d` solves
  `(M∘M) d = diag(P_B W)` and `M = I − P_B`. Then `A = MW + MDM` has a zero
  diagonal, so the moment has mean zero under any heteroskedasticity, and its
  variance needs no third or fourth moments.

**Robust variance of the moment:** `2σ̂′(A_s∘A_s)σ̂ + Σᵢ(âᵢ² − noiseᵢ)σ̂ᵢ²`,
where:
- `σ̂ᵢ² = ûᵢ²/Mᵢᵢ`;
- `â = MW P_B y₀`;
- `noise = diag(G(Q′Σ̂Q)G′)`, with `G = MWQ` and `Q` an orthonormal basis of
  span `B`.

`ρ̂`'s standard error is that square root divided by `|b − 2ρ̂a|`.

**Prior art (checked 2026-11-18; see `ExtRef/README.md`).** Both corrections
are Jin & Lee (2012):
- the pooled `τ` is their homoskedastic root estimator with
  `P = W − tr(W M)/(n − d)·I` (§3.1, footnote 12);
- the zero diagonal under heteroskedasticity is their §3.2, which builds it
  more simply, as `P = A − Diag(AM)Diag(M)⁻¹M`, without solving `(M∘M)d`.

Liu & Yang (2015) apply the same per-unit idea to the QML score. All of these
take a regressor matrix of fixed dimension, with `PM` bounded in row and
column sums.

What these papers do not cover (and see the comparison under Results):
- a basis whose size `K` grows with `n`, where `M` is dense and the trace term
  is first-order. The spline GMM papers (Chen, Song & Yu 2025; Lu, Tian &
  Yang 2024) avoid this regime with rate conditions such as `k²/n → 0`;
- the noise term in the robust variance, which is negligible at fixed `d`;
- the link to Geniaux's own-observation bias.

Su (2012) estimates `ρ` with linear moments only.

## Scripts

Run from the repository root with `julia --project=. research/trace_cfe/<file>.jl`.
Every script uses fixed seeds, and its output is committed beside it as
`<file>_output.txt`.

| Script | What it measures | Run time |
|---|---|---|
| `tc_first_look.jl` | Bias of the plain CFE and the pooled correction; K = 29, 92, 191; constant variance | 35 s |
| `tc_first_look_het.jl` | The same with spatially varying variances; adds the per-unit approximation `d_i = (P_B W)_ii / M_ii` | 36 s |
| `tc_lib.jl` | Shared code: the design objects, the three estimators and the variance of the moment | — |
| `tc_null.jl` | Null-distribution check: fixed design, `f` in span `B`, errors redrawn 2,000 times; normal and χ²₃ errors, constant and varying variances, `ρ₀` = 0 and 0.4. The first block of its output is a 20-replication warm-up; ignore it | 43 s |
| `tc_decomp.jl` | Splits the robust variance's bias into its quadratic and linear parts, before the noise term was subtracted | 6 s |
| `tc_jinlee.jl` | Jin & Lee (2012, §3.2) against the exact diagonal in the concurvity design: their `P = W − Diag(WM)Diag(M)⁻¹M` (JL-W) and their two-step `A = G(ρ̂)` (JL-G), all with the same robust variance; 16 cells at K = 100 and 4 at K = 199, 500 replications each | 15 min |
| `tc_concurvity.jl` | `x2 = (1−ω)·smooth(u,v) + ω·noise` with free smooths of `(u, v)` and `x2`, K = 100, 500 replications per cell, against an oracle | 24 s |

`tc_first_look*.jl` are self-contained and predate `tc_lib.jl`. Their
per-unit correction is the approximation that the exact diagonal later
replaced.

## Results (n = 900, 6-NN `W`, Chebyshev bases)

**Bias of `ρ̂`, the first look:**

| K | Plain CFE | Pooled `τ`, constant variance | With varying variances: pooled `τ` | Per-unit approximation |
|---|---|---|---|---|
| 29 | −0.07 | 0.000 | — | — |
| 92 | −0.19 to −0.22 | −0.002 | −0.05 to −0.06 | −0.006 to −0.007 |
| 191 | −0.47 to −0.53 | 0.000 | −0.08 | −0.011 to −0.012 |

**Null-distribution check (`tc_null.jl`), all 8 cells:**
- the moment's mean has |z| ≤ 1.5;
- its simulated variance against the exact formula: ratio 0.94–0.98, |z| ≤ 1.9;
- the robust variance estimator against the exact formula: ratio 0.998–1.026.
  The remaining bias is conservative and comes from the quadratic plug-in;
  z reaches 7 in the constant-variance χ²₃ cell;
- **power check:** under varying variances the homoskedastic formula has a
  ratio of 0.64 (z = −160);
- coverage of the trace-corrected estimator is 0.948–0.958, and of the plain
  CFE 0.01–0.05.

Before the noise term was subtracted, the robust variance ran 3–10% high.
`tc_decomp.jl` traces the excess to the linear part: `â` carries
`MWP_B ε`, and the analytic prediction of that bias (1.661) matches the
simulation (1.668).

**Concurvity (`tc_concurvity.jl`), 16 cells:**
- the exact-diagonal correction's bias is ≤ 0.004 in every cell, including
  `ω = 0`, where `x2` is purely a function of location;
- its coverage is 0.936–0.950, with the standard errors 2–4% below the
  simulated sd, within Monte Carlo error at R = 500;
- the plain CFE has bias −0.16 to −0.34;
- the pooled `τ` leaves about −0.03 under varying variances;
- the sd is about 1.4–1.5 times the oracle's.

In this design, removing the trace term leaves no separate "population
channel".

**Against Jin & Lee (`tc_jinlee.jl`), 20 cells:**
- all three estimators are unbiased, |bias| ≤ 0.009;
- ours, with `M` on both sides of the moment matrix, has a smaller sd in every
  cell: 0.76–0.88 of JL-W's, so 0.58–0.77 of its variance. The gain is largest
  with constant variances (0.58–0.73) and holds at K = 199 (0.62–0.77);
- JL-G's two-step `G(ρ̂)` gains nothing over JL-W;
- coverage: ours 0.936–0.950 (se/sd 0.95–1.02); JL-W 0.924–0.942 and JL-G
  0.908–0.948 (se/sd 0.92–0.98). The Monte Carlo SE of a coverage at R = 500
  is 0.010.

Not yet checked: the efficiency gain has no theory yet, and the robust variance
for Jin & Lee's `C` has not had its own null-distribution check.

## Not settled

- A penalised smoother (`M = I − S_λ`, not idempotent) instead of a
  projection, and Geniaux's own E3 design.
- Large n: `A` is a dense n × n matrix. At n = 900 that is cheap; at Geniaux's
  n = 25,357 it would need iterative solves.
- Theory: a dense `P_B` breaks the bounded-row-sum condition of the Kelejian &
  Prucha CLT, so a different CLT for linear-quadratic forms is needed, along
  with the rate at which K may grow.
- The 1–3% conservative quadratic plug-in bias of the robust variance.

## References

- Geniaux, G. (2026). arXiv:2611.16592v1, preprint.
- Lee, L.-F. (2007). GMM and 2SLS estimation of mixed regressive, spatial
  autoregressive models. *Journal of Econometrics*, 137(2), 489–514.
  doi:10.1016/j.jeconom.2005.10.004
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional Science
  and Urban Economics*, 42(3), 446–458. doi:10.1016/j.regsciurbeco.2011.12.004
- Liu, S. F., & Yang, Z. (2015). Modified QML estimation of spatial
  autoregressive models with unknown heteroskedasticity and nonnormality.
  *Regional Science and Urban Economics*. doi:10.1016/j.regsciurbeco.2015.02.003
- Su, L. (2012). Semiparametric GMM estimation of spatial autoregressive
  models. *Journal of Econometrics*, 167(2), 543–560.
  doi:10.1016/j.jeconom.2011.09.034
- Cattaneo, M. D., Jansson, M., & Newey, W. K. (2018). Inference in linear
  regression models with many covariates and heteroscedasticity. *JASA*,
  113(523), 1350–1361. doi:10.1080/01621459.2017.1328360
- Kline, P., Saggio, R., & Sølvsten, M. (2020). Leave-out estimation of
  variance components. *Econometrica*, 88(5), 1859–1898.
  doi:10.3982/ECTA16410
- Lee, J. (2026). arXiv:2608.22706v1, preprint (buffered spatial
  cross-fitting, §2.8 and §3).
