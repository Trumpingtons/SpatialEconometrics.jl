# A CLT for the zero-diagonal closed-form moment when the basis grows

**Status: a proof sketch, written 2026-11-19. It has not been checked by
anyone else.** Each step below says whether it is proved here, sketched, or
assumed. The numerical checks are in `tc_clt.jl` and its output. ROADMAP #13,
research item, theory step 1.

## Setting

- `y = ρWy + f + ε`, with `n` units, `W` known, and `ε` independent.
- `B` is an `n × K` basis, `P_B` the projection on its span, `M = I − P_B`,
  and `hᵢᵢ = (P_B)ᵢᵢ` the leverages. `K` may grow with `n`.
- The moment is Jin & Lee's (2012): `g(ρ) = y₀(ρ)′ C y₀(ρ)`, with
  `y₀(ρ) = y − ρWy`, `C = PM` and `diag(C) = 0`. For the first step,
  `P = W′ − Δ` with `Δ = Diag(W′M)Diag(M)⁻¹`. The two-step REsd replaces `W′`
  by `G(ρ̃)′`; see Remark 2.
- The root `ρ̂` solves `g(ρ̂) = 0`, a quadratic in `ρ`.
- Write `Cs = (C + C′)/2`, `Σ = Diag(σ₁², …, σₙ²)`, and `L` for the strictly
  lower-triangular part of `Cs`.

The design (`W`, `B`, `f`) is fixed. Every statement is conditional on it.

## Assumptions

- **A1.** The `εᵢ` are independent, with mean 0, variances
  `σᵢ² ∈ [σ̲², σ̄²]` where `σ̲ > 0`, and `supᵢ E εᵢ⁴ < ∞`.
- **A2.** `diag(W) = 0`, and `W` has row and column sums bounded in absolute
  value. So `‖W‖_op ≤ c_W`, and each column has `‖Weᵢ‖ ≤ c_W`.
  **This is the only bounded-row-sum condition. It is on `W`, not on `C` or
  `M`.**
- **A3.** The largest leverage stays below 1: `maxᵢ hᵢᵢ ≤ h̄ < 1`.
- **A4.** At first, `f ∈ span(B)`. Remark 1 drops this.
- **A5 (non-degeneracy).** `liminf ‖Cs‖_F²/n > 0`. This is assumed, not
  derived. Measured values are 0.05–0.12 in `tc_clt.jl`.
- **A6 (Lindeberg for the linear part).** `maxᵢ bᵢ² / ‖b‖² → 0`, where
  `b = MP′f`. It holds if `maxᵢ hᵢᵢ → 0` (Lemma 2b), but the measurements
  show that this is not necessary.

## Lemma 1: the moment at the true ρ (proved)

At `ρ₀`, `y₀ = f + ε`. Since `Mf = 0`, `Cf = PMf = 0`, so
`g(ρ₀) = b′ε + ε′Cε`, with `b = C′f = MP′f`. Then:
- `E g(ρ₀) = tr(CΣ) = Σᵢ Cᵢᵢσᵢ² = 0`, under any heteroskedasticity, because
  `diag(C) = 0`;
- `V := Var g(ρ₀) = 2σ′(Cs∘Cs)σ + Σᵢ bᵢ²σᵢ²`, where `σ` is the vector of
  variances. The zero diagonal removes every third- and fourth-moment term,
  including the covariance between the linear and quadratic parts, because
  `E εᵢεⱼεₖ ≠ 0` only when `i = j = k`.

`tc_jinlee.jl` checks this exact variance against simulation.

## Lemma 2: the moment matrix is bounded although M is dense (proved)

(a) `‖M‖_op = 1`, so `‖C‖_op ≤ ‖P‖_op ≤ c_W + maxᵢ|Δᵢᵢ|`, and
`‖Cs‖_op ≤ ‖C‖_op`.

(b) `Δᵢᵢ = (W′M)ᵢᵢ/Mᵢᵢ = −(W′P_B)ᵢᵢ/(1 − hᵢᵢ)`, because `diag(W) = 0`. By
Cauchy–Schwarz, `|(W′P_B)ᵢᵢ| = |(Weᵢ)′(P_Beᵢ)| ≤ ‖Weᵢ‖ √hᵢᵢ`. So

    |Δᵢᵢ| ≤ c_W √hᵢᵢ / (1 − hᵢᵢ),

which is bounded under A3 and tends to 0 when `maxᵢ hᵢᵢ → 0`.

(c) With A2 and A3, `‖C‖_op = O(1)`, and with A5,
`‖Cs‖_op / ‖Cs‖_F = O(n^{−1/2})`.

The point of the lemma: Kelejian & Prucha's (2001) CLT asks for bounded row
and column sums of the moment matrix. A dense `M` has row sums of order up to
`√(n hᵢᵢ)`, which breaks that. But the CLT argument below only needs norm
ratios, and the operator norm survives the projection. In `tc_clt.jl`,
`‖Cs‖_op` stays at 0.83–1.08 and the bound in (b) holds in every design.

## Theorem 1: CLT for the moment (sketched)

Under A1–A6, `g(ρ₀)/√V → N(0, 1)`, provided

- (a) `‖Cs‖_op / ‖Cs‖_F → 0`, which Lemma 2(c) and A5 give at rate
  `n^{−1/2}`; and
- (b) `‖L‖_op / ‖Cs‖_F → 0`.

The norm of the triangular truncation grows like `log n` (Kwapień &
Pełczyński 1970; not read). If that bound applies here, (a) gives (b) at rate
`log n/√n`. Measured, `‖L‖_op/‖Cs‖_op` stays at 0.64–0.69 with no visible
growth in `n`.

**Sketch.** Because `diag(Cs) = 0`, `ε′Cε = ε′Csε = 2Σᵢ εᵢ(Lε)ᵢ`. Write
`g = Σᵢ ξᵢ` with `ξᵢ = εᵢ(bᵢ + 2(Lε)ᵢ)`. Since `(Lε)ᵢ` depends only on
`ε₁, …, εᵢ₋₁`, the `ξᵢ` are martingale differences. Apply the martingale CLT
(Hall & Heyde 1980, Corollary 3.1; not read here, but this is the route of
Kelejian & Prucha 2001). Two conditions are needed:

1. **The conditional variance:**
   `Σᵢ σᵢ²(bᵢ + 2(Lε)ᵢ)² / V → 1` in probability. Its mean is exactly 1.
   Its variance has two parts:
   - the linear cross term `4Σᵢσᵢ²bᵢ(Lε)ᵢ = 4(L′Σb)′ε`, with variance
     `16 Σⱼ σⱼ²(L′Σb)ⱼ² ≤ c‖L‖_op²‖b‖²`;
   - the quadratic form `4ε′L′ΣLε`, with variance
     `≤ c‖L′ΣL‖_F² ≤ c‖L‖_op²‖L‖_F²`. The fourth-moment diagonal terms are
     bounded by the same quantity.

   Since `V ≥ 2σ̲⁴‖Cs‖_F²` and `V ≥ σ̲²‖b‖²`, both parts are
   `O(‖L‖_op²/‖Cs‖_F²)` relative to `V²`, which tends to 0 by (b).
2. **Lyapunov:** `Σᵢ E ξᵢ⁴ / V² → 0`. By independence and Rosenthal's
   inequality,
   `E ξᵢ⁴ ≤ c(bᵢ⁴ + ‖ℓᵢ‖⁴)`, where `ℓᵢ` is row `i` of `L`. Then:
   - `Σbᵢ⁴ ≤ maxbᵢ²‖b‖²`, which is `o(V²)` by A6;
   - `Σ‖ℓᵢ‖⁴ ≤ maxᵢ‖ℓᵢ‖²‖L‖_F² ≤ ‖Cs‖_op²‖Cs‖_F²`, which is `o(V²)` by (a).

   Only fourth moments are used, while Kelejian & Prucha assume `4 + η`.

To check in a full proof: the constants in the two variance bounds, and that
nothing in the step depends on bounded row sums of `L`.

## Corollary: the estimator (sketched)

Expand `0 = g(ρ̂) = g(ρ₀) + g′(ρ₀)(ρ̂ − ρ₀) + a(ρ̂ − ρ₀)²`, where
`a = (Wy)′C(Wy)`. Two things need showing:
- `J := −E g′(ρ₀) = 2[f′W′Csf + tr(W′CsΣ)]`, with `liminf J/n > 0`. This
  is the identification condition; it is assumed.
- `g′(ρ₀) − E g′(ρ₀) = O_p(√n)` and `a = O_p(n)`. Both are linear-quadratic
  forms in `ε` whose matrices have bounded operator norm, by Lemma 2, and
  Frobenius norm `O(√n)`.

Then `(ρ̂ − ρ₀) J/√V → N(0, 1)`, and `ρ̂` is `√n`-consistent.

**The rate for K.** Nothing in Theorem 1 needs `K/n → 0`. A3 and A5 allow
`K/n → c ∈ [0, 1)`. This is weaker than the published rate conditions for
spline GMM, such as `k_n²/n → 0` (Lu, Tian & Yang 2024). It is possible
because the zero diagonal removes the trace bias exactly, at any `K`. Rate
conditions are otherwise needed to make that bias small. A large `K` still
costs efficiency, because `‖Cs‖_F²/n` falls as `K/n` rises (from 0.11 at
K/n = 0.04 to 0.05 at K/n = 0.15 in `tc_clt.jl`), but it does not cost
validity.

**Remark 1 (approximation error, sketched).** If `f = P_Bf + r`, then `g`
gains `r′Cr + r′(C + C′)ε`. The second term joins the linear part. The
first is a bias of size at most `‖C‖_op‖r‖² ≤ c n‖r‖_∞²`, and it must be
`o(√V) = o(√n)`. So `√n ‖r‖_∞² → 0`, the usual undersmoothing condition.
For a surface of smoothness `s` in `d` dimensions with `‖r‖_∞ = O(K^{−s/d})`,
this means `K ≫ n^{d/(4s)}`. **That is the binding rate, and it is a lower
bound on K, not an upper one.**

**Remark 2 (REsd, the two-step, sketched).** For every `ρ`, `P(ρ) = G(ρ)′ −
Δ(ρ)` makes `P(ρ)M` zero-diagonal. So `E g` at `ρ₀` is 0 whatever `ρ̃` is
plugged into `P`, and its derivative in `ρ̃` has mean 0 (Neyman
orthogonality). The plug-in adds `(ρ̃ − ρ₀)·O_p(√n) = O_p(1)` to `g`, which
is `o_p(√V)`. So REsd has the same limit as the infeasible `P(ρ₀)`. A full
proof needs `‖∂P/∂ρ‖_op` bounded near `ρ₀`. That holds if `‖G(ρ)‖_op` is
bounded, i.e. `sup‖(I − ρW)⁻¹‖_op < ∞`, which is the usual SAR condition.

**Remark 3 (a penalised smoother).** If `M = I − S_λ` with `S_λ` symmetric
and its eigenvalues in `[0, 1]`, as for a ridge-penalised spline, then
`‖M‖_op ≤ 1` still holds, and Lemma 2(a) carries over. Lemma 2(b) used
`Mᵢᵢ = 1 − hᵢᵢ` and idempotency, so it needs redoing. Not checked
numerically.

## Numerical check (`tc_clt.jl`)

Design: 6-NN `W`, random locations, heteroskedastic standardised `χ²₃` errors,
`R = 2000` draws per design. `z = g(ρ₀)/√V` uses the exact `V` of Lemma 1.

| Design | n | K/n | max hᵢᵢ | ‖Cs‖_op/‖Cs‖_F | skew z | exc. kurt. z | KS | size of \|z\| > 1.96 |
|---|---|---|---|---|---|---|---|---|
| hat basis, K ≈ 2√n | 400 | 0.09 | 0.43 | 0.18 | 0.41 | 1.63 | 0.041 | 0.054 |
| | 2500 | 0.04 | 0.27 | 0.06 | 0.16 | 0.05 | 0.018 | 0.057 |
| hat basis, K ≈ 0.15n | 400 | 0.16 | 0.60 | 0.19 | 0.32 | 1.33 | 0.038 | 0.050 |
| | 2500 | 0.14 | 0.76 | 0.08 | 0.14 | 0.16 | 0.030 | 0.048 |
| Chebyshev, K ≈ 2√n | 400 | 0.09 | 0.72 | 0.17 | 0.51 | 1.86 | 0.038 | 0.050 |
| | 2500 | 0.04 | 0.75 | 0.06 | 0.14 | 0.05 | 0.016 | 0.046 |
| **power check:** two complete groups | 400 | 0.09 | 0.43 | 1.00 | 3.02 | 15.3 | 0.193 | 0.041 |
| | 1600 | 0.05 | 0.37 | 1.00 | 2.75 | 11.1 | 0.185 | 0.048 |

The Monte Carlo SEs at R = 2000 are 0.055 for the skewness, 0.11 for the
excess kurtosis and 0.005 for the size. The KS 95% critical value is 0.030.
The full output, with n = 900 and 1600 and every quantity, is in
`tc_clt_output.txt`.

- **Normality improves with n in all three kNN regimes.** That includes
  K/n fixed at about 0.15, and the Chebyshev basis, whose largest leverage
  stays at 0.7–0.8. So the CLT does not need `max hᵢᵢ → 0`.
- **The skewness falls like `‖Cs‖_op/‖Cs‖_F`.** Its ratio to that quantity
  is about 2.3–2.5 across n, which is the Berry–Esseen-type rate the sketch
  predicts.
- **The bound in Lemma 2(b) holds in every design.** For example, the
  largest `|Δᵢᵢ|` is 0.148 against a bound of 0.196.
- **Power check.** With `W` made of two complete groups, `‖Cs‖_op/‖Cs‖_F`
  stays at 1, and `z` stays far from normal at every n (skewness 2.4–3.0,
  KS 0.19). So the check can detect a failing CLT. This is the
  Kelejian–Prucha case of a number of neighbours that grows with n.
- **Caveat:** the size of `|z| > 1.96` is close to 0.05 even in the power
  check. A symmetric two-sided size is not a sensitive test of normality
  here; the skewness and the KS distance are.

## Still to do

1. Turn the sketches into proofs. Theorem 1 needs the variance-bound
   constants and either the triangular-truncation bound or a direct bound on
   `‖L‖_op`. The Corollary needs the LLN for `g′` written out.
2. A primitive condition for A5, in terms of `W` and `K/n`.
3. Theory step 2: the consistency of the robust variance `V̂`, including the
   noise term for the estimated `P_By₀`.
4. Remark 3 for a penalised smoother, with a numerical check.
5. Read the CLT literature cited here before relying on it: de Jong (1987)
   for clean quadratic forms, and Bhansali, Giraitis & Kokoszka (2007) for
   conditions on `‖A‖_op/‖A‖_F`. Either may already contain Theorem 1's
   quadratic part.

## References

All five have been checked against Scite and exist. None has been read for
this note.
- Bhansali, R. J., Giraitis, L., & Kokoszka, P. S. (2007). Convergence of
  quadratic forms with nonvanishing diagonal. *Statistics & Probability
  Letters*, 77(7), 726–734. doi:10.1016/j.spl.2006.11.007
- de Jong, P. (1987). A central limit theorem for generalized quadratic forms.
  *Probability Theory and Related Fields*, 75, 261–277. doi:10.1007/BF00354037
- Hall, P., & Heyde, C. C. (1980). *Martingale Limit Theory and Its
  Application*. Academic Press. doi:10.1016/C2013-0-10818-5
- Kelejian, H. H., & Prucha, I. R. (2001). On the asymptotic distribution of
  the Moran I test statistic with applications. *Journal of Econometrics*,
  104(2), 219–257. doi:10.1016/S0304-4076(01)00064-1
- Kwapień, S., & Pełczyński, A. (1970). The main triangle projection in matrix
  spaces and its applications. *Studia Mathematica*, 34, 43–67.
  doi:10.4064/sm-34-1-43-67

Read, and held in `ExtRef/`: Jin & Lee (2012); Lu, Tian & Yang (2024).
