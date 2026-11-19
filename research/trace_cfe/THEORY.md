# A CLT for the zero-diagonal closed-form moment when the basis grows

**Status: written 2026-11-19, and not checked by anyone else.** Each step
below says whether it is proved here, sketched, or assumed. "Proved" means
proved from the stated results of the cited works. Revised the same day after
reading de Jong (1987): Theorem 1 is now proved, not sketched, and step 2 has
a first analysis. The numerical checks are in `tc_clt.jl` and its output. ROADMAP #13,
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

(b) For any `n × n` matrix `A`, `|(A′M)ᵢᵢ| = |(Aeᵢ)′(Meᵢ)| ≤ ‖A‖_op ‖Meᵢ‖`,
and `‖Meᵢ‖² = Mᵢᵢ = 1 − hᵢᵢ`. So `Δ = Diag(A′M)Diag(M)⁻¹` satisfies

    |Δᵢᵢ| ≤ ‖A‖_op / √(1 − hᵢᵢ),

which is bounded under A3. This covers REsd's `A = G(ρ)`, whose diagonal is
not zero. When `diag(A) = 0`, as for `A = W`, there is a finer bound:
`(A′M)ᵢᵢ = −(A′P_B)ᵢᵢ`, and Cauchy–Schwarz gives

    |Δᵢᵢ| ≤ ‖Aeᵢ‖ √hᵢᵢ / (1 − hᵢᵢ),

which tends to 0 when `maxᵢ hᵢᵢ → 0`.

(c) With A2 and A3, `‖C‖_op = O(1)`, and with A5,
`‖Cs‖_op / ‖Cs‖_F = O(n^{−1/2})`.

The point of the lemma: Kelejian & Prucha's (2001) CLT asks for bounded row
and column sums of the moment matrix. A dense `M` has row sums of order up to
`√(n hᵢᵢ)`, which breaks that. But the CLT argument below only needs norm
ratios, and the operator norm survives the projection. In `tc_clt.jl`,
`‖Cs‖_op` stays at 0.83–1.08 and the bound in (b) holds in every design.

## Lemma 3: the triangular part (proved)

Let `L` be the strictly lower-triangular part of a symmetric `A`. Then
`‖L‖_op ≤ ⌈log₂ n⌉ ‖A‖_op`.

*Proof.* Pad the index set to `2^m` with `m = ⌈log₂ n⌉`. At level
`k = 1, …, m`, cut the indices into consecutive blocks of size `2^{m−k+1}`,
and each block into two halves. Let `L_k` hold the entries of `A` whose row is
in the second half and whose column is in the first half of the same block.
Each pair `i > j` belongs to exactly one level: the finest one at which `i`
and `j` share a block but not a half. So `L = Σ_k L_k`. Within one level the
blocks have disjoint rows and disjoint columns, so `‖L_k‖_op` is the largest
operator norm of its sub-blocks, and no sub-block of `A` has a norm above
`‖A‖_op`. ∎

The log factor is known to be the right order for triangular truncation
(Kwapień & Pełczyński 1970; not read). In `tc_clt.jl` the measured ratio
`‖L‖_op/‖Cs‖_op` is only 0.64–0.69.

## Theorem 1: CLT for the moment (proved, given Hall & Heyde's Corollary 3.1)

Assume A1. Write `σ̲, σ̄` for the variance bounds and `μ̄₄ = supᵢ E εᵢ⁴`. If

- (a′) `(log n)² ‖Cs‖_op² / ‖Cs‖_F² → 0`, and
- (c′) `maxᵢ bᵢ² / V → 0`,

then `g(ρ₀)/√V → N(0, 1)`. Under A2–A5, (a′) holds at rate `log² n/n` by
Lemma 2. (c′) is implied by A6, and it is weaker than A6 when the quadratic
part dominates `V`.

*Proof.* Write `c = Cs`, which is symmetric with zero diagonal, and
`uᵢ = Σ_{j<i} cᵢⱼεⱼ`, so that `u = Lε`. Then `ε′Cε = 2Σᵢ εᵢuᵢ`, and
`g = Σᵢ ξᵢ` with `ξᵢ = εᵢ(bᵢ + 2uᵢ)`. Since `uᵢ` depends only on
`ε₁, …, εᵢ₋₁`, the `ξᵢ` are martingale differences for
`Fᵢ = σ(ε₁, …, εᵢ)`. Also

    V = Σᵢ σᵢ²bᵢ² + 2Σ_{i≠j} σᵢ²σⱼ²cᵢⱼ²  ≥  max(σ̲²‖b‖², 2σ̲⁴‖Cs‖_F²).

Hall & Heyde's (1980) Corollary 3.1 needs the two conditions below. Its
nesting condition holds because `Fᵢ` does not depend on `n`.

1. **Conditional variance.** `S = Σᵢ E(ξᵢ²|Fᵢ₋₁) = Σᵢ σᵢ²(bᵢ + 2uᵢ)²`, with
   `E S = V`, and `S − V = 4A + 4(B − E B)`, where:
   - `A = Σᵢσᵢ²bᵢuᵢ = (L′Σb)′ε`. So
     `Var A ≤ σ̄²‖L′Σb‖² ≤ σ̄⁶‖L‖_op²‖b‖²`.
   - `B = u′Σu = ε′Dε` with `D = L′ΣL`. For independent errors and a
     symmetric `D`, `Var(ε′Dε) ≤ (μ̄₄ + 2σ̄⁴)‖D‖_F²`. Also
     `‖D‖_F² ≤ ‖D‖_op tr D ≤ σ̄⁴‖L‖_op²‖L‖_F²`, and `‖L‖_F² = ‖Cs‖_F²/2`.

   With the lower bounds on `V`, `Var(S)/V² ≤ c₁‖L‖_op²/V`, where `c₁`
   depends only on `σ̲`, `σ̄` and `μ̄₄`. By Lemma 3,
   `‖L‖_op²/V ≤ ⌈log₂ n⌉²‖Cs‖_op²/(2σ̲⁴‖Cs‖_F²)`, which tends to 0 by
   (a′). So `S/V → 1` in probability.
2. **Lyapunov, which implies the conditional Lindeberg condition.**
   By independence, `E ξᵢ⁴ = E εᵢ⁴ · E(bᵢ + 2uᵢ)⁴ ≤ 8μ̄₄(bᵢ⁴ + 16 E uᵢ⁴)`.
   For a weighted sum of independent terms,
   `E uᵢ⁴ ≤ max(μ̄₄, 3σ̄⁴)(Σⱼcᵢⱼ²)²`. Then:
   - `Σᵢbᵢ⁴ ≤ maxᵢbᵢ²‖b‖² ≤ maxᵢbᵢ² V/σ̲²`;
   - `Σᵢ(Σⱼcᵢⱼ²)² ≤ maxᵢ‖cᵢ‖² ‖Cs‖_F² ≤ ‖Cs‖_op²‖Cs‖_F²`, because a row
     norm is at most the operator norm.

   So `Σᵢ E ξᵢ⁴/V² ≤ c₂[maxᵢbᵢ²/V + ‖Cs‖_op²/‖Cs‖_F²]`, which tends to 0 by
   (c′) and (a′). ∎

**What de Jong (1987) adds** (read 2026-11-19, from the page images).
His Theorem 5.2 treats the quadratic part alone: `Σ_{i≠j} aᵢⱼXᵢXⱼ`, with the
`Xᵢ` independent and not necessarily identically distributed, `EXᵢ = 0`,
`EXᵢ² = 1` and `aᵢᵢ = 0`. It gives normality under three conditions:
- (1) `K(n)⁴σ(n)⁻² maxᵢ Σⱼaᵢⱼ² → 0` for some sequence `K(n)`;
- (2) `maxᵢ EXᵢ²1{|Xᵢ| > K(n)} → 0`;
- (3) `σ(n)⁻² maxᵢ μᵢ² → 0`, where the `μᵢ` are the eigenvalues of `(aᵢⱼ)`.

With `Xᵢ = εᵢ/σᵢ` and `aᵢⱼ = σᵢσⱼcᵢⱼ`, condition (3) is our condition (a)
without the log factor, and (1)–(2) hold with uniformly bounded `2 + δ`
moments. So **the quadratic part alone needs neither fourth moments nor
Lemma 3.** The martingale proof above is needed for the linear and quadratic
parts jointly, which de Jong's theorem does not cover.

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
proof needs `‖∂P/∂ρ‖_op` bounded near `ρ₀`. Since `∂G/∂ρ = G²` and `Δ` is
linear in `G′`, `∂P/∂ρ = G′² − Δ[G′²]`: the same diagonal correction applied
to `G′²`. By Lemma 2(b), `|Δ[G′²]ᵢᵢ| ≤ ‖G‖_op²/√(1 − hᵢᵢ)`. So the bound holds
if `‖G(ρ)‖_op` is bounded, i.e. `sup‖(I − ρW)⁻¹‖_op < ∞`, which is the usual
SAR condition, and the leverage is bounded away from 1.

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

## Theory step 2: the robust variance (first analysis, 2026-11-19)

The estimator, as coded in `tc_lib.jl` and `tc_jinlee.jl`, is

    V̂ = 2σ̂′(Cs∘Cs)σ̂ + Σᵢ (âᵢ² − noiseᵢ) σ̂ᵢ²,   σ̂ᵢ² = ûᵢ²/Mᵢᵢ,  û = My₀(ρ̂).

**The linear part.** Since `CP_B = PMP_B = 0`, `â = (C + C′)P_By₀ = b + η`, with
`η = C′P_Bε`. So `E âᵢ² = bᵢ² + Var ηᵢ`, and
`Σᵢ Var ηᵢ = tr(C′P_BΣP_BC) ≤ σ̄²‖C‖_op² K`. So without the noise term, the
linear part is biased upwards by `O(K)`, against `V ≍ n`: a relative bias of
`O(K/n)`. That matches the 3–10% excess measured at K/n of 0.1–0.2 before the
term was added. The noise term, `diag(H(Q′Σ̂Q)H′)` with `H = C′Q`, estimates
`Var ηᵢ` and removes it.

**The plug-in variances are the weak point when K/n does not vanish.**
`E ûᵢ² = Σⱼ Mᵢⱼ²σⱼ²`, with `Mᵢⱼ = −hᵢⱼ` for `j ≠ i` and
`Σ_{j≠i} hᵢⱼ² = hᵢᵢ(1 − hᵢᵢ)`. So

    E σ̂ᵢ² − σᵢ² = Σ_{j≠i} hᵢⱼ²(σⱼ² − σᵢ²) / (1 − hᵢᵢ),

which is at most `hᵢᵢ · maxⱼ|σⱼ² − σᵢ²|`. It is zero under
homoskedasticity, `O(K/n)` on average otherwise, and does not vanish if
`K/n → c > 0`. The products `σ̂ᵢ²σ̂ⱼ²` in the quadratic part add a
correlation bias of the same order: `Cov(ûᵢ², ûⱼ²)` involves
`(Σₖ MᵢₖMⱼₖσₖ²)²`, and summing `cᵢⱼ²hᵢⱼ²` over all pairs gives at most
`maxᵢⱼ cᵢⱼ² · K`, so the relative bias is `O(K/n)`. This is probably the 1–3%
conservative bias seen in `tc_null.jl`.

**The conjecture for step 2,** under A1–A6 with bounded eighth moments, which
the variance of the degree-4 terms of `V̂` needs:
- `V̂/V → 1` in probability when `K/n → 0`;
- the same when `K/n → c > 0` under homoskedasticity;
- `V̂` is inconsistent under heteroskedasticity with `K/n → c > 0`.

This is the many-covariates problem for heteroskedasticity-robust variances
(Cattaneo, Jansson & Newey 2018; Kline, Saggio & Sølvsten 2020). It suggests a
leave-out estimator of the `σᵢ²` and of the products `σᵢ²σⱼ²`, in the manner
of Kline et al., as the remedy. If it works, that would be the first part of
this project that is genuinely new.

**Numerical check (`tc_vhat.jl`, 2026-11-19).**
- Setup: n = 900, a kNN `W`, a Chebyshev surface basis plus one covariate, with
  K/n from 0.05 to 0.39, normal and `χ²₃` errors, and R = 1000. Everything is
  evaluated at the true ρ, and `E V̂/V` is reported with its `z`.
- Checks of the check: the oracle `V̂` (true `σ²`) gives 0.997–1.002 in every
  cell, and the simulated `Var g/V` agrees with 1. The homoskedastic `V̂` fails
  under heteroskedasticity (0.67–0.75 at small K/n, `z` down to −118), so the
  test has power.

Results for the current `V̂` (HC2 plug-in):

| Variances | K/n = 0.05 | 0.10 | 0.19 | 0.28 | 0.39 |
|---|---|---|---|---|---|
| homoskedastic, normal | 1.004 | 1.006 | 1.009 | 1.011 | 1.030 |
| smooth in space, normal | 1.004 | 1.006 | 0.998 | 0.962 | 0.998 |
| rough (iid log-normal), normal | 1.010 | 1.024 | 1.042 | 1.069 | 1.134 |
| rough, `χ²₃` | 1.028 | 1.058 | 1.130 | 1.167 | 1.154 |

So the conjecture holds in part:
- **Rough heteroskedasticity:** the bias grows with K/n, as predicted. It is
  upward (conservative), about 1/3 of K/n for normal errors, and `z` is 5–9
  from K/n = 0.1.
- **Smooth heteroskedasticity:** no bias is detectable. This is consistent
  with the formula, because `σⱼ² − σᵢ²` is small where `hᵢⱼ²` puts its mass.
- **Homoskedastic errors:** at most 3%.

**Cattaneo, Jansson & Newey's `(M∘M)⁻¹û²` cannot be used here.** Their
estimator needs `max hᵢᵢ < 1/2`. These designs have `max hᵢᵢ` from 0.75 to
1.0, because the Chebyshev basis has boundary leverage. At K/n ≥ 0.28 the
estimate reaches 1.1–2.7 × V. A Gaussian removal of `Cov(sᵢ, sⱼ)` makes it
worse, and even goes negative. A remedy must therefore cope with leverage near
1: Kline, Saggio & Sølvsten's leave-out with sample splitting, or a local
(hat-function) basis, whose leverage is bounded. Neither has been tried.

## Still to do

1. Theorem 1 is proved (2026-11-19). The Corollary still needs the LLN for
   `g′` and the root-selection argument written out, and Remark 2 needs the
   same.
2. A primitive condition for A5, in terms of `W` and `K/n`.
3. Theory step 2: the conjecture is checked numerically (`tc_vhat.jl`).
   Still to do: prove the `K/n → 0` case, and find a `V̂` that works with
   rough heteroskedasticity and leverage near 1, for example Kline, Saggio &
   Sølvsten's sample splitting.
4. Remark 3 for a penalised smoother, with a numerical check.
5. de Jong (1987) is read: its Theorem 5.2 is Theorem 1's quadratic part.
   Bhansali, Giraitis & Kokoszka (2007) is still unread, because it is
   paywalled; it is in `ExtRef/TO-FETCH.md`. Cattaneo, Jansson & Newey
   (2018) §3 and Kline, Saggio & Sølvsten (2020) §4.2 have been read via
   alphaXiv. Hall & Heyde's Corollary 3.1 is used as it is usually stated,
   and has not been read here.

## References

All have been checked to exist (via Scite, or by resolving the DOI). Read for
this note: de Jong (1987), §2 and §5, from the page images of Springer's free
copy; Cattaneo, Jansson & Newey (2018), §3 (arXiv 1507.02493); Kline, Saggio &
Sølvsten (2020), §4.2 (arXiv 1806.01494). The others have not been read.
- Bhansali, R. J., Giraitis, L., & Kokoszka, P. S. (2007). Convergence of
  quadratic forms with nonvanishing diagonal. *Statistics & Probability
  Letters*, 77(7), 726–734. doi:10.1016/j.spl.2006.11.007
- Cattaneo, M. D., Jansson, M., & Newey, W. K. (2018). Inference in linear
  regression models with many covariates and heteroscedasticity. *Journal of
  the American Statistical Association*, 113(523), 1350–1361.
  doi:10.1080/01621459.2017.1328360
- de Jong, P. (1987). A central limit theorem for generalized quadratic forms.
  *Probability Theory and Related Fields*, 75, 261–277. doi:10.1007/BF00354037
- Hall, P., & Heyde, C. C. (1980). *Martingale Limit Theory and Its
  Application*. Academic Press. doi:10.1016/C2013-0-10818-5
- Kelejian, H. H., & Prucha, I. R. (2001). On the asymptotic distribution of
  the Moran I test statistic with applications. *Journal of Econometrics*,
  104(2), 219–257. doi:10.1016/S0304-4076(01)00064-1
- Kline, P., Saggio, R., & Sølvsten, M. (2020). Leave-out estimation of
  variance components. *Econometrica*, 88(5), 1859–1898.
  doi:10.3982/ECTA16410
- Kwapień, S., & Pełczyński, A. (1970). The main triangle projection in matrix
  spaces and its applications. *Studia Mathematica*, 34, 43–67.
  doi:10.4064/sm-34-1-43-67

Read, and held in `ExtRef/`: Jin & Lee (2012); Lu, Tian & Yang (2024).
