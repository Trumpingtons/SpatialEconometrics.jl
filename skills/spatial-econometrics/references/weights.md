# Building and checking `W`

`W` is an assumption about who affects whom, and every spatial estimate is
conditional on it. Say how it was built in every report.

## Constructors

| Source | Constructor | Default scaling |
|---|---|---|
| a GeoDa `.gal` contiguity file | `W, ids = read_gal(path; ids = table_ids)` | row-standardised |
| a GeoDa `.gwt` distance file | `W, ids = read_gwt(path; ids = table_ids)` | see docstring |
| coordinates, `k` nearest | `knn_weights(coords, k; binary = false, row_standardize = true, symmetrize = false)` | row-standardised |
| coordinates, distance decay | `distance_weights(coords, cutoff; decay = :inverse, alpha = 1.0, row_standardize = true)` | row-standardised |
| coordinates, a distance band | `band_weights(coords, d_min, d_max; row_standardize = true)` | row-standardised |
| an adjacency matrix (network) | `network_weights(A; normalize = :row)` | row-normalised |
| group membership | `group_weights` | see docstring |
| covariates, locations unknown | `covariate_weights(Xcov, k)` | row-standardised |
| neighbours at exactly `l` steps | `order_weights(W, l)` | row-standardised |
| separate regions | `block_diagonal_weights([W1, W2, …])` | as given |
| a regular `L × L` lattice | `make_rook_weights(L)`, `make_queen_weights(L)` | row-standardised |

`coords` is `N × D`, rows are units, in the same row order as `y` and `X`.
`make_rook_weights` and `make_queen_weights` are for simulations and
examples, not real data.

Rescaling an existing `W`: `row_standardize` (each row sums to 1),
`spectral_standardize` (divide by the spectral radius, keeping relative
connectedness), `minmax_standardize` (divide by the smaller of the largest
row and column sum). The `!` forms work in place.

## Check it: `spatial_connectivity(W)`

It reports `N`, density, the minimum, mean and maximum neighbour counts,
`n_islands`, `is_symmetric` and `spectral_radius`. Look at:

- **`n_islands > 0`.** An island has no neighbours, its row of `Wy` is
  zero, and it drops out of the spatial part of the model while still
  counting towards `N`. Fix the cause (a distance cutoff below the island's
  nearest-neighbour distance, a unit separated by water in a contiguity
  file) or say how many there are. `knn_weights` never makes islands.
- **`spectral_radius`.** It is 1 for a row-standardised `W`. Anything else
  changes the admissible range of `ρ` (below).
- **`is_symmetric = false`** is normal after row standardisation and not a
  problem in itself. `knn_weights` is asymmetric even before it (`j` among
  `i`'s nearest does not make `i` among `j`'s); `symmetrize = true` makes
  the neighbour relation mutual.
- **Neighbour counts.** A minimum of 1 with a mean of 6 means a few units
  hang on a single link, and their spatial lag is one neighbour's value.

## Pitfalls

- **Latitude and longitude.** The coordinate constructors use Euclidean
  distance. On raw degrees, a degree of longitude shrinks with latitude, so
  distances and cutoffs are distorted away from the equator. Project the
  coordinates first (any equal-distance projection for the region), or at
  least scale longitude by `cos(latitude)` over a small area.
  `ConleySpatialHAC` has a `HaversineDistance` metric for the standard
  errors; the `W` constructors do not.
- **Row order.** `read_gal` without `ids` returns rows in the file's record
  order, which need not match the data table. Pass `ids`. For
  coordinate-based `W`, build from the table's own rows.
- **The range of `ρ` depends on `W`.** For a row-standardised `W` the
  admissible interval is `(1/λ_min, 1)`, where `λ_min` is `W`'s smallest
  eigenvalue: `1/λ_min` is −1 only on a bipartite graph (a rook lattice) and
  below it otherwise (about −1.92 on a queen lattice, −1.1 to −1.2 on `k`NN
  weights). The default search `(−0.995, 0.995)` assumes a row-standardised
  `W`. A binary or distance `W` that is not row-standardised has radius
  above 1 (about 4 for a binary rook lattice), so the fits refuse the
  default interval with an `ArgumentError`. Row-standardise it, or pass
  bounds inside `(−1/r(W), 1/r(W))`.
- **`W^l` is not "neighbours at distance `l`".** `W²` counts walks, so it
  has a nonzero diagonal and double-counts. Use `order_weights(W, l)`.
- **A proxy `W` biases `ρ̂`.** `covariate_weights` (or any `W` built from
  something other than true locations) biases `ρ̂` towards zero, and more
  data does not cure it. In Santi et al.'s (2021) Table 2 the relative bias
  is about −80% for a poor proxy, −8% for a fair one and under 1% for a
  good one. The quality of the proxy cannot be
  measured from the data. Report `ρ̂` from a proxy `W` as attenuated.
- **Choosing `W` by fit.** `select_knn_w` and the Bayesian comparisons pick
  `W` from the data; the SEs of the final fit do not account for that
  choice. Say that `W` was selected and from which candidates.
