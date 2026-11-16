# Spatial log-GARCH

API reference for `SpatialGARCH`. See
[Spatial log-GARCH](../models/spatial_garch.md) for the discussion, including
what identifies `λ` and what a covariate-distance `W` costs.

## Module

```@docs
SpatialEconometrics.SpatialGARCH
```

## Estimation

```@docs
SpatialLogGARCHModel
fit_spatial_log_garch
```

The covariate-distance weight matrix, [`covariate_weights`](@ref SpatialEconometrics.SpatialWeights.covariate_weights), is
documented with the other [weight matrices](weights.md).
