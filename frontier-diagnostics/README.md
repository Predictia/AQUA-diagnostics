# Deprecated frontier packages

The emulator diagnostics that previously lived here as independent pip
projects (`rmse`, `acc`, `ssim`, `psd`, `cov`, `rollout`,
`spatiotemporal_rollout`, `variability`, `spatial_agg`, `seasonal_cycle`,
`ipo`, `amo`, `vertical_zonal_mean`, `zonal_mean_profile`) have been
consolidated into `aqua.diagnostics.emulator` for D393.2.4.1.

Their standalone `pyproject.toml` files were removed to eliminate generic
top-level package names and Private-license metadata. Historical sources
remain only as read-only reference material until the next housekeeping
cleanup.

Inherited DestinE tools that stay as frontier packages:

- `tropical_cyclones/` (Apache-2.0)
- `tropical_rainfall/`

`global_biases_significance/` was folded into
`aqua.diagnostics.global_biases.StatGlobalBiases` (`method:
welch_yearly|autocorr_effective`).
