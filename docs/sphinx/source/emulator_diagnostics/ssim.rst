SSIM
====

SSIM measures structural similarity using Gaussian weights and one reference
data range for the analysis period. It produces per-time scores, a time-mean
local map, and its latitude-weighted global mean. Configure ``win_size``,
``sigma`` and ``data_range`` under ``params.default``; HEALPix input requires
regridding. Run ``python -m aqua.diagnostics.emulator.ssim.cli_ssim --config CONFIG``.

The interpretation follows Ebert-Uphoff et al. (2024), *Structural similarity
metrics for weather and climate model evaluation*.
