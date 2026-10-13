# Emulator diagnostics (`aqua.diagnostics.emulator`)

AQUA-native suite for Destination Earth DE_393 climate-emulator evaluation
(D393.2.4.1 Workstream A). Prediction and reference datasets are retrieved as a
pair through `EmulatorDiagnostic`, and all NetCDF/figure output uses
`OutputSaver`.

## Contents

| Diagnostic | Role |
|---|---|
| `rmse` | Spatial and temporal root-mean-square error |
| `ssim` | Structural similarity (sharpness / long-term stability) |
| `acc` | Anomaly correlation (climatology from config, no hardcoded default) |
| `psd` | Power spectrum: FFT on lat-lon, `healpy.anafast` on HEALPix |
| `cov` | Pairwise Pearson correlation (historic name retained) |
| `rollout` | Long-term field-mean time series |
| `spatiotemporal_rollout` | Side-by-side GIF rollout |
| `variability` | Temporal standard-deviation maps |
| `spatial_agg` | Temporal-mean spatial maps |
| `seasonal_cycle` | Regional monthly cycles |
| `ipo` / `amo` | IPO and AMO indices |
| `vertical_zonal_mean` / `zonal_mean_profile` | Latitude–pressure and zonal profiles |

Collection configs:

- `config/collections/emulator/config-emulator-cesm2.yaml`
- `config/collections/emulator/config-emulator-ukesm1.yaml`
- `config/collections/emulator/config-emulator-ifsnemo.yaml`

## SSIM design (D4)

Uses `skimage.metrics.structural_similarity` with a common reference-period
`data_range`, Gaussian weights (`sigma=1.5`), temporal and spatial reductions,
and a latitude-weighted global mean of the local map. HEALPix sources require a
`regrid` target.

## PSD accepted limitation

Regular lat-lon grids use an isotropic 2-D FFT radial spectrum. True spherical
harmonics are produced only for HEALPix. Carry regular-grid spherical harmonics
into D393.2.4.2.

## CLI

```bash
python -m aqua.diagnostics.emulator.rmse.cli_rmse --config path/to/config.yaml
python -m aqua.diagnostics.emulator.ssim.cli_ssim --config path/to/config.yaml
```

Or drive the suite through `emueval` in `WP2-evaluation-framework`.

## Tests

```bash
pytest -m emulator --cov=aqua.diagnostics.emulator --cov-fail-under=70
```
