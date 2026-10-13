"""Synthetic numerical tests for RMSE and SSIM (KPI #9)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.emulator.rmse.rmse import RMSE
from aqua.diagnostics.emulator.ssim.ssim import SSIM


pytestmark = pytest.mark.emulator


def _pair(nt=5, nlat=8, nlon=10, noise=0.0, smooth=False, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(nt)
    lat = np.linspace(-75, 75, nlat)
    lon = np.linspace(0, 324, nlon)
    base = rng.normal(size=(nt, nlat, nlon))
    pred_vals = base + noise * rng.normal(size=base.shape)
    if smooth:
        padded = np.pad(pred_vals, ((0, 0), (1, 1), (1, 1)), mode="edge")
        pred_vals = (
            padded[:, 0:-2, 0:-2]
            + padded[:, 0:-2, 1:-1]
            + padded[:, 0:-2, 2:]
            + padded[:, 1:-1, 0:-2]
            + padded[:, 1:-1, 1:-1]
            + padded[:, 1:-1, 2:]
            + padded[:, 2:, 0:-2]
            + padded[:, 2:, 1:-1]
            + padded[:, 2:, 2:]
        ) / 9.0
    coords = {"time": time, "lat": lat, "lon": lon}
    pred = xr.DataArray(pred_vals, dims=("time", "lat", "lon"), coords=coords, name="2t")
    ref = xr.DataArray(base, dims=("time", "lat", "lon"), coords=coords, name="2t")
    return pred, ref


def test_rmse_identical_fields_are_zero():
    pred, ref = _pair(noise=0.0)
    spatial = RMSE.spatial(pred, ref)
    temporal = RMSE.temporal(pred, ref, reader=None)
    assert float(spatial.max()) == pytest.approx(0.0, abs=1e-12)
    assert float(temporal.max()) == pytest.approx(0.0, abs=1e-12)


def test_rmse_increases_with_noise():
    pred0, ref = _pair(noise=0.0, seed=1)
    pred1, _ = _pair(noise=0.5, seed=1)
    pred2, _ = _pair(noise=1.5, seed=1)
    r0 = float(RMSE.temporal(pred0, ref, reader=None).mean())
    r1 = float(RMSE.temporal(pred1, ref, reader=None).mean())
    r2 = float(RMSE.temporal(pred2, ref, reader=None).mean())
    assert r0 < r1 < r2


def test_ssim_identical_fields_are_one():
    pred, ref = _pair(noise=0.0)
    temporal, spatial = SSIM.compute(pred, ref)
    global_score = SSIM.global_from_local(spatial)
    assert float(temporal.min()) == pytest.approx(1.0, abs=1e-6)
    assert float(spatial.min()) == pytest.approx(1.0, abs=1e-6)
    assert float(global_score) == pytest.approx(1.0, abs=1e-6)


def test_ssim_degrades_with_noise_and_smoothing():
    pred0, ref = _pair(noise=0.0, seed=2)
    pred_noise, _ = _pair(noise=1.0, seed=2)
    pred_smooth, _ = _pair(noise=0.0, smooth=True, seed=2)
    _, spatial0 = SSIM.compute(pred0, ref)
    _, spatial_noise = SSIM.compute(pred_noise, ref)
    _, spatial_smooth = SSIM.compute(pred_smooth, ref)
    g0 = float(SSIM.global_from_local(spatial0))
    g_noise = float(SSIM.global_from_local(spatial_noise))
    g_smooth = float(SSIM.global_from_local(spatial_smooth))
    assert g_noise < g0
    assert g_smooth < g0


def test_ssim_requires_latlon_grid():
    values = np.ones((3, 16))
    pred = xr.DataArray(values, dims=("time", "ncells"), name="2t")
    ref = pred.copy()
    with pytest.raises(ValueError, match="latitude/longitude"):
        SSIM.compute(pred, ref)
