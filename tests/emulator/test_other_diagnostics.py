"""Smoke tests for remaining emulator numerical cores."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.emulator.acc.acc import ACC
from aqua.diagnostics.emulator.cov.cov import COV
from aqua.diagnostics.emulator.psd.psd import PSD
from aqua.diagnostics.emulator.variability.variability import Variability


pytestmark = pytest.mark.emulator


def _field(seed=0, nt=6, nlat=8, nlon=10):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(nt, nlat, nlon)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(nt),
            "lat": np.linspace(-60, 60, nlat),
            "lon": np.linspace(0, 330, nlon),
        },
        name="2t",
    )


def test_acc_perfect_anomaly_correlation():
    field = _field(seed=3)
    clim = field.mean("time")
    spatial = ACC.spatial(field, field, clim)
    temporal = ACC.temporal(field, field, clim)
    assert float(spatial.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(temporal.mean()) == pytest.approx(1.0, abs=1e-6)


def test_cov_self_correlation_is_one():
    field = _field(seed=4)
    corr = COV.correlation(field, field)
    assert float(corr.mean()) == pytest.approx(1.0, abs=1e-6)


def test_psd_fft_returns_positive_spectrum():
    field = _field(seed=5)
    spectrum = PSD.fft_spectrum(field, average_time=True)
    assert np.all(np.asarray(spectrum) >= 0)


def test_variability_difference_shape():
    pred = _field(seed=6)
    ref = _field(seed=7)
    result = Variability.compute(pred, ref)
    assert set(result.data_vars) >= {
        "std_prediction",
        "std_reference",
        "std_difference",
    }
    assert result["std_difference"].dims == ("lat", "lon")
