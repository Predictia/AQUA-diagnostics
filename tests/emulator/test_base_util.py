"""Unit tests for EmulatorDiagnostic helpers (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.emulator.base import EmulatorDiagnostic
from aqua.diagnostics.emulator.util import (
    as_dataarray,
    ensure_latlon_regrid,
    horizontal_dims,
    normalize_regions,
    parse_variable_levels,
)


pytestmark = pytest.mark.emulator


def _toy_field(name="2t", nt=4, nlat=6, nlon=8, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(nt)
    lat = np.linspace(-60, 60, nlat)
    lon = np.linspace(0, 330, nlon)
    data = rng.normal(size=(nt, nlat, nlon))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=name,
    )


def test_iter_variables_normalizes_levels():
    assert list(EmulatorDiagnostic.iter_variables("2t")) == [("2t", None, "2t")]
    assert list(EmulatorDiagnostic.iter_variables({"name": "zg", "level": 50000})) == [
        ("zg", 50000, "zg_50000")
    ]
    assert list(
        EmulatorDiagnostic.iter_variables([{"name": "zg", "level": [50000, 85000]}])
    ) == [
        ("zg", 50000, "zg_50000"),
        ("zg", 85000, "zg_85000"),
    ]


def test_parse_variable_levels_matches_iter():
    assert parse_variable_levels({"name": "2t"}) == [("2t", None, "2t")]


def test_align_intersects_time_and_casts_coords():
    pred = _toy_field(seed=1).isel(time=slice(0, 3)).to_dataset()
    ref = _toy_field(seed=2).isel(time=slice(1, 4)).to_dataset()
    pred["lat"] = pred["lat"].astype(np.float64)
    ref["lat"] = ref["lat"].astype(np.float32)
    out_pred, out_ref = EmulatorDiagnostic.align(pred, ref)
    assert out_pred.sizes["time"] == 2
    assert out_ref.sizes["time"] == 2
    assert out_pred["lat"].dtype == np.float32
    assert out_ref["lat"].dtype == np.float32


def test_ensure_latlon_regrid_requires_target_for_healpix():
    healpix = xr.DataArray(np.ones((2, 12)), dims=("time", "ncells"), name="2t")
    with pytest.raises(ValueError, match="regrid"):
        ensure_latlon_regrid(None, healpix)
    assert ensure_latlon_regrid("r050", healpix) is healpix


def test_horizontal_dims_and_as_dataarray():
    field = _toy_field()
    assert set(horizontal_dims(field)) == {"lat", "lon"}
    ds = field.to_dataset()
    assert as_dataarray(ds, "2t").name == "2t"


def test_normalize_regions():
    regions = normalize_regions(
        [{"name": "tropics", "lat": [-20, 20]}, {"name": "global"}]
    )
    assert regions[0]["name"] == "tropics"
    assert normalize_regions(None)[0]["name"] == "global"
