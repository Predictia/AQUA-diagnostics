"""Utilities shared by emulator diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr
from smmregrid import GridInspector


def parse_variable_levels(var_info):
    """Normalize variable configuration into ``(name, level, key)`` tuples."""
    from .base import EmulatorDiagnostic

    return list(EmulatorDiagnostic.iter_variables(var_info))


def ensure_latlon_regrid(regrid, data):
    """Validate that image-based diagnostics receive a latitude/longitude grid.

    HEALPix arrays (usually carrying ``ncells`` or ``cell``) cannot be treated
    as rectangular images.  A configured regrid is therefore mandatory.
    """
    dims = set(data.dims)
    if dims.intersection({"ncells", "cell"}) and not regrid:
        raise ValueError(
            "This diagnostic requires a regular latitude/longitude grid for "
            "HEALPix data; configure 'regrid' for prediction and reference."
        )
    return data


def horizontal_dims(da):
    """Return horizontal dimensions inferred by :class:`GridInspector`."""
    if isinstance(da, xr.Dataset):
        if not da.data_vars:
            return []
        inspected = next(iter(da.data_vars.values()))
    else:
        inspected = da

    try:
        grid_types = GridInspector(inspected).get_gridtype()
        if grid_types:
            dims = [
                dim
                for dim in grid_types[0].horizontal_dims
                if dim in inspected.dims
            ]
            if dims:
                return dims
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    named = [
        dim
        for dim in inspected.dims
        if "lat" in dim.lower() or "lon" in dim.lower()
    ]
    if named:
        return named
    for candidate in (("ncells",), ("cell",), ("y", "x")):
        if all(dim in inspected.dims for dim in candidate):
            return list(candidate)
    return [
        dim
        for dim in inspected.dims
        if dim.lower() not in {"time", "plev", "lev", "level", "pressure"}
    ]


def fldmean_horizontal(reader, da):
    """Compute AQUA's field mean using dimensions inferred from the data."""
    dims = horizontal_dims(da)
    if reader is None:
        lat_dims = [dim for dim in dims if "lat" in dim.lower()]
        if len(lat_dims) == 1 and da[lat_dims[0]].ndim == 1:
            weights = np.cos(np.deg2rad(da[lat_dims[0]]))
            result = da.weighted(weights).mean(dims, skipna=True)
        else:
            result = da.mean(dims, skipna=True)
        return result

    fldstat = getattr(reader, "tgt_fldstat", None) or getattr(
        reader, "src_fldstat", None
    )
    if fldstat is not None and dims:
        fldstat.horizontal_dims = dims
    kwargs = {"dims": dims} if dims else {}
    result = reader.fldmean(da, **kwargs)
    for dim in dims:
        if dim in result.dims:
            result = result.mean(dim, skipna=True)
    return result


def as_dataarray(data, var=None):
    """Extract a DataArray from a DataArray or Dataset."""
    if isinstance(data, xr.DataArray):
        return data
    if not isinstance(data, xr.Dataset):
        raise TypeError("expected an xarray Dataset or DataArray")
    if var is not None and var in data:
        return data[var]
    if len(data.data_vars) == 1:
        return next(iter(data.data_vars.values()))
    raise KeyError(f"variable {var!r} is not available in {list(data.data_vars)}")


def temporal_stat(data, stat="mean", percentile=99.9):
    """Aggregate a field over time."""
    if "time" not in data.dims:
        return data
    if stat == "percentile":
        result = data.quantile(percentile / 100.0, dim="time", skipna=True)
        return result.drop_vars("quantile") if "quantile" in result.coords else result
    if stat not in {"mean", "std", "max", "min"}:
        raise ValueError(f"unsupported temporal statistic: {stat}")
    return getattr(data, stat)(dim="time", skipna=True)


def normalize_regions(regions):
    """Normalize list/mapping region configuration."""
    if regions is None:
        return [{"name": "global", "lat": None, "lon": None}]
    normalized = []
    entries = regions.items() if isinstance(regions, Mapping) else enumerate(regions)
    for key, value in entries:
        if not isinstance(value, Mapping):
            continue
        normalized.append(
            {
                "name": value.get("name", str(key)),
                "lat": value.get("lat", value.get("lat_limits")),
                "lon": value.get("lon", value.get("lon_limits")),
            }
        )
    return normalized or [{"name": "global", "lat": None, "lon": None}]
