"""Interdecadal Pacific Oscillation tripole index."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import cheby1, filtfilt
from aqua.core.fldstat.area_selection import AreaSelection

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, fldmean_horizontal


class IPO(EmulatorDiagnostic):
    """Compute the Henley et al. (2015) Tripole Index."""

    def __init__(
        self,
        *args,
        field="tos",
        regions=None,
        window=13,
        filter_type="chebyshev",
        cutoff_freq=None,
        filter_order=4,
        ripple_db=1.0,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "ipo")
        super().__init__(*args, **kwargs)
        self.field = field
        self.regions = regions or {
            "southwest": {"lat": [-45, -10], "lon": [140, 220]},
            "equatorial": {"lat": [-10, 10], "lon": [170, 270]},
            "northwest": {"lat": [25, 45], "lon": [140, 215]},
        }
        self.window = window
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq or 1.0 / (window * 12)
        self.filter_order = filter_order
        self.ripple_db = ripple_db
        self.index = self.index_ref = None
        self.index_filtered = self.index_filtered_ref = None

    @staticmethod
    def monthly(data):
        return data.resample(time="MS").mean(skipna=True)

    @staticmethod
    def anomalies(data):
        return data.groupby("time.month") - data.groupby("time.month").mean("time")

    @staticmethod
    def _region_mean(data, region, reader=None):
        subset = AreaSelection().select_area(
            data, lat=region["lat"], lon=region["lon"], drop=False
        )
        return fldmean_horizontal(reader, subset)

    @classmethod
    def tripole_index(cls, data, regions, reader=None, name="tpi"):
        """Compute the tripole index from three regional means."""
        anomalies = cls.anomalies(data)
        southwest = cls._region_mean(anomalies, regions["southwest"], reader)
        equatorial = cls._region_mean(anomalies, regions["equatorial"], reader)
        northwest = cls._region_mean(anomalies, regions["northwest"], reader)
        return (equatorial - 0.5 * (southwest + northwest)).rename(name)

    def apply_filter(self, index):
        """Apply the configured rolling or Chebyshev low-pass filter."""
        if not self.window:
            return index
        if self.filter_type == "rolling":
            width = self.window * 12
            return index.rolling(
                time=width, center=True, min_periods=max(1, width // 2)
            ).mean().rename(f"{index.name}_filtered")
        if self.filter_type != "chebyshev":
            raise ValueError(f"unknown IPO filter type: {self.filter_type}")
        normalized = min(self.cutoff_freq / 0.5, 0.9)
        b, a = cheby1(
            self.filter_order, self.ripple_db, normalized, btype="low"
        )
        values = filtfilt(b, a, index.values)
        return xr.DataArray(
            values,
            dims=index.dims,
            coords=index.coords,
            attrs=index.attrs,
            name=f"{index.name}_filtered",
        )

    @staticmethod
    def stat_maps(data, index):
        """Return regression and Pearson-correlation maps."""
        data, index = xr.align(data, index, join="inner")
        data_anom = IPO.anomalies(data)
        regression = xr.cov(index, data_anom, dim="time") / index.var(
            "time", skipna=True
        )
        correlation = xr.corr(index, data_anom, dim="time")
        return regression, correlation

    def evaluate(self, pred, ref):
        pred = self.monthly(pred)
        ref = self.monthly(ref)
        self.index = self.tripole_index(
            pred, self.regions, self.reader, "tpi"
        )
        self.index_ref = self.tripole_index(
            ref, self.regions, self.reader_ref, "tpi_ref"
        )
        self.index_filtered = self.apply_filter(self.index)
        self.index_filtered_ref = self.apply_filter(self.index_ref)
        return xr.Dataset(
            {
                "prediction": self.index,
                "reference": self.index_ref,
                "prediction_filtered": self.index_filtered,
                "reference_filtered": self.index_filtered_ref,
            }
        )

    def run(
        self,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        pred, ref = self.retrieve_pair(
            self.field,
            reader_kwargs=reader_kwargs,
            ref_reader_kwargs=ref_reader_kwargs,
        )
        result = self.evaluate(
            as_dataarray(pred, self.field), as_dataarray(ref, self.field)
        )
        if save_netcdf:
            self.save_pair_netcdf(
                result, "index", extra_keys={"var": self.field}
            )
        return result
