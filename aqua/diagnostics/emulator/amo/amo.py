"""Atlantic Multidecadal Oscillation index."""

from __future__ import annotations

import xarray as xr
from aqua.core.fldstat.area_selection import AreaSelection

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, fldmean_horizontal


class AMO(EmulatorDiagnostic):
    """Compute the Trenberth-Shea regional-minus-global AMO index."""

    def __init__(
        self,
        *args,
        field="tos",
        region=None,
        detrend=True,
        detrend_variable="tos",
        smooth=True,
        window=5,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "amo")
        super().__init__(*args, **kwargs)
        self.field = field
        self.amo_region = region or {"lat": [25, 60], "lon": [-75, -7]}
        self.apply_detrending = detrend
        self.detrending_variable = detrend_variable
        self.apply_smoothing = smooth
        self.window = window
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
        selected = AreaSelection().select_area(
            data, lat=region["lat"], lon=region["lon"], drop=False
        )
        return fldmean_horizontal(reader, selected)

    @classmethod
    def amo_index(
        cls,
        data,
        region,
        reader=None,
        *,
        global_data=None,
        detrend=True,
        name="amo",
    ):
        """Apply the unchanged regional anomaly minus global anomaly core."""
        anomalies = cls.anomalies(data)
        index = cls._region_mean(anomalies, region, reader)
        if detrend:
            global_anomalies = (
                cls.anomalies(global_data) if global_data is not None else anomalies
            )
            index = index - fldmean_horizontal(reader, global_anomalies)
        return index.rename(name)

    def apply_filter(self, index):
        width = self.window * 12
        return index.rolling(
            time=width, center=True, min_periods=max(1, width // 2)
        ).mean().rename(f"{index.name}_filtered")

    _apply_filter = apply_filter

    @staticmethod
    def stat_maps(data, index):
        data, index = xr.align(data, index, join="inner")
        anomalies = AMO.anomalies(data)
        regression = xr.cov(index, anomalies, dim="time") / index.var(
            "time", skipna=True
        )
        return regression, xr.corr(index, anomalies, dim="time")

    def evaluate(self, pred, ref, *, pred_global=None, ref_global=None):
        pred, ref = self.monthly(pred), self.monthly(ref)
        if pred_global is not None:
            pred_global = self.monthly(pred_global)
        if ref_global is not None:
            ref_global = self.monthly(ref_global)
        self.index = self.amo_index(
            pred,
            self.amo_region,
            self.reader,
            global_data=pred_global,
            detrend=self.apply_detrending,
            name="amo",
        )
        self.index_ref = self.amo_index(
            ref,
            self.amo_region,
            self.reader_ref,
            global_data=ref_global,
            detrend=self.apply_detrending,
            name="amo_ref",
        )
        self.index_filtered = (
            self.apply_filter(self.index) if self.apply_smoothing else self.index
        )
        self.index_filtered_ref = (
            self.apply_filter(self.index_ref)
            if self.apply_smoothing
            else self.index_ref
        )
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
        pred_global = ref_global = None
        if self.apply_detrending and self.detrending_variable != self.field:
            pred_global_ds, ref_global_ds = self.retrieve_pair(
                self.detrending_variable,
                reader_kwargs=reader_kwargs,
                ref_reader_kwargs=ref_reader_kwargs,
            )
            pred_global = as_dataarray(pred_global_ds, self.detrending_variable)
            ref_global = as_dataarray(ref_global_ds, self.detrending_variable)
        result = self.evaluate(
            as_dataarray(pred, self.field),
            as_dataarray(ref, self.field),
            pred_global=pred_global,
            ref_global=ref_global,
        )
        if save_netcdf:
            self.save_pair_netcdf(
                result, "index", extra_keys={"var": self.field}
            )
        return result

    retrieve = run
    compute_index = evaluate
