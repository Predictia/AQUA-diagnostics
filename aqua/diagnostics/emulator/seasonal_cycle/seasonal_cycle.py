"""Regional seasonal-cycle comparison."""

from __future__ import annotations

import xarray as xr
from aqua.core.fldstat.area_selection import AreaSelection

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, fldmean_horizontal, normalize_regions


class SeasonalCycle(EmulatorDiagnostic):
    """Compute monthly regional means and anomalies."""

    def __init__(self, *args, variables=None, regions=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "seasonal_cycle")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.regions = normalize_regions(regions)
        self.seasonal_cycles = {}

    @staticmethod
    def cycle(pred_series, ref_series):
        """Compute the unchanged monthly-climatology/anomaly core."""
        pred_series, ref_series = EmulatorDiagnostic.align(pred_series, ref_series)
        pred_monthly = pred_series.resample(time="MS").mean(skipna=True)
        ref_monthly = ref_series.resample(time="MS").mean(skipna=True)
        pred_monthly, ref_monthly = xr.align(
            pred_monthly, ref_monthly, join="inner"
        )
        if pred_monthly.sizes.get("time", 0) < 12:
            raise ValueError("seasonal cycle requires at least 12 monthly means")
        pred_cycle = pred_monthly.groupby("time.month").mean("time")
        ref_cycle = ref_monthly.groupby("time.month").mean("time")
        return xr.Dataset(
            {
                "prediction": pred_cycle,
                "reference": ref_cycle,
                "prediction_anomaly": pred_cycle - pred_cycle.mean("month"),
                "reference_anomaly": ref_cycle - ref_cycle.mean("month"),
            }
        )

    @staticmethod
    def _regional_series(data, region, reader=None):
        if region.get("lat") is not None or region.get("lon") is not None:
            data = AreaSelection().select_area(
                data,
                lat=region.get("lat"),
                lon=region.get("lon"),
                drop=False,
            )
        return fldmean_horizontal(reader, data).squeeze()

    def evaluate(self, pred, ref, *, key="variable", region=None):
        region = region or self.regions[0]
        pred_series = self._regional_series(pred, region, self.reader)
        ref_series = self._regional_series(ref, region, self.reader_ref)
        result = self.cycle(pred_series, ref_series)
        result.attrs.update(variable=key, region=region["name"])
        output_key = f"{key}__{region['name']}"
        self.seasonal_cycles[output_key] = result
        return result

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        for var, level, key in self.iter_variables(variables or self.variables):
            pred, ref = self.retrieve_pair(
                var, level, reader_kwargs, ref_reader_kwargs
            )
            pred_da, ref_da = as_dataarray(pred, var), as_dataarray(ref, var)
            for region in self.regions:
                result = self.evaluate(
                    pred_da, ref_da, key=key, region=region
                )
                if save_netcdf:
                    self.save_pair_netcdf(
                        result,
                        "seasonal_cycle",
                        extra_keys={
                            "var": var,
                            "level": level,
                            "region": region["name"],
                        },
                    )
        return self.seasonal_cycles

    compute = run
