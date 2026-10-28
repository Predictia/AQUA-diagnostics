"""Spatially averaged rollout comparison."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, fldmean_horizontal


class Rollout(EmulatorDiagnostic):
    """Compare global-mean prediction and reference trajectories."""

    def __init__(self, *args, variables=None, annual=False, **kwargs):
        kwargs.setdefault("diagnostic_name", "rollout")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.annual = annual.get("plot", True) if isinstance(annual, dict) else annual
        self.rollouts = {}

    @staticmethod
    def timeseries(pred, ref, pred_reader=None, ref_reader=None):
        """Return aligned AQUA field-mean trajectories."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        pred_mean = fldmean_horizontal(pred_reader, pred)
        ref_mean = fldmean_horizontal(ref_reader, ref)
        pred_mean, ref_mean = xr.align(pred_mean, ref_mean, join="inner")
        return xr.Dataset(
            {
                "prediction": pred_mean,
                "reference": ref_mean,
            }
        )

    @staticmethod
    def annual_timeseries(data):
        """Aggregate a rollout dataset to annual means."""
        return data.resample(time="YS").mean(skipna=True).dropna("time", how="all")

    def evaluate(self, pred, ref, *, key="variable"):
        result = self.timeseries(pred, ref, self.reader, self.reader_ref)
        result.attrs["variable"] = key
        self.rollouts[key] = result
        if self.annual:
            self.rollouts[f"{key}_annual"] = self.annual_timeseries(result)
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
            self.evaluate(
                as_dataarray(pred, var), as_dataarray(ref, var), key=key
            )
            if save_netcdf:
                for result_key in (key, f"{key}_annual"):
                    if result_key in self.rollouts:
                        product = (
                            "annual" if result_key.endswith("_annual") else "timeseries"
                        )
                        self.save_pair_netcdf(
                            self.rollouts[result_key],
                            product,
                            extra_keys={"var": var, "level": level},
                        )
        return self.rollouts
