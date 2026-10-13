"""One-dimensional zonal-mean profiles."""

from __future__ import annotations

import xarray as xr
from aqua.core.fixer import EvaluateFormula

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, temporal_stat


class ZonalMeanProfile(EmulatorDiagnostic):
    """Compare latitude profiles after temporal and zonal aggregation."""

    def __init__(
        self,
        *args,
        variables=None,
        aggregation="mean",
        percentile=99.9,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "zonal_mean_profile")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.aggregation_stat = (
            aggregation.get("stat", "mean")
            if isinstance(aggregation, dict)
            else aggregation
        )
        self.percentile = (
            aggregation.get("percentile", percentile)
            if isinstance(aggregation, dict)
            else percentile
        )
        self.profiles = {}

    @staticmethod
    def zonal_mean(data, stat="mean", percentile=99.9):
        """Apply the frontier temporal statistic then average longitude."""
        aggregated = temporal_stat(data, stat, percentile)
        lon = next(
            (dim for dim in aggregated.dims if "lon" in dim.lower()), None
        )
        if lon is None:
            raise ValueError("unable to determine longitude dimension")
        return aggregated.mean(lon).squeeze()

    compute_zonal_mean = zonal_mean

    def compute(self, pred, ref):
        pred, ref = self.align(pred, ref)
        pred_profile = self.zonal_mean(
            pred, self.aggregation_stat, self.percentile
        )
        ref_profile = self.zonal_mean(
            ref, self.aggregation_stat, self.percentile
        )
        return xr.Dataset(
            {
                "zonal_mean_prediction": pred_profile,
                "zonal_mean_reference": ref_profile,
                "zonal_mean_difference": pred_profile - ref_profile,
            },
            attrs={"aggregation_stat": self.aggregation_stat},
        )

    def _retrieve_formula(
        self,
        info,
        reader_kwargs=None,
        ref_reader_kwargs=None,
    ):
        pred, ref = self.retrieve_pair(
            None,
            reader_kwargs=reader_kwargs,
            ref_reader_kwargs=ref_reader_kwargs,
        )
        kwargs = {
            "formula": info["name"],
            "long_name": info.get("long_name"),
            "short_name": info.get("standard_name", info["name"]),
            "units": info.get("units"),
            "loglevel": self.loglevel,
        }
        pred_result = EvaluateFormula(data=pred, **kwargs).evaluate()
        ref_result = EvaluateFormula(data=ref, **kwargs).evaluate()
        if pred_result is None or ref_result is None:
            raise ValueError(f"failed to evaluate formula {info['name']}")
        return pred_result, ref_result

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        for info in variables or self.variables:
            info = {"name": info} if isinstance(info, str) else info
            var = info["name"]
            if info.get("formula", False):
                pred_da, ref_da = self._retrieve_formula(
                    info, reader_kwargs, ref_reader_kwargs
                )
            else:
                pred, ref = self.retrieve_pair(
                    var,
                    reader_kwargs=reader_kwargs,
                    ref_reader_kwargs=ref_reader_kwargs,
                )
                pred_da, ref_da = as_dataarray(pred, var), as_dataarray(ref, var)
            result = self.compute(pred_da, ref_da)
            result.attrs["variable"] = var
            self.profiles[var] = result
            if save_netcdf:
                self.save_pair_netcdf(
                    result,
                    "zonal_mean_profile",
                    extra_keys={"var": var, "stat": self.aggregation_stat},
                )
        return self.profiles

    compute_profile = run
