"""Spatial climatology maps."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray


class SpatialAgg(EmulatorDiagnostic):
    """Compute time-mean prediction and reference maps."""

    def __init__(self, *args, variables=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "spatial_agg")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.climatologies = {}

    @staticmethod
    def compute(pred, ref):
        """Compute whole-period temporal means."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        pred_mean = pred.mean("time") if "time" in pred.dims else pred
        ref_mean = ref.mean("time") if "time" in ref.dims else ref
        return xr.Dataset(
            {
                "prediction": pred_mean,
                "reference": ref_mean,
                "difference": pred_mean - ref_mean,
            }
        )

    def evaluate(self, pred, ref, *, key="variable"):
        result = self.compute(pred, ref)
        result.attrs["variable"] = key
        self.climatologies[key] = result
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
            result = self.evaluate(
                as_dataarray(pred, var), as_dataarray(ref, var), key=key
            )
            if save_netcdf:
                self.save_pair_netcdf(
                    result,
                    "climatology",
                    extra_keys={"var": var, "level": level},
                )
        return self.climatologies
