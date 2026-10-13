"""Temporal variability comparison."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray


class Variability(EmulatorDiagnostic):
    """Compare temporal standard-deviation fields."""

    def __init__(self, *args, variables=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "variability")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.variability = {}

    @staticmethod
    def compute(pred, ref):
        """Return prediction, reference, and difference standard deviations."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        pred_std = pred.std("time")
        ref_std = ref.std("time")
        return xr.Dataset(
            {
                "std_prediction": pred_std,
                "std_reference": ref_std,
                "std_difference": pred_std - ref_std,
            }
        )

    def evaluate(self, pred, ref, *, key="variable"):
        result = self.compute(pred, ref)
        result.attrs["variable"] = key
        self.variability[key] = result
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
                    "variability",
                    extra_keys={"var": var, "level": level},
                )
        return self.variability

    compute_variability = run
