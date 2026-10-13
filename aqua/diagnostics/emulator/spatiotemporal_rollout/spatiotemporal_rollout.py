"""Frame-wise spatial rollout data preparation."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, ensure_latlon_regrid


class SpatiotemporalRollout(EmulatorDiagnostic):
    """Prepare aligned reference/prediction fields for rollout animations."""

    def __init__(self, *args, variables=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "spatiotemporal_rollout")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.rollouts = {}

    @staticmethod
    def prepare(pred, ref):
        """Align frames and return a two-variable rollout dataset."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        if "time" not in pred.dims or pred.sizes["time"] == 0:
            raise ValueError("spatiotemporal rollout requires overlapping time data")
        return xr.Dataset(
            {
                "prediction": pred.transpose("time", ...),
                "reference": ref.transpose("time", ...),
            }
        )

    def evaluate(self, pred, ref, *, key="variable"):
        ensure_latlon_regrid(self.regrid, pred)
        ensure_latlon_regrid(self.ref_regrid, ref)
        result = self.prepare(pred, ref)
        result.attrs["variable"] = key
        self.rollouts[key] = result
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
                    "frames",
                    extra_keys={"var": var, "level": level},
                )
        return self.rollouts

    generate_videos = run
