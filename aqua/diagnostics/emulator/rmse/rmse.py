"""Root-mean-square error diagnostic."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, fldmean_horizontal


class RMSE(EmulatorDiagnostic):
    """Compute spatial and temporal RMSE for emulator predictions."""

    def __init__(self, *args, variables=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "rmse")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.spatial_rmse = {}
        self.temporal_rmse = {}

    @staticmethod
    def spatial(pred, ref):
        """Return ``sqrt(mean_time((prediction-reference)**2))``."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        squared = (pred - ref) ** 2
        return np.sqrt(squared.mean("time")) if "time" in squared.dims else np.sqrt(squared)

    @staticmethod
    def temporal(pred, ref, reader=None):
        """Return ``sqrt(field_mean((prediction-reference)**2))``."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        return np.sqrt(fldmean_horizontal(reader, (pred - ref) ** 2))

    def evaluate_spatial(self, pred=None, ref=None, var=None, level=None, key=None):
        """Compute one spatial RMSE map and store it by variable key."""
        if pred is None or ref is None:
            if var is None:
                raise ValueError("var is required when data are not supplied")
            pred, ref = self.retrieve_pair(var, level=level)
        pred = as_dataarray(pred, var)
        ref = as_dataarray(ref, var)
        key = key or (var if level is None else f"{var}_{int(level)}")
        result = self.spatial(pred, ref).rename("rmse")
        result.attrs.update(
            long_name="spatial root mean square error",
            diagnostic="RMSE",
        )
        self.spatial_rmse[key] = result
        return result

    def evaluate_temporal(self, pred=None, ref=None, var=None, level=None, key=None):
        """Compute one temporal RMSE series and store it by variable key."""
        if pred is None or ref is None:
            if var is None:
                raise ValueError("var is required when data are not supplied")
            pred, ref = self.retrieve_pair(var, level=level)
        pred = as_dataarray(pred, var)
        ref = as_dataarray(ref, var)
        key = key or (var if level is None else f"{var}_{int(level)}")
        result = self.temporal(pred, ref, self.reader).rename("rmse")
        result.attrs.update(
            long_name="temporal root mean square error",
            diagnostic="RMSE",
        )
        self.temporal_rmse[key] = result
        return result

    # Frontier-compatible names.
    spatial_rmse_for_pair = evaluate_spatial
    temporal_rmse_for_pair = evaluate_temporal

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        """Retrieve configured variables and compute both RMSE products."""
        for var, level, key in self.iter_variables(variables or self.variables):
            pred, ref = self.retrieve_pair(
                var,
                level=level,
                reader_kwargs=reader_kwargs,
                ref_reader_kwargs=ref_reader_kwargs,
            )
            spatial = self.evaluate_spatial(pred, ref, var, level, key)
            temporal = self.evaluate_temporal(pred, ref, var, level, key)
            if save_netcdf:
                extra = {"var": var, "level": level}
                self.save_pair_netcdf(
                    spatial.to_dataset(), "spatial", extra_keys=extra
                )
                self.save_pair_netcdf(
                    temporal.to_dataset(), "temporal", extra_keys=extra
                )
        return {
            "spatial": self.spatial_rmse,
            "temporal": self.temporal_rmse,
        }
