"""Pairwise correlation diagnostic."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray


class COV(EmulatorDiagnostic):
    """Compare Pearson correlation maps for configured variable pairs."""

    def __init__(self, *args, variable_pairs=None, **kwargs):
        kwargs.setdefault("diagnostic_name", "cov")
        super().__init__(*args, **kwargs)
        self.variable_pairs = variable_pairs or []
        self.correlations = {}

    @staticmethod
    def correlation(da1, da2, reader=None):
        """Compute the Pearson correlation over time."""
        da1, da2 = EmulatorDiagnostic.align(da1, da2)
        if reader is None:
            mean1, mean2 = da1.mean("time"), da2.mean("time")
            std1, std2 = da1.std("time"), da2.std("time")
            numerator = ((da1 - mean1) * (da2 - mean2)).mean("time")
        else:
            mean1, mean2 = reader.timmean(da1), reader.timmean(da2)
            std1, std2 = reader.timstd(da1), reader.timstd(da2)
            numerator = reader.timmean((da1 - mean1) * (da2 - mean2))
        return (numerator / (std1 * std2)).rename("correlation")

    @staticmethod
    def _var_info(info):
        if isinstance(info, str):
            return info, None, info
        name = info["name"]
        level = info.get("level")
        level = int(level) if level is not None else None
        return name, level, name if level is None else f"{name}_{level}"

    def evaluate(self, pred1, pred2, ref1, ref2, *, pair_name="pair"):
        """Compute model, reference and model-minus-reference correlation."""
        corr_model = self.correlation(pred1, pred2, self.reader)
        corr_ref = self.correlation(ref1, ref2, self.reader_ref)
        result = xr.Dataset(
            {
                "correlation_prediction": corr_model,
                "correlation_reference": corr_ref,
                "correlation_difference": corr_model - corr_ref,
            }
        )
        self.correlations[pair_name] = result
        return result

    def run(
        self,
        variable_pairs=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        """Retrieve and compute all configured correlation pairs."""
        cache = {}
        for pair in variable_pairs or self.variable_pairs:
            var1, level1, key1 = self._var_info(pair["var1"])
            var2, level2, key2 = self._var_info(pair["var2"])
            for var, level, key in ((var1, level1, key1), (var2, level2, key2)):
                if key not in cache:
                    pred, ref = self.retrieve_pair(
                        var, level, reader_kwargs, ref_reader_kwargs
                    )
                    cache[key] = (
                        as_dataarray(pred, var),
                        as_dataarray(ref, var),
                    )
            pair_name = f"{key1}_vs_{key2}"
            result = self.evaluate(
                cache[key1][0],
                cache[key2][0],
                cache[key1][1],
                cache[key2][1],
                pair_name=pair_name,
            )
            if save_netcdf:
                self.save_pair_netcdf(
                    result, "correlation", extra_keys={"pair": pair_name}
                )
        return self.correlations
