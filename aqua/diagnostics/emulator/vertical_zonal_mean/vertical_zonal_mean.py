"""Latitude-pressure zonal-mean diagnostic."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, temporal_stat


class VerticalZonalMean(EmulatorDiagnostic):
    """Compare temporally aggregated latitude-pressure cross-sections."""

    def __init__(
        self,
        *args,
        variables=None,
        aggregation="mean",
        percentile=99.9,
        latitude_bins=None,
        significance=False,
        significance_alpha=0.05,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "vertical_zonal_mean")
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
        self.latitude_bins = latitude_bins
        self.significance = significance
        self.significance_alpha = significance_alpha
        self.vertical_profiles = {}

    @staticmethod
    def level_coord(data):
        for name in ("plev", "level", "lev", "pressure", "p"):
            if name in data.dims or name in data.coords:
                return name
        return None

    @classmethod
    def common_levels(cls, pred, ref):
        pred_coord, ref_coord = cls.level_coord(pred), cls.level_coord(ref)
        if pred_coord is None or ref_coord is None:
            raise ValueError("vertical zonal mean requires pressure levels")
        levels = sorted(
            set(pred[pred_coord].values) & set(ref[ref_coord].values),
            reverse=True,
        )
        if not levels:
            raise ValueError("prediction and reference have no common pressure levels")
        pred = pred.sel({pred_coord: levels})
        ref = ref.sel({ref_coord: levels})
        if pred_coord != ref_coord:
            pred = pred.rename({pred_coord: ref_coord})
        return pred, ref, ref_coord

    @staticmethod
    def zonal_mean(data):
        lon = next((dim for dim in data.dims if "lon" in dim.lower()), None)
        if lon is None:
            raise ValueError("unable to determine longitude dimension")
        return data.mean(lon)

    @staticmethod
    def bin_latitude(data, bins):
        lat = next((dim for dim in data.dims if "lat" in dim.lower()), None)
        if lat is None:
            raise ValueError("unable to determine latitude dimension")
        edges = np.linspace(-90, 90, int(bins) + 1)
        labels = (edges[:-1] + edges[1:]) / 2
        return data.groupby_bins(lat, edges, labels=labels).mean().rename(
            {f"{lat}_bins": lat}
        )

    @staticmethod
    def _welch(model, reference, min_samples=3):
        model = model[np.isfinite(model)]
        reference = reference[np.isfinite(reference)]
        if len(model) < min_samples or len(reference) < min_samples:
            return np.nan
        return stats.ttest_ind(model, reference, equal_var=False).pvalue

    @classmethod
    def significance_mask(cls, pred, ref, alpha=0.05):
        """Welch t-test on yearly zonal means."""
        pred = cls.zonal_mean(pred).resample(time="YS").mean(skipna=True)
        ref = cls.zonal_mean(ref).resample(time="YS").mean(skipna=True)
        pred = pred.rename(time="time_prediction").chunk(
            {"time_prediction": -1}
        )
        ref = ref.rename(time="time_reference").chunk({"time_reference": -1})
        pvalues = xr.apply_ufunc(
            cls._welch,
            pred,
            ref,
            input_core_dims=[["time_prediction"], ["time_reference"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return (pvalues < alpha).fillna(False).rename("significance_mask")

    def compute(self, pred, ref):
        """Compute paired cross-sections and optional significance."""
        pred, ref, level = self.common_levels(pred, ref)
        pred, ref = self.align(pred, ref)
        mask = None
        if self.significance:
            mask = self.significance_mask(
                pred, ref, alpha=self.significance_alpha
            )
        pred_zonal = self.zonal_mean(
            temporal_stat(pred, self.aggregation_stat, self.percentile)
        )
        ref_zonal = self.zonal_mean(
            temporal_stat(ref, self.aggregation_stat, self.percentile)
        )
        if self.latitude_bins:
            pred_zonal = self.bin_latitude(pred_zonal, self.latitude_bins)
            ref_zonal = self.bin_latitude(ref_zonal, self.latitude_bins)
            if mask is not None:
                mask = self.bin_latitude(mask.astype(float), self.latitude_bins) > 0.5
        result = xr.Dataset(
            {
                "zonal_mean_prediction": pred_zonal,
                "zonal_mean_reference": ref_zonal,
                "zonal_mean_difference": pred_zonal - ref_zonal,
            }
        )
        if mask is not None:
            result["significance_mask"] = mask
        result.attrs.update(
            level_coordinate=level,
            aggregation_stat=self.aggregation_stat,
        )
        return result

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        for var, _, key in self.iter_variables(variables or self.variables):
            if key != var:
                self.logger.warning(
                    "VerticalZonalMean retrieves all levels; ignoring level in %s",
                    key,
                )
            pred, ref = self.retrieve_pair(
                var,
                reader_kwargs=reader_kwargs,
                ref_reader_kwargs=ref_reader_kwargs,
            )
            result = self.compute(
                as_dataarray(pred, var), as_dataarray(ref, var)
            )
            self.vertical_profiles[var] = result
            if save_netcdf:
                self.save_pair_netcdf(
                    result,
                    "vertical_zonal_mean",
                    extra_keys={"var": var, "stat": self.aggregation_stat},
                )
        return self.vertical_profiles
