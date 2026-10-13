"""Structural Similarity Index Measure for emulator fields."""

from __future__ import annotations

import numpy as np
import xarray as xr
from skimage.metrics import structural_similarity

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, ensure_latlon_regrid, horizontal_dims


class SSIM(EmulatorDiagnostic):
    """Compute temporal, local spatial, and global SSIM.

    SSIM is evaluated independently at each time step using a data range
    derived from the reference over the complete analysis period.  The local
    SSIM maps are averaged over time.  ``global_ssim`` is the latitude-weighted
    mean of that time-mean local map (an unweighted mean is used when no
    one-dimensional latitude coordinate exists).
    """

    def __init__(
        self,
        *args,
        variables=None,
        win_size=None,
        sigma=1.5,
        data_range=None,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "ssim")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.win_size = win_size
        self.sigma = sigma
        self.data_range = data_range
        self.temporal_ssim = {}
        self.spatial_ssim = {}
        self.global_ssim = {}

    @staticmethod
    def reference_data_range(ref, configured=None):
        """Return the common SSIM data range for the analysis period."""
        if configured is not None:
            value = float(configured)
        else:
            value = float(ref.max(skipna=True) - ref.min(skipna=True))
        if not np.isfinite(value) or value < 0:
            raise ValueError("SSIM data_range must be finite and non-negative")
        # skimage rejects zero, while constant identical fields have exact SSIM 1.
        return value if value > 0 else np.finfo(np.float64).eps

    @staticmethod
    def _frame_ssim(pred, ref, *, data_range, win_size=None, sigma=1.5):
        pred_values = np.asarray(pred, dtype=np.float64)
        ref_values = np.asarray(ref, dtype=np.float64)
        valid = np.isfinite(pred_values) & np.isfinite(ref_values)
        if not valid.any():
            return np.nan, np.full_like(pred_values, np.nan)
        fill = float(np.nanmean(ref_values[valid]))
        pred_values = np.where(valid, pred_values, fill)
        ref_values = np.where(valid, ref_values, fill)
        kwargs = {
            "data_range": data_range,
            "gaussian_weights": True,
            "sigma": sigma,
            "full": True,
        }
        if win_size is not None:
            kwargs["win_size"] = int(win_size)
        score, local = structural_similarity(ref_values, pred_values, **kwargs)
        local = np.where(valid, local, np.nan)
        return score, local

    @classmethod
    def compute(
        cls,
        pred,
        ref,
        *,
        win_size=None,
        sigma=1.5,
        data_range=None,
    ):
        """Return temporal scores and the time-mean local SSIM map."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        if "time" not in pred.dims:
            pred = pred.expand_dims(time=[0])
            ref = ref.expand_dims(time=[0])

        spatial_dims = horizontal_dims(pred)
        if len(spatial_dims) != 2:
            raise ValueError(
                "SSIM requires exactly two horizontal latitude/longitude dimensions"
            )
        other_dims = [
            dim for dim in pred.dims if dim not in {"time", *spatial_dims}
        ]
        if other_dims:
            raise ValueError(
                f"Select a level before SSIM; unsupported dimensions: {other_dims}"
            )
        pred = pred.transpose("time", *spatial_dims)
        ref = ref.transpose("time", *spatial_dims)
        value_range = cls.reference_data_range(ref, data_range)

        scores = []
        maps = []
        for index in range(pred.sizes["time"]):
            score, local = cls._frame_ssim(
                pred.isel(time=index).values,
                ref.isel(time=index).values,
                data_range=value_range,
                win_size=win_size,
                sigma=sigma,
            )
            scores.append(score)
            maps.append(local)

        temporal = xr.DataArray(
            scores,
            dims=("time",),
            coords={"time": pred.time},
            name="ssim",
            attrs={"data_range": value_range, "sigma": sigma},
        )
        local_all = xr.DataArray(
            np.stack(maps),
            dims=("time", *spatial_dims),
            coords={
                "time": pred.time,
                spatial_dims[0]: pred[spatial_dims[0]],
                spatial_dims[1]: pred[spatial_dims[1]],
            },
        )
        spatial = local_all.mean("time", skipna=True).rename("ssim")
        spatial.attrs.update(data_range=value_range, sigma=sigma)
        return temporal, spatial

    @staticmethod
    def global_from_local(local_map):
        """Compute the latitude-weighted global mean of a local SSIM map."""
        lat_dims = [dim for dim in local_map.dims if "lat" in dim.lower()]
        if len(lat_dims) == 1 and local_map[lat_dims[0]].ndim == 1:
            lat = lat_dims[0]
            weights = np.cos(np.deg2rad(local_map[lat]))
            return local_map.weighted(weights).mean(skipna=True).rename("global_ssim")
        return local_map.mean(skipna=True).rename("global_ssim")

    def evaluate(self, pred, ref, *, key="variable"):
        """Compute and retain all SSIM products for one pair."""
        ensure_latlon_regrid(self.regrid, pred)
        ensure_latlon_regrid(self.ref_regrid, ref)
        temporal, spatial = self.compute(
            pred,
            ref,
            win_size=self.win_size,
            sigma=self.sigma,
            data_range=self.data_range,
        )
        global_score = self.global_from_local(spatial)
        self.temporal_ssim[key] = temporal
        self.spatial_ssim[key] = spatial
        self.global_ssim[key] = global_score
        return temporal, spatial, global_score

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        """Retrieve configured variables and compute SSIM products."""
        for var, level, key in self.iter_variables(variables or self.variables):
            pred, ref = self.retrieve_pair(
                var,
                level=level,
                reader_kwargs=reader_kwargs,
                ref_reader_kwargs=ref_reader_kwargs,
            )
            temporal, spatial, global_score = self.evaluate(
                as_dataarray(pred, var), as_dataarray(ref, var), key=key
            )
            if save_netcdf:
                extra = {"var": var, "level": level}
                self.save_pair_netcdf(
                    xr.Dataset(
                        {
                            "ssim": temporal,
                            "global_ssim": global_score,
                        }
                    ),
                    "temporal",
                    extra_keys=extra,
                )
                self.save_pair_netcdf(
                    spatial.to_dataset(), "spatial", extra_keys=extra
                )
        return {
            "temporal": self.temporal_ssim,
            "spatial": self.spatial_ssim,
            "global": self.global_ssim,
        }
