"""Vertical zonal-mean plotting."""

import matplotlib.pyplot as plt
import numpy as np

from ..base import EmulatorPlotMixin


class PlotVerticalZonalMean(EmulatorPlotMixin):
    """Plot reference, prediction and difference latitude-pressure sections."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "vertical_zonal_mean")
        super().__init__(**kwargs)

    def plot(self, data, *, title="Vertical zonal mean", save=False,
             extra_keys=None, format="pdf", dpi=300):
        ref = data.zonal_mean_reference
        pred = data.zonal_mean_prediction
        diff = data.zonal_mean_difference
        level = data.attrs.get("level_coordinate") or next(
            dim for dim in ref.dims if dim in {"plev", "lev", "level", "pressure"}
        )
        lat = next(dim for dim in ref.dims if "lat" in dim.lower())
        pressure = ref[level].values
        if np.nanmax(pressure) > 2000:
            pressure = pressure / 100
        values = np.concatenate(
            [ref.values[np.isfinite(ref)], pred.values[np.isfinite(pred)]]
        )
        lower, upper = np.nanpercentile(values, [2, 98])
        diffmax = float(np.nanpercentile(np.abs(diff), 98))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, field, subtitle, cmap, vmin, vmax in (
            (axes[0], ref, "Reference", "viridis", lower, upper),
            (axes[1], pred, "Prediction", "viridis", lower, upper),
            (axes[2], diff, "Difference", "RdBu_r", -diffmax, diffmax),
        ):
            contour = ax.contourf(
                field[lat], pressure, field.transpose(level, lat),
                levels=18, cmap=cmap, vmin=vmin, vmax=vmax, extend="both"
            )
            ax.invert_yaxis()
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Pressure [hPa]")
            ax.set_title(subtitle)
            fig.colorbar(contour, ax=ax, orientation="horizontal")
        if "significance_mask" in data:
            mask = data.significance_mask.transpose(level, lat)
            axes[2].contourf(
                mask[lat], pressure, mask.astype(float),
                levels=[0.5, 1.5], colors="none", hatches=["...."]
            )
        fig.suptitle(title)
        fig.tight_layout()
        if save:
            self.save_plot(
                fig, "vertical_zonal_mean", extra_keys, dpi=dpi, format=format
            )
        return fig, axes
