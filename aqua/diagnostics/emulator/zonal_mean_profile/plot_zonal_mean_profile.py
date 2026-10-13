"""Zonal profile plotting."""

import matplotlib.pyplot as plt

from ..base import EmulatorPlotMixin


class PlotZonalMeanProfile(EmulatorPlotMixin):
    """Plot prediction and reference zonal-mean latitude profiles."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "zonal_mean_profile")
        super().__init__(**kwargs)

    def plot(self, data, *, title="Zonal mean profile", save=False,
             extra_keys=None, format="pdf", dpi=300):
        ref = data.zonal_mean_reference
        pred = data.zonal_mean_prediction
        lat = next(dim for dim in ref.dims if "lat" in dim.lower())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ref[lat], ref, label="Reference", color="blue")
        ax.plot(pred[lat], pred, label="Prediction", color="red")
        ax.set_xlabel("Latitude")
        ax.set_title(title)
        ax.set_xlim(-90, 90)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save:
            self.save_plot(
                fig, "zonal_mean_profile", extra_keys, dpi=dpi, format=format
            )
        return fig, ax
