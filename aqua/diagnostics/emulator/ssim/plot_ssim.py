"""Plotting helpers for SSIM."""

import matplotlib.pyplot as plt

from aqua.core.graphics import plot_single_map

from ..base import EmulatorPlotMixin


class PlotSSIM(EmulatorPlotMixin):
    """Create local-map and temporal SSIM figures."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "ssim")
        super().__init__(**kwargs)

    def plot_spatial(self, data, *, title="Time-mean local SSIM", save=False,
                     extra_keys=None, format="pdf", dpi=300):
        fig, ax = plot_single_map(
            data=data,
            return_fig=True,
            contour=True,
            title=title,
            sym=False,
            vmin=-1,
            vmax=1,
            cmap="RdYlGn",
        )
        if save:
            self.save_plot(fig, "spatial", extra_keys, dpi=dpi, format=format)
        return fig, ax

    def plot_temporal(self, data, *, title="Temporal SSIM", save=False,
                      extra_keys=None, format="pdf", dpi=300):
        fig, ax = plt.subplots(figsize=(12, 6))
        data.plot(ax=ax)
        ax.set_ylim(-1, 1)
        ax.set_title(title)
        ax.set_ylabel("SSIM")
        ax.grid(True)
        fig.tight_layout()
        if save:
            self.save_plot(fig, "temporal", extra_keys, dpi=dpi, format=format)
        return fig, ax
