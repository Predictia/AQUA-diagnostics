"""Plotting helpers for RMSE."""

import matplotlib.pyplot as plt

from aqua.core.graphics import plot_single_map

from ..base import EmulatorPlotMixin


class PlotRMSE(EmulatorPlotMixin):
    """Create map and time-series RMSE figures."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "rmse")
        super().__init__(**kwargs)

    def plot_spatial(self, data, *, title="Spatial RMSE", vmax=None, save=False,
                     extra_keys=None, format="pdf", dpi=300):
        """Plot an RMSE map with a non-negative red color scale."""
        if vmax is None:
            vmax = float(data.quantile(0.98, skipna=True))
            if not vmax > 0:
                vmax = 1.0
        fig, ax = plot_single_map(
            data=data,
            return_fig=True,
            contour=True,
            title=title,
            sym=False,
            vmin=0,
            vmax=vmax,
            cmap="Reds",
        )
        if save:
            self.save_plot(fig, "spatial", extra_keys, dpi=dpi, format=format)
        return fig, ax

    def plot_temporal(self, data, *, title="Temporal RMSE", save=False,
                      extra_keys=None, format="pdf", dpi=300):
        """Plot temporal RMSE as a time series."""
        fig, ax = plt.subplots(figsize=(12, 6))
        data.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("RMSE")
        ax.grid(True)
        fig.tight_layout()
        if save:
            self.save_plot(fig, "temporal", extra_keys, dpi=dpi, format=format)
        return fig, ax
