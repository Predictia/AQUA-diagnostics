"""Plotting for anomaly correlation."""

from .._plot import MapSeriesPlotter


class PlotACC(MapSeriesPlotter):
    """Plot ACC maps and series on their natural ``[-1, 1]`` range."""

    cmap = "RdBu_r"
    vmin = -1
    vmax = 1
    symmetric = True

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "acc")
        super().__init__(**kwargs)

    def plot_spatial(self, data, **kwargs):
        return self.plot_map(data, product=kwargs.pop("product", "spatial"), **kwargs)

    def plot_temporal(self, data, **kwargs):
        return self.plot_series(
            data, ylabel="ACC", product=kwargs.pop("product", "temporal"), **kwargs
        )
