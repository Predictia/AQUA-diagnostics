"""Plot pairwise correlation maps."""

from .._plot import MapSeriesPlotter


class PlotCOV(MapSeriesPlotter):
    """Plot Pearson correlation and correlation-difference maps."""

    cmap = "RdBu_r"
    vmin = -1
    vmax = 1
    symmetric = True

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "cov")
        super().__init__(**kwargs)
