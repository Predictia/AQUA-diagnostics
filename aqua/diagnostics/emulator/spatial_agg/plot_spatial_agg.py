"""Spatial aggregation plotting."""

from .._plot import MapSeriesPlotter


class PlotSpatialAgg(MapSeriesPlotter):
    """Plot reference, prediction, and climatological difference maps."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "spatial_agg")
        super().__init__(**kwargs)

    def plot(self, data, **kwargs):
        return self.plot_three_maps(
            data.reference,
            data.prediction,
            data.difference,
            product=kwargs.pop("product", "climatology"),
            **kwargs,
        )
