"""Variability plotting."""

from .._plot import MapSeriesPlotter


class PlotVariability(MapSeriesPlotter):
    """Plot reference, prediction, and standard-deviation difference."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "variability")
        super().__init__(**kwargs)

    def plot(self, data, **kwargs):
        return self.plot_three_maps(
            data.std_reference,
            data.std_prediction,
            data.std_difference,
            product=kwargs.pop("product", "variability"),
            **kwargs,
        )
