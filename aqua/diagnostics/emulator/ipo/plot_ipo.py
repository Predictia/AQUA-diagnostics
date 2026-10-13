"""IPO plotting."""

from .._plot import MapSeriesPlotter


class PlotIPO(MapSeriesPlotter):
    """Plot the filtered IPO prediction and reference indices."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "ipo")
        super().__init__(**kwargs)

    def plot_index(self, data, **kwargs):
        return self.plot_series(
            data.prediction_filtered,
            reference=data.reference_filtered,
            labels=("Prediction", "Reference"),
            ylabel="Tripole Index",
            product=kwargs.pop("product", "index"),
            **kwargs,
        )
