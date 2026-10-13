"""AMO plotting."""

from .._plot import MapSeriesPlotter


class PlotAMO(MapSeriesPlotter):
    """Plot filtered prediction and reference AMO indices."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "amo")
        super().__init__(**kwargs)

    def plot_index(self, data, **kwargs):
        return self.plot_series(
            data.prediction_filtered,
            reference=data.reference_filtered,
            labels=("Prediction", "Reference"),
            ylabel="AMO Index",
            product=kwargs.pop("product", "index"),
            **kwargs,
        )
