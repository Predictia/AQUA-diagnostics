"""Rollout plotting."""

from .._plot import MapSeriesPlotter


class PlotRollout(MapSeriesPlotter):
    """Plot prediction and reference rollout time series."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "rollout")
        super().__init__(**kwargs)

    def plot(self, data, **kwargs):
        return self.plot_series(
            data.prediction,
            reference=data.reference,
            labels=("Prediction", "Reference"),
            product=kwargs.pop("product", "timeseries"),
            **kwargs,
        )
