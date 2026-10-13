"""Seasonal-cycle plotting."""

import matplotlib.pyplot as plt

from ..base import EmulatorPlotMixin


class PlotSeasonalCycle(EmulatorPlotMixin):
    """Plot monthly means and anomalies."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "seasonal_cycle")
        super().__init__(**kwargs)

    def plot(self, data, *, title="Seasonal cycle", save=False,
             extra_keys=None, format="pdf", dpi=300):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for name, label in (("prediction", "Prediction"), ("reference", "Reference")):
            axes[0].plot(data.month, data[name], marker="o", label=label)
        for name, label in (
            ("prediction_anomaly", "Prediction"),
            ("reference_anomaly", "Reference"),
        ):
            axes[1].plot(data.month, data[name], marker="o", label=label)
        axes[0].set_ylabel("Monthly mean")
        axes[1].set_ylabel("Monthly anomaly")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(list("JFMAMJJASOND"))
        for ax in axes:
            ax.legend()
            ax.grid(True)
        fig.suptitle(title)
        fig.tight_layout()
        if save:
            self.save_plot(
                fig, "seasonal_cycle", extra_keys, dpi=dpi, format=format
            )
        return fig, axes
