"""PSD plotting."""

import matplotlib.pyplot as plt

from ..base import EmulatorPlotMixin


class PlotPSD(EmulatorPlotMixin):
    """Plot prediction and reference spectra on logarithmic axes."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "psd")
        super().__init__(**kwargs)

    def plot(self, data, *, title="Power Spectral Density", save=False,
             product="spectrum", extra_keys=None, format="pdf", dpi=300):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(
            data.wavenumber, data.psd_prediction, label="Prediction"
        )
        ax.loglog(
            data.wavenumber,
            data.psd_reference,
            label="Reference",
            linestyle="--",
        )
        ax.set_xlabel("Wavenumber / spherical harmonic degree")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.5)
        fig.tight_layout()
        if save:
            self.save_plot(fig, product, extra_keys, dpi=dpi, format=format)
        return fig, ax
