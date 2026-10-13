"""Small reusable plotting building blocks for emulator diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aqua.core.graphics import plot_single_map

from .base import EmulatorPlotMixin


class MapSeriesPlotter(EmulatorPlotMixin):
    """Plot maps and one-dimensional series with OutputSaver integration."""

    cmap = "RdBu_r"
    vmin = None
    vmax = None
    symmetric = False

    def plot_map(
        self,
        data,
        *,
        title=None,
        product="map",
        cmap=None,
        vmin=None,
        vmax=None,
        symmetric=None,
        save=False,
        extra_keys=None,
        format="pdf",
        dpi=300,
    ):
        fig, ax = plot_single_map(
            data=data,
            return_fig=True,
            contour=True,
            title=title,
            cmap=cmap or self.cmap,
            vmin=self.vmin if vmin is None else vmin,
            vmax=self.vmax if vmax is None else vmax,
            sym=self.symmetric if symmetric is None else symmetric,
        )
        if save:
            self.save_plot(
                fig, product, extra_keys, dpi=dpi, format=format
            )
        return fig, ax

    def plot_series(
        self,
        data,
        *,
        reference=None,
        labels=("Prediction", "Reference"),
        title=None,
        ylabel=None,
        product="timeseries",
        save=False,
        extra_keys=None,
        format="pdf",
        dpi=300,
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        data.plot(ax=ax, label=labels[0])
        if reference is not None:
            reference.plot(ax=ax, label=labels[1])
            ax.legend()
        ax.set_title(title or "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout()
        if save:
            self.save_plot(
                fig, product, extra_keys, dpi=dpi, format=format
            )
        return fig, ax

    def plot_three_maps(
        self,
        reference,
        prediction,
        difference,
        *,
        titles=("Reference", "Prediction", "Difference"),
        save=False,
        product="comparison",
        extra_keys=None,
        format="pdf",
        dpi=300,
    ):
        """Create a lightweight three-panel map comparison."""
        lat = next((d for d in reference.dims if "lat" in d.lower()), None)
        lon = next((d for d in reference.dims if "lon" in d.lower()), None)
        if lat is None or lon is None:
            raise ValueError("three-map plots require latitude and longitude")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        valid = np.concatenate(
            [
                reference.values[np.isfinite(reference.values)],
                prediction.values[np.isfinite(prediction.values)],
            ]
        )
        vmin, vmax = (
            np.nanpercentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        )
        diffmax = float(np.nanpercentile(np.abs(difference), 98))
        for ax, field, title, cmap, lower, upper in (
            (axes[0], reference, titles[0], "viridis", vmin, vmax),
            (axes[1], prediction, titles[1], "viridis", vmin, vmax),
            (axes[2], difference, titles[2], "RdBu_r", -diffmax, diffmax),
        ):
            mesh = ax.pcolormesh(
                field[lon], field[lat], field.transpose(lat, lon),
                cmap=cmap, vmin=lower, vmax=upper, shading="auto"
            )
            ax.set_title(title)
            fig.colorbar(mesh, ax=ax, orientation="horizontal")
        fig.tight_layout()
        if save:
            self.save_plot(
                fig, product, extra_keys, dpi=dpi, format=format
            )
        return fig, axes
