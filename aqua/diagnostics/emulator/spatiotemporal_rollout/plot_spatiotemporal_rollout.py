"""Animation plotting for spatiotemporal rollouts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from aqua.core.util import coord_names

from ..base import EmulatorPlotMixin


class PlotSpatiotemporalRollout(EmulatorPlotMixin):
    """Render aligned rollout fields as a GIF."""

    def __init__(self, **kwargs):
        kwargs.setdefault("diagnostic_name", "spatiotemporal_rollout")
        super().__init__(**kwargs)

    def plot(
        self,
        data,
        *,
        title="Spatiotemporal rollout",
        fps=3,
        dpi=150,
        cmap="RdBu_r",
        extra_keys=None,
    ):
        pred, ref = data.prediction.load(), data.reference.load()
        lon, lat = coord_names(ref)
        if not lon or not lat:
            raise ValueError("rollout animations require latitude/longitude coordinates")
        lower = float(min(pred.min(skipna=True), ref.min(skipna=True)))
        upper = float(max(pred.max(skipna=True), ref.max(skipna=True)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        images = [
            axes[0].pcolormesh(
                ref[lon], ref[lat], ref.isel(time=0), cmap=cmap,
                vmin=lower, vmax=upper, shading="auto"
            ),
            axes[1].pcolormesh(
                pred[lon], pred[lat], pred.isel(time=0), cmap=cmap,
                vmin=lower, vmax=upper, shading="auto"
            ),
        ]
        axes[0].set_title("Reference")
        axes[1].set_title("Prediction")
        time_label = fig.suptitle(title)
        fig.colorbar(images[0], ax=axes, orientation="horizontal")

        def update(index):
            images[0].set_array(np.asarray(ref.isel(time=index)).ravel())
            images[1].set_array(np.asarray(pred.isel(time=index)).ravel())
            time_label.set_text(f"{title}: {ref.time.values[index]}")
            return *images, time_label

        movie = animation.FuncAnimation(
            fig, update, frames=ref.sizes["time"], blit=False
        )
        folder = Path(self.outputsaver.generate_folder("gif"))
        filename = self.outputsaver.generate_name("animation", extra_keys)
        path = folder / f"{filename}.gif"
        movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        self.logger.info("Saved rollout animation: %s", path)
        return str(path)
