import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import cbar_get_label, coord_names, get_projection, load_yaml

@dataclass
class VideoSettings:
    """Container for default video-specific settings."""

    outputdir: str = "./videos"
    format: str = "gif"
    fps: Optional[int] = None
    dpi: int = 150
    cmap: str = "RdBu_r"
    projection: str = "plate_carree"
    figure_size: Tuple[float, float] = (12.0, 6.0)
    target_label: str = "Target"
    prediction_label: str = "Prediction"
    colorbar_steps: Optional[int] = None

class SpatiotemporalRollout:
    """
    Diagnostic that produces videos (GIFs) of target vs prediction rollouts.
    
    Args:
        config: Path to the diagnostic YAML configuration file or dictionary
        loglevel: Log level (default: WARNING)
    """

    def __init__(self, config, loglevel = "WARNING"):
        """Initialize the SpatiotemporalRollout diagnostic."""

        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "SpatiotemporalRollout")

        # Load configuration
        if isinstance(config, str):
            self.logger.debug("Loading configuration file %s", config)
            self.config: Dict[str, Any] = load_yaml(config)
        else:
            self.logger.debug("Configuration provided as dictionary")
            self.config = config

        # Evaluation dates
        self.startdate = self.config.get("dates", {}).get("startdate")
        self.enddate = self.config.get("dates", {}).get("enddate")
        if not self.startdate or not self.enddate:
            raise ValueError("Both 'startdate' and 'enddate' must be provided in 'dates'.")

        # Prepare video settings
        self.video_settings = self._prepare_video_settings(self.config.get("video", {}))

        # Initialize readers
        self._reader()

    def _reader(self):
        """Initialise Reader instances for data and reference datasets."""

        config_data = self.config.get("data", {})
        config_data_ref = self.config.get("data_ref", {})

        self.reader_data = Reader(catalog=config_data.get("catalog"),
                                  model=config_data.get("model"),
                                  exp=config_data.get("exp"),
                                  source=config_data.get("source"),
                                  regrid=config_data.get("regrid"),
                                  fix=config_data.get("fix"),
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel)

        self.reader_data_ref = Reader(catalog=config_data_ref.get("catalog"),
                                      model=config_data_ref.get("model"),
                                      exp=config_data_ref.get("exp"),
                                      source=config_data_ref.get("source"),
                                      regrid=config_data_ref.get("regrid"),
                                      fix=config_data_ref.get("fix"),
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel)

    @staticmethod
    def _prepare_video_settings(video_cfg):
        """Prepare video settings from configuration."""

        settings = VideoSettings()
        for field_name in settings.__dataclass_fields__:
            # Override default settings with values from configuration
            if field_name in video_cfg and video_cfg[field_name] is not None:
                setattr(settings, field_name, video_cfg[field_name])

        # Create output directory if it doesn't exist
        os.makedirs(settings.outputdir, exist_ok=True)
        return settings

    def retrieve(self):
        """Retrieve all variables defined in the configuration."""

        variables = self.config.get("variables", [])
        if not variables:
            self.logger.warning("No variables defined in configuration; nothing to retrieve.")
            self.retrieved_data = {}
            self.retrieved_data_ref = {}
            return

        self.retrieved_data = {}
        self.retrieved_data_ref = {}

        # Process each variable
        for var_cfg in variables:
            var_name = var_cfg.get("name")
            if not var_name:
                self.logger.warning("Skipping variable entry without 'name': %s", var_cfg)
                continue

            levels = var_cfg.get("level")
            if levels is None:
                level_list: Iterable[Optional[int]] = [None]
            elif isinstance(levels, (int, float)):
                level_list = [int(levels)]
            elif isinstance(levels, list):
                level_list = [int(level) for level in levels]
            else:
                self.logger.warning("Invalid level specification for %s: %s", var_name, levels)
                continue

            # Process each level
            for level in level_list:
                key = f"{var_name}_{level}" if level is not None else var_name
                retrieve_args = {"var": var_name}
                if level is not None:
                    retrieve_args["level"] = level

                try:
                    data_ref = self.reader_data_ref.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                    if level is not None and "plev" in data_ref.coords:
                        data_ref = data_ref.isel(plev=0, drop=True)
                    # Apply regridding if configured
                    if self.reader_data_ref.tgt_grid_name is not None:
                        self.logger.debug(
                            "Applying regridding to reference data for %s to %s",
                            var_name,
                            self.reader_data_ref.tgt_grid_name,
                        )
                        data_ref = self.reader_data_ref.regrid(data_ref)
                    self.retrieved_data_ref[key] = data_ref

                except Exception as exc:
                    self.logger.error("Failed to retrieve reference data for %s: %s", key, exc, exc_info=True)
                    continue

                try:
                    data = self.reader_data.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                    if level is not None and "plev" in data.coords:
                        data = data.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_data.tgt_grid_name is not None:
                        self.logger.debug(
                            "Applying regridding to prediction data for %s to %s",
                            var_name,
                            self.reader_data.tgt_grid_name,
                        )
                        data = self.reader_data.regrid(data)
                    self.retrieved_data[key] = data

                except Exception as exc:
                    self.logger.error("Failed to retrieve prediction data for %s: %s", key, exc, exc_info=True)
                    self.retrieved_data_ref.pop(key, None)

        missing = set(self.retrieved_data_ref) ^ set(self.retrieved_data)
        if missing:
            for key in missing:
                self.retrieved_data_ref.pop(key, None)
                self.retrieved_data.pop(key, None)
            self.logger.warning("Dropped variables with incomplete data: %s", sorted(missing))

        self.logger.info("Retrieved data for keys: %s", sorted(self.retrieved_data.keys()))

        # Log regridding information
        if self.reader_data_ref.tgt_grid_name is not None:
            self.logger.info(f"Reference data regridded to: {self.reader_data_ref.tgt_grid_name}")
        if self.reader_data.tgt_grid_name is not None:
            self.logger.info(f"Prediction data regridded to: {self.reader_data.tgt_grid_name}")

    def generate_videos(self):
        """Generate rollout videos for each retrieved variable."""

        if not hasattr(self, "retrieved_data") or not hasattr(self, "retrieved_data_ref"):
            raise RuntimeError("Data not retrieved. Call 'retrieve' before generating videos.")

        outputs = {}
        for key, data in self.retrieved_data.items():
            # Skip if the variable is not in the reference data
            if key not in self.retrieved_data_ref:
                continue

            # Create video for the variable
            result = self._create_video_for_variable(key, data, self.retrieved_data_ref[key])
            if result is not None:
                outputs[key] = result

        return outputs

    def _create_video_for_variable(self, key, data, data_ref):
        """Create video for a given variable."""

        base_var, _ = self._parse_processed_key(key)
        if base_var not in data or base_var not in data_ref:
            self.logger.warning("Variable '%s' not found in datasets for key '%s'.", base_var, key)
            return None

        da = data[base_var]
        da_ref = data_ref[base_var]

        # Align the data along the time dimension for the videos to be synchronized
        try:
            da, da_ref = xr.align(da, da_ref, join="inner")
        except Exception as exc:
            self.logger.error("Failed to align data for %s: %s", key, exc, exc_info=True)
            return None

        if "time" not in da.dims or da.sizes.get("time", 0) == 0:
            self.logger.warning("No time dimension available for %s; skipping.", key)
            return None

        # Transpose the data to put the time dimension first
        try:
            da = da.transpose("time", ...)
            da_ref = da_ref.transpose("time", ...)
        except ValueError:
            pass

        # Determine the longitude and latitude coordinates
        lon_name, lat_name = coord_names(da_ref)
        if not lon_name or not lat_name:
            self.logger.error("Unable to determine lat/lon coordinates for %s; skipping.", key)
            return None

        if lon_name not in da.dims or lat_name not in da.dims:
            self.logger.error("Prediction data missing lat/lon for %s; skipping.", key)
            return None

        # Load the data into memory
        da = da.load()
        da_ref = da_ref.load()

        # Resolve colorbar limits and steps
        variable_cfg = self._variable_config(key)
        vmin, vmax = self._resolve_colorbar_limits(da, da_ref, variable_cfg)
        color_steps = self._resolve_colorbar_steps(variable_cfg)
        cmap = self._resolve_colormap(variable_cfg, color_steps)

        # Resolve projection and figure size
        projection = self._resolve_projection()
        figure_size = self._resolve_figure_size(variable_cfg)

        # Set the data in the proper shape
        lon = da_ref[lon_name].values
        lat = da_ref[lat_name].values
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Create the figure
        fig, axes = plt.subplots(1, 2,
                                 figsize=figure_size,
                                 subplot_kw={"projection": projection},
                                 constrained_layout=False)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax in axes:
            ax.coastlines()

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.94, wspace=0.08)

        norm, boundaries = self._resolve_norm(vmin, vmax, color_steps)

        # Plot the first time step of the reference and prediction data
        mesh_ref = axes[0].pcolormesh(lon2d, lat2d,
                                      da_ref.isel(time=0).values,
                                      transform=ccrs.PlateCarree(),
                                      cmap=cmap,
                                      norm=norm,
                                      shading="auto")

        mesh_pred = axes[1].pcolormesh(lon2d, lat2d,
                                       da.isel(time=0).values,
                                       transform=ccrs.PlateCarree(),
                                       cmap=cmap,
                                       norm=norm,
                                       shading="auto")

        # Set the titles for the reference and prediction data
        axes[0].set_title(self.video_settings.target_label)
        axes[1].set_title(self.video_settings.prediction_label)

        # Set the colorbar
        colorbar_label = self._resolve_colorbar_label(da_ref, variable_cfg)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        colorbar = fig.colorbar(mappable,
                                ax=axes.ravel().tolist(),
                                orientation="horizontal",
                                fraction=0.05,
                                pad=0.12)

        # Set the ticks for the colorbar
        ticks_override = None
        if boundaries is not None:
            ticks_override = np.array([vmin, vmax])
            colorbar.set_ticks(ticks_override)
        if colorbar_label:
            colorbar.set_label(colorbar_label)

        # Get the times
        times = da_ref["time"].values
        # Try to convert the times to a pandas index
        try:
            time_index = da_ref["time"].to_index()
        except Exception:
            time_index = None
        time_text = fig.text(0.5, 0.95, "Time: ", ha="center", va="bottom", fontsize=11)

        def _format_time(index):
            """Format the time for to a common format"""
            if time_index is not None:
                timestamp = time_index[index]
                if hasattr(timestamp, "strftime"):
                    return timestamp.strftime("%Y-%m-%d %H:%M")
                return str(timestamp)

            value = np.asarray(times[index]).item()
            if np.issubdtype(type(value), np.datetime64):
                return np.datetime_as_string(value, unit="h")
            return str(value)

        def _update(frame_index):
            """Update the plot for a given frame index"""
            for mesh, data_array in ((mesh_ref, da_ref), (mesh_pred, da)):
                frame_values = np.ma.masked_invalid(data_array.isel(time=frame_index).values)
                mesh.set_array(frame_values.ravel())

            timestamp = _format_time(frame_index)
            time_text.set_text(f"Time: {timestamp}")
            return mesh_ref, mesh_pred, time_text

        # Generate the output path
        output_path = self._generate_output_path(key)

        # Resolve the fps for the current variable, falling back to global settings
        variable_cfg = self._variable_config(key)
        variable_fps = variable_cfg.get("fps")
        current_fps = variable_fps if variable_fps is not None else self.video_settings.fps
        if current_fps is None:
            self.logger.warning("No FPS defined for variable %s or in global settings; defaulting to 3.", key)
            current_fps = 3

        writer = self._resolve_animation_writer(output_path, current_fps)

        # Save the video
        self.logger.info("Saving rollout animation for %s to %s", key, output_path)
        try:
            anim = animation.FuncAnimation(fig, _update, frames=len(times), blit=False)
            anim.save(output_path, writer=writer, dpi=self.video_settings.dpi)
        except Exception as exc:
            self.logger.error("Failed to save video for %s: %s", key, exc, exc_info=True)
            plt.close(fig)
            return None

        plt.close(fig)
        return output_path

    def _resolve_animation_writer(self, output_path, fps):
        """Resolve the animation writer for the given output path and fps"""
        fmt = self.video_settings.format.lower()
        if fmt == "gif":
            return animation.PillowWriter(fps=fps)

        plt.close('all')
        raise ValueError(f"Unsupported animation format '{self.video_settings.format}'. "
                         "Currently only 'gif' is supported without external dependencies.")

    def _resolve_projection(self):
        """Resolve the projection for the video"""
        projection_config = self.video_settings.projection
        if isinstance(projection_config, str):
            try:
                return get_projection(projection_config)
            except Exception as exc:
                self.logger.warning("Failed to get projection '%s': %s. Falling back to PlateCarree.", projection_config, exc)
                return ccrs.PlateCarree()
        if isinstance(projection_config, ccrs.Projection):
            return projection_config
        self.logger.warning("Unsupported projection configuration %s; using PlateCarree.", projection_config)
        return ccrs.PlateCarree()

    def _resolve_figure_size(self, variable_cfg):
        """Resolve the figure size for the video"""
        fig_size_override = variable_cfg.get("figure_size")
        if fig_size_override and len(fig_size_override) == 2:
            return float(fig_size_override[0]), float(fig_size_override[1])
        return self.video_settings.figure_size

    def _resolve_colorbar_label(self, data_ref, variable_cfg):
        """Resolve the colorbar label for the video"""
        label = variable_cfg.get("colorbar_label") or variable_cfg.get("cbar_label")
        if label:
            return label

        default_label = getattr(self.video_settings, "colorbar_label", None)
        if default_label:
            return default_label

        try:
            return cbar_get_label(data_ref, loglevel=self.loglevel)
        except Exception as exc:
            self.logger.debug("Failed to determine colorbar label automatically: %s", exc)
            return None

    def _resolve_colorbar_limits(self, da, da_ref, variable_cfg):
        """Resolve the colorbar limits for the video"""
        vmin = variable_cfg.get("vmin", variable_cfg.get("colorbar_min"))
        vmax = variable_cfg.get("vmax", variable_cfg.get("colorbar_max"))
        if vmin is not None and vmax is not None:
            return float(vmin), float(vmax)

        combined = xr.concat([da, da_ref], dim="_merge")
        data_min = combined.min(skipna=True).item() if combined.count() > 0 else np.nan
        data_max = combined.max(skipna=True).item() if combined.count() > 0 else np.nan

        if np.isnan(data_min) or np.isnan(data_max):
            self.logger.warning("Could not determine colorbar limits for variable %s; using defaults.", da.name)
            return 0.0, 1.0

        if data_min == data_max:
            delta = 1e-6 if data_min == 0 else abs(data_min) * 0.05
            data_min -= delta
            data_max += delta

        return float(data_min), float(data_max)

    def _resolve_colorbar_steps(self, variable_cfg):
        """Resolve the colorbar steps for the video"""
        steps = variable_cfg.get("colorbar_steps")
        if steps is None:
            steps = self.video_settings.colorbar_steps
        if steps is None:
            return None
        try:
            steps_int = int(steps)
        except (TypeError, ValueError):
            self.logger.warning("Invalid colorbar_steps value %s; ignoring.", steps)
            return None
        if steps_int <= 1:
            self.logger.warning("colorbar_steps must be greater than 1; received %s.", steps)
            return None
        return steps_int

    def _resolve_colormap(self, variable_cfg, steps):
        """Resolve the colormap for the video"""
        cmap_name = variable_cfg.get("cmap", self.video_settings.cmap)
        try:
            base_cmap = plt.get_cmap(cmap_name)
        except ValueError:
            self.logger.warning("Unknown colormap %s; falling back to default.", cmap_name)
            base_cmap = plt.get_cmap(self.video_settings.cmap)

        if steps is None:
            return base_cmap

        # Evenly sample discrete colors, avoiding gradient flicker in GIF quantization.
        sampled = base_cmap(np.linspace(0, 1, steps))
        return ListedColormap(sampled, name=f"{base_cmap.name}_{steps}")

    def _resolve_norm(self, vmin, vmax, steps):
        """Resolve the norm for the video"""
        if steps is None:
            return Normalize(vmin=vmin, vmax=vmax), None

        boundaries = np.linspace(vmin, vmax, steps + 1)
        norm = BoundaryNorm(boundaries, ncolors=len(boundaries) - 1, clip=True)
        return norm, boundaries

    def _variable_config(self, key):
        """Resolve the variable configuration for the video"""
        base_var, level = self._parse_processed_key(key)
        for var_cfg in self.config.get("variables", []):
            if var_cfg.get("name") != base_var:
                continue
            levels = var_cfg.get("level")
            if levels is None and level is None:
                return var_cfg.get("video", var_cfg)
            if isinstance(levels, (int, float)) and level is not None and int(levels) == level:
                return var_cfg.get("video", var_cfg)
            if isinstance(levels, list) and level in [int(lvl) for lvl in levels]:
                return var_cfg.get("video", var_cfg)
        return {}

    def _generate_output_path(self, key):
        """Generate the output path for the video"""
        data_cfg = self.config.get("data", {})
        ref_cfg = self.config.get("data_ref", {})

        filename_parts = [self._sanitize_filename_part(data_cfg.get("model", "model")),
                          self._sanitize_filename_part(data_cfg.get("exp", "exp")),
                          self._sanitize_filename_part(data_cfg.get("source", "src")),
                          "vs",
                          self._sanitize_filename_part(ref_cfg.get("model", "refmodel")),
                          self._sanitize_filename_part(ref_cfg.get("exp", "refexp")),
                          key,
                          self._sanitize_filename_part(self.startdate),
                          self._sanitize_filename_part(self.enddate)]

        filename = "_".join(filter(None, filename_parts))
        extension = f".{self.video_settings.format.lower()}"
        return os.path.join(self.video_settings.outputdir, f"{filename}{extension}")

    def _sanitize_filename_part(self, value):
        """Sanitize the filename part"""
        text = str(value) if value is not None else ""
        for char in [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            text = text.replace(char, "_")
        return text

    def _parse_processed_key(self, key):
        """Parse the processed key"""
        parts = key.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            try:
                level = int(parts[-1])
                base = "_".join(parts[:-1])
                return base, level
            except ValueError:
                return key, None
        return key, None
