import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import cbar_get_label, coord_names, get_projection, load_yaml

# TODO: As I have mentioned in other diagnostics, I am aware that some code is repeated
# and could be refactored to avoid code duplication.

@dataclass
class PlotSettings:
    """Configuration class for map appearance"""

    cmap = "viridis"
    figure_size = (6.0, 4.5)
    projection = ccrs.PlateCarree()
    add_coastlines = True
    colorbar_orientation = "horizontal"
    discrete_colorbar = False
    n_colorbar_levels = 10
    vmin = None
    vmax = None


class SpatialAgg:
    """Diagnostic that computes temporal climatology maps."""
    def __init__(self, config, loglevel):
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "SpatialAgg")

        # Load configuration
        if isinstance(config, str):
            self.logger.debug("Loading configuration file %s", config)
            self.config: Dict[str, Any] = load_yaml(config)
        else:
            self.logger.debug("Configuration provided as dictionary")
            self.config = config

        self._validate_config()

        # Evaluation dates
        self.startdate = self.config["dates"]["startdate"]
        self.enddate = self.config["dates"]["enddate"]

        # Output directories
        self.outputdir_fig = self.config.get("outputdir_fig", "./figs_spatial_agg")
        self.figure_format = self._normalise_extension(self.config.get("figure_format", "pdf"))
        os.makedirs(self.outputdir_fig, exist_ok=True)

        self.plot_settings = self._load_plot_settings(self.config.get("plot", {}))

        self._init_readers()

    def _validate_config(self):
        """Validate the configuration"""
        required_sections = ["dates", "variables"]
        missing = [section for section in required_sections if section not in self.config]
        if missing:
            raise ValueError(
                "Missing required configuration sections: " + ", ".join(missing)
            )

        # It is possible to evaluate only one of the datasets
        if not self.config.get("data") and not self.config.get("data_ref"): 
            raise ValueError("At least one of 'data' or 'data_ref' must be provided.")

        for key in ("startdate", "enddate"):
            if not self.config["dates"].get(key):
                raise ValueError(f"'dates.{key}' must be provided in the configuration.")

        if not self.config.get("variables"):
            raise ValueError("The configuration must define at least one variable.")

    def _init_readers(self):
        """Initialize the readers"""
        self.reader_data = self._build_reader(self.config.get("data"))
        self.reader_data_ref = self._build_reader(self.config.get("data_ref"))

    def _build_reader(self, cfg):
        """Build the reader"""
        if not cfg:
            return None
        else:
            return Reader(catalog=cfg.get("catalog"),
                          model=cfg.get("model"),
                          exp=cfg.get("exp"),
                          source=cfg.get("source"),
                          regrid=cfg.get("regrid"),
                          fix=cfg.get("fix"),
                          areas=cfg.get("areas"),
                          startdate=self.startdate,
                          enddate=self.enddate,
                          loglevel=self.loglevel)

    def _load_plot_settings(self, plot_cfg):
        """Load the plot settings"""
        settings = PlotSettings()
        if "cmap" in plot_cfg and plot_cfg["cmap"]:
            settings.cmap = str(plot_cfg["cmap"])

        figure_size = plot_cfg.get("figure_size")
        if isinstance(figure_size, (list, tuple)) and len(figure_size) == 2:
            settings.figure_size = (float(figure_size[0]), float(figure_size[1]))

        projection_cfg = plot_cfg.get("projection")
        if projection_cfg is not None:
            try:
                settings.projection = get_projection(projection_cfg)
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialise projection '%s': %s. Using PlateCarree.",
                    projection_cfg,
                    exc,
                )
                settings.projection = ccrs.PlateCarree()

        coastlines = plot_cfg.get("coastlines")
        if coastlines is not None:
            settings.add_coastlines = bool(coastlines)

        orientation = plot_cfg.get("colorbar_orientation")
        if orientation in {"horizontal", "vertical"}:
            settings.colorbar_orientation = orientation

        # Add discrete colorbar settings
        discrete = plot_cfg.get("discrete_colorbar")
        if discrete is not None:
            settings.discrete_colorbar = bool(discrete)
        
        n_levels = plot_cfg.get("n_colorbar_levels")
        if isinstance(n_levels, int) and n_levels > 0:
            settings.n_colorbar_levels = n_levels

        # Add vmin/vmax settings
        vmin = plot_cfg.get("vmin")
        if vmin is not None:
            settings.vmin = float(vmin)
        
        vmax = plot_cfg.get("vmax")
        if vmax is not None:
            settings.vmax = float(vmax)

        return settings

    def retrieve(self) -> None:
        """Retrieve all variables defined in the configuration."""
        # Get the variables to evaluate
        variables = self.config.get("variables", [])
        if not variables:
            self.logger.warning("No variables defined in configuration; nothing to retrieve.")
            self.retrieved_data = {}
            self.retrieved_data_ref = {}
            return None

        # Initialize the retrieved data
        self.retrieved_data = {}
        self.retrieved_data_ref = {}

        for var_cfg in variables:
            var_name = var_cfg.get("name")
            if not var_name:
                self.logger.warning("Skipping variable entry without 'name': %s", var_cfg)
                continue

            # Process each level
            for level in self._normalise_levels(var_cfg.get("level")):
                key = self._build_key(var_name, level)
                retrieve_args = {"var": var_name}
                if level is not None:
                    retrieve_args["level"] = level

                if self.reader_data_ref is not None:
                    data_ref = self._retrieve_dataset(reader=self.reader_data_ref,
                                                      retrieve_args=retrieve_args,
                                                      level=level,
                                                      key=key,
                                                      storage=self.retrieved_data_ref,
                                                      label="reference")
                    if data_ref is None:
                        self.logger.debug("Reference retrieval failed for %s", key)

                if self.reader_data is not None:
                    data = self._retrieve_dataset(reader=self.reader_data,
                                                  retrieve_args=retrieve_args,
                                                  level=level,
                                                  key=key,
                                                  storage=self.retrieved_data,
                                                  label="experiment")
                    if data is None:
                        self.logger.debug("Experiment retrieval failed for %s", key)

        self._drop_incomplete_keys()
        self.logger.info("Retrieved data for keys: %s", sorted(set(self.retrieved_data) | set(self.retrieved_data_ref)))

    def _retrieve_dataset(self, reader, retrieve_args, level, key, storage, label):
        """Retrieve the dataset"""
        try:
            data = reader.retrieve(**retrieve_args)
            # TODO: Generalize this to handle other pressure levels naming conventions
            # This works now under the assumption that the pressure level is in the coordinates
            # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
            if level is not None and "plev" in data.coords:
                data = data.isel(plev=0, drop=True)
            # Apply regridding if configured
            # If regrid=False AQUA sets tgt_grid_name to None
            if reader.tgt_grid_name:
                self.logger.debug("Applying regridding to %s data for %s towards %s", label, key, reader.tgt_grid_name)
                data = reader.regrid(data)
            storage[key] = data
            self.logger.debug("Retrieved %s data for %s", label, key)
            return data
        except Exception as exc:
            self.logger.error("Failed to retrieve %s data for %s: %s", label, key, exc, exc_info=True)
            return None

    def _drop_incomplete_keys(self):
        """Drop the incomplete keys"""
        keys_data = set(self.retrieved_data.keys())
        keys_ref = set(self.retrieved_data_ref.keys())
        if not keys_data or not keys_ref:
            return None
        missing = keys_data ^ keys_ref
        if not missing:
            return None
        self.logger.warning("Dropping variables with incomplete pairings: %s", sorted(missing))
        for key in missing:
            self.retrieved_data.pop(key, None)
            self.retrieved_data_ref.pop(key, None)

    def compute_climatology(self, save_fig=True, save_data=False):
        """Compute climatology maps for available datasets"""
        if not hasattr(self, "retrieved_data") and not hasattr(self, "retrieved_data_ref"):
            raise RuntimeError("Data not retrieved. Call 'retrieve' before computing climatologies.")

        # Initialize the results
        results = {}

        available_keys = sorted(set(self.retrieved_data) | set(self.retrieved_data_ref))
        if not available_keys:
            self.logger.warning("No data to process.")
            return results

        # Process each key
        for key in available_keys:
            base_var, level = self._parse_processed_key(key)
            climatologies = self._prepare_climatologies(key, base_var)
            if not climatologies:
                continue

            fig = None
            try:
                fig = self._plot_climatologies(base_var, level, climatologies)
            except Exception as exc:
                self.logger.error("Failed to plot climatology for %s: %s", key, exc, exc_info=True)
                if fig is not None:
                    plt.close(fig)
                continue

            combined = xr.Dataset({
                entry["role"]: entry["data"] for entry in climatologies
            })
            combined.attrs["variable"] = base_var
            if level is not None:
                combined.attrs["level"] = level
            results[key] = combined

            if save_fig:
                self._save_figure(fig, key)
            plt.close(fig)

        return results

    def _prepare_climatologies(self, key, base_var):
        """Prepare the climatologies"""
        climatologies = []
        for entry in self._dataset_entries():
            dataset = entry["data"].get(key)
            if dataset is None or base_var not in dataset:
                continue

            try:
                climatology = entry["reader"].timmean(dataset[[base_var]])
            except Exception as exc:
                self.logger.error("Failed to compute climatology for %s (%s): %s", key, entry["role"], exc, exc_info=True)
                continue

            # Safety check to ensure the time dimension is removed
            data_array = climatology[base_var].squeeze(drop=True)
            if "time" in data_array.dims: 
                data_array = data_array.mean(dim="time")

            climatologies.append({"role": entry["role"], "label": entry["label"], "data": data_array})

        # TODO: Do I really need to align the grids?
        # if len(climatologies) > 1:
        #     try:
        #         aligned = xr.align(*[item["data"] for item in climatologies], join="inner")
        #     except Exception as exc:
        #         self.logger.error("Failed to align climatologies for %s: %s", key, exc, exc_info=True)
        #         return []
        #     for aligned_da, item in zip(aligned, climatologies):
        #         item["data"] = aligned_da

        return climatologies

    def _dataset_entries(self):
        """Get the dataset entries"""
        entries = []
        if self.reader_data_ref is not None:
            entries.append({"role": "reference", "label": self.config.get("data_ref", {}).get("label", "Reference"),
                            "data": getattr(self, "retrieved_data_ref", {}), "reader": self.reader_data_ref})
        if self.reader_data is not None:
            entries.append({"role": "experiment", "label": self.config.get("data", {}).get("label", "Experiment"),
                            "data": getattr(self, "retrieved_data", {}), "reader": self.reader_data})
        return entries

    def _plot_climatologies(self, base_var, level, climatologies):
        """Plot the climatologies"""
        # Set the number of panels
        n_panels = len(climatologies)
        if n_panels == 0:
            raise ValueError("No climatologies to plot.")

        # CHeck for arrays full of NaNs
        valid_arrays = [entry["data"].values for entry in climatologies if np.isfinite(entry["data"].values).any()]
        if not valid_arrays:
            raise ValueError("All climatology arrays contain only NaNs.")

        # Use configured vmin/vmax if provided, otherwise compute automatically
        if self.plot_settings.vmin is not None:
            vmin = self.plot_settings.vmin
        else:
            vmin = min(float(np.nanmin(values)) for values in valid_arrays)
        
        if self.plot_settings.vmax is not None:
            vmax = self.plot_settings.vmax
        else:
            vmax = max(float(np.nanmax(values)) for values in valid_arrays)
        
        # If the minimum and maximum values are close, add a small delta
        if np.isclose(vmin, vmax):
            delta = abs(vmin) * 0.1 if vmin != 0.0 else 1.0
            vmin -= delta
            vmax += delta

        width, height = self.plot_settings.figure_size
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(width * n_panels, height),
                                 subplot_kw={"projection": self.plot_settings.projection})
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        axes = np.atleast_1d(axes)

        # Create normalization for discrete colorbar if requested
        if self.plot_settings.discrete_colorbar:
            levels = np.linspace(vmin, vmax, self.plot_settings.n_colorbar_levels)
            norm = mcolors.BoundaryNorm(levels, ncolors=256)
        else:
            norm = None

        mesh = None
        for ax, entry in zip(axes, climatologies):
            data_array = entry["data"].squeeze(drop=True)
            lon_name, lat_name = coord_names(data_array)
            if not lon_name or not lat_name:
                raise ValueError("Unable to determine lat/lon coordinates for plotting.")

            plot_data = data_array.transpose(lat_name, lon_name)
            lon = plot_data[lon_name].values
            lat = plot_data[lat_name].values
            lon2d, lat2d = np.meshgrid(lon, lat)

            if self.plot_settings.add_coastlines:
                ax.coastlines()

            # Only pass vmin/vmax when not using a custom norm
            pcolormesh_kwargs = {"transform": ccrs.PlateCarree(),
                                 "cmap": self.plot_settings.cmap,
                                 "shading": "auto"}
            if norm is not None:
                pcolormesh_kwargs["norm"] = norm
            else:
                pcolormesh_kwargs["vmin"] = vmin
                pcolormesh_kwargs["vmax"] = vmax
            
            mesh = ax.pcolormesh(lon2d, lat2d, plot_data.values, **pcolormesh_kwargs)
            ax.set_title(entry["label"])

        # Create colorbar below the plots
        if self.plot_settings.colorbar_orientation == "horizontal":
            # For horizontal colorbar below, use cax to place it explicitly
            cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
            colorbar = fig.colorbar(mesh, cax=cbar_ax, orientation=self.plot_settings.colorbar_orientation)
        else:
            # For vertical colorbar, use standard positioning
            colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation=self.plot_settings.colorbar_orientation,
                                    pad=0.05)
        cbar_label = cbar_get_label(climatologies[0]["data"])
        if cbar_label:
            colorbar.set_label(cbar_label)

        fig.suptitle(self._build_title(base_var, level))
        fig.tight_layout()
        return fig

    def _build_title(self, base_var, level):
        """Build the title"""
        level_part = f" at level {level}" if level is not None else ""
        return f"Spatial Climatology: {base_var}{level_part}".strip()

    @staticmethod
    def _normalise_levels(levels):
        """Normalise the levels"""
        if levels is None:
            return [None]
        if isinstance(levels, (int, float)):
            return [int(levels)]
        if isinstance(levels, list):
            return [int(level) for level in levels]
        raise ValueError(f"Unsupported level specification: {levels}")

    @staticmethod
    def _build_key(var_name, level):
        """Build the key"""
        return f"{var_name}_{level}" if level is not None else var_name

    @staticmethod
    def _parse_processed_key(key):
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

    @staticmethod
    def _normalise_extension(extension):
        ext = extension.lower().lstrip(".")
        if not ext:
            return "pdf"
        else:
            return ext

    def _save_figure(self, fig, processed_key):
        """Save the figure"""
        filename = self._generate_filename(processed_key, "figure", "spatial_agg")
        try:
            fig.savefig(filename)
            self.logger.info("Saved climatology figure to %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save figure %s: %s", filename, exc, exc_info=True)

    def _generate_filename(self, processed_key, output_type, suffix):
        """Generate the filename"""
        data_cfg = self.config.get("data", {})
        ref_cfg = self.config.get("data_ref", {})

        filename_parts = [self._sanitize_filename_part(data_cfg.get("model", "model")),
                          self._sanitize_filename_part(data_cfg.get("exp", "exp")),
                          self._sanitize_filename_part(data_cfg.get("source", "src")),
                          "vs",
                          self._sanitize_filename_part(ref_cfg.get("model", "refmodel")),
                          self._sanitize_filename_part(ref_cfg.get("exp", "refexp")),
                          processed_key,
                          self._sanitize_filename_part(self.startdate),
                          self._sanitize_filename_part(self.enddate),
                          suffix]

        filename = "_".join(filter(None, filename_parts))

        if output_type == "figure":
            extension = f".{self.figure_format}"
            directory = self.outputdir_fig
        else:
            raise ValueError(f"Unsupported output type '{output_type}'.")

        if directory is None:
            directory = "."

        return os.path.join(directory, f"{filename}{extension}")

    @staticmethod
    def _sanitize_filename_part(value):
        """Sanitize the filename part"""
        text = str(value) if value is not None else ""
        for char in [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']:
            text = text.replace(char, "_")
        return text