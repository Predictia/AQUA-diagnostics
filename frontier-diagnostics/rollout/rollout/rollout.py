import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.exceptions import NotEnoughDataError
from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import load_yaml
from smmregrid import GridInspector


class Rollout:

    def __init__(self, config: Any, loglevel: str = "WARNING"):
        """
        Initialize the Rollout diagnostic class.
        
        Args:
            config: Configuration for the Rollout diagnostic
            loglevel (str, optional): Logging level. Defaults to 'WARNING'.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "Rollout")

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
        self.outputdir_fig = self.config.get("outputdir_fig", "./figs_rollout")
        self.outputdir_data = self.config.get("outputdir_data", "./data_rollout")
        # Figure format
        self.figure_format = self._normalise_extension(self.config.get("figure_format", "pdf"))

        # Annual aggregation
        annual_cfg = self.config.get("annual", False)

        if isinstance(annual_cfg, dict):
            self.plot_annual = bool(annual_cfg.get("plot", True))
        else:
            self.plot_annual = bool(annual_cfg)

        # Create output directories
        os.makedirs(self.outputdir_fig, exist_ok=True)
        if self.outputdir_data:
            os.makedirs(self.outputdir_data, exist_ok=True)

        self._reader()

    # TODO: Move this to a shared utility module.
    def _validate_config(self):
        """Validate the configuration."""
        required_sections = ["data", "data_ref", "dates", "variables"]
        missing = [section for section in required_sections if section not in self.config]
        if missing:
            raise ValueError("Missing required configuration sections: " + ", ".join(missing))

        for key in ("startdate", "enddate"):
            if not self.config["dates"].get(key):
                raise ValueError(f"Configuration must provide dates ({key})")

    def _reader(self):
        """Initializes the Reader class for both reference and main data"""

        config_data = self.config.get("data", {})
        config_data_ref = self.config.get("data_ref", {})

        self.reader_data = Reader(
            catalog=config_data.get("catalog"),
            model=config_data.get("model"),
            exp=config_data.get("exp"),
            source=config_data.get("source"),
            regrid=config_data.get("regrid"),
            fix=config_data.get("fix"),
            areas=config_data.get("areas"),
            startdate=self.startdate,
            enddate=self.enddate,
            loglevel=self.loglevel,
        )

        self.reader_data_ref = Reader(
            catalog=config_data_ref.get("catalog"),
            model=config_data_ref.get("model"),
            exp=config_data_ref.get("exp"),
            source=config_data_ref.get("source"),
            regrid=config_data_ref.get("regrid"),
            fix=config_data_ref.get("fix"),
            areas=config_data_ref.get("areas"),
            startdate=self.startdate,
            enddate=self.enddate,
            loglevel=self.loglevel,
        )

        self.logger.debug('Reader classes initialized for data_ref, data, and climatology')

    def retrieve(self):
        """
        Retrieves all variables defined in the configuration.
        If regridding is configured, data will be regridded automatically after retrieval.
        """

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

            # Process each level
            for level in self._normalise_levels(var_cfg.get("level")):
                key = self._build_key(var_name, level)
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
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_data_ref.tgt_grid_name is not None:
                        self.logger.debug(f"Applying regridding to reference data for {key}")
                        data_ref = self.reader_data_ref.regrid(data_ref)
                    self.retrieved_data_ref[key] = data_ref
                    self.logger.debug("Retrieved reference data for %s", key)
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
                        self.logger.debug(f"Applying regridding to experiment data for {key}")
                        data = self.reader_data.regrid(data)
                    self.retrieved_data[key] = data
                    self.logger.debug("Retrieved experiment data for %s", key)
                except Exception as exc:
                    self.logger.error("Failed to retrieve experiment data for %s: %s", key, exc, exc_info=True)
                    self.retrieved_data_ref.pop(key, None)

        # Identify and drop mismatched variables
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
            self.logger.info(f"Experiment data regridded to: {self.reader_data.tgt_grid_name}")

    def compute_rollout(self, save_fig, save_data):
        """
        Computes rollout plots and optionally saves outputs.
        
        Args:
            save_fig (bool, optional): Whether to save figures. Defaults to True.
            save_data (bool, optional): Whether to save data. Defaults to False.
        """

        if not hasattr(self, "retrieved_data") or not hasattr(self, "retrieved_data_ref"):
            raise RuntimeError("Data not retrieved. Call 'retrieve' before computing rollouts.")

        results = {}
        config_data = self.config.get("data", {})
        config_data_ref = self.config.get("data_ref", {})

        # Process each variable
        for key in sorted(self.retrieved_data):
            if key not in self.retrieved_data_ref:
                self.logger.warning("Variable '%s' not present in reference data; skipping.", key)
                continue

            data = self.retrieved_data[key]
            data_ref = self.retrieved_data_ref[key]

            base_var, level = self._parse_processed_key(key)
            if base_var not in data or base_var not in data_ref:
                self.logger.warning("Variable '%s' not present in datasets for key '%s'; skipping.", base_var, key)
                continue

            # Identify horizontal dimensions for spatial averaging
            fldmean_kwargs = {}
            horizontal_dims = []
            try:
                grid_types = GridInspector(data).get_gridtype()
                if grid_types:
                    horizontal_dims = [dim for dim in grid_types[0].horizontal_dims if dim in data.dims]
                    if not horizontal_dims:
                        self.logger.warning("GridInspector did not return usable horizontal dims for %s. Falling back to non-time dims.", key)
            except Exception as exc:
                self.logger.warning("Could not infer horizontal dims for fldmean on %s: %s", key, exc)

            if horizontal_dims:
                fldmean_kwargs["dims"] = horizontal_dims
                
                # Synchronise the underlying FldStat horizontal dims with the data
                fldstat_obj_data = self.reader_data.tgt_fldstat or self.reader_data.src_fldstat # Use whichever one exist
                if fldstat_obj_data is not None:
                    fldstat_obj_data.horizontal_dims = horizontal_dims
                fldstat_obj_ref = self.reader_data_ref.tgt_fldstat or self.reader_data_ref.src_fldstat # Use whichever one exist
                if fldstat_obj_ref is not None:
                    fldstat_obj_ref.horizontal_dims = horizontal_dims
            
            ts_target = self.reader_data_ref.fldmean(data_ref, **fldmean_kwargs)
            ts_exp = self.reader_data.fldmean(data, **fldmean_kwargs)

            # Ensure residual spatial dims are collapsed (e.g. ncells)
            # This is necessary to work on HealPix like grids
            for dim in horizontal_dims:
                if dim in ts_target.dims:
                    ts_target = ts_target.mean(dim=dim)
                if dim in ts_exp.dims:
                    ts_exp = ts_exp.mean(dim=dim)

            # Temporal subsetting of the data
            try:
                ts_target = ts_target.sel(time=slice(self.startdate, self.enddate))
                ts_exp = ts_exp.sel(time=slice(self.startdate, self.enddate))
            except Exception as exc:
                self.logger.warning("Failed to subset time range for %s; proceeding with original data: %s", key, exc)

            # Extract the variable from the time series
            try:
                da_target = ts_target[base_var]
                da_exp = ts_exp[base_var]
            except KeyError:
                self.logger.warning("Time series does not contain variable '%s' for key '%s'; skipping.", base_var, key)
                continue

            # Align the time series to ensure they cover the same time range
            try:
                da_target, da_exp = xr.align(da_target, da_exp, join="inner")
            except Exception as exc:
                self.logger.error("Failed to align time series for %s: %s", key, exc, exc_info=True)
                continue

            # Check if the time series is empty
            if da_target.size == 0:
                self.logger.warning("No overlapping timestamps for %s; skipping.", key)
                continue

            # Create the figure and plot the time series
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(da_target["time"].values,
                    da_target.values,
                    label=config_data_ref.get("label", "Target"),
                    color="black",
                    linewidth=2)
            ax.plot(da_exp["time"].values,
                    da_exp.values,
                    label=config_data.get("label", "Experiment"),
                    color="blue",
                    linewidth=2)

            # Set the labels and title
            ax.set_xlabel("Time")
            units = da_target.attrs.get("units") or da_exp.attrs.get("units")
            ylabel = f"{base_var} ({units})" if units else base_var
            ax.set_ylabel(ylabel)

            title = self._build_title(base_var, level, config_data, config_data_ref)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
            fig.tight_layout()

            # Create the combined dataset
            combined = xr.Dataset({"target": da_target, "experiment": da_exp})
            combined.attrs["variable"] = base_var
            if level is not None:
                combined.attrs["level"] = level

            results[key] = combined

            if save_fig:
                self._save_figure(fig, key)
            if save_data:
                self._save_data(combined, key)

            plt.close(fig)

            # Compute annual rollout if configured
            if self.plot_annual:
                try:
                    annual_result = self._compute_annual_timeseries(ts_target, ts_exp,
                                                                    base_var, level,
                                                                    config_data, config_data_ref,
                                                                    key, save_fig, save_data)
                except NotEnoughDataError as exc:
                    self.logger.warning("Skipping annual rollout for %s: %s", key, exc)
                except Exception as exc:
                    self.logger.error("Failed to compute annual rollout for %s: %s", key, exc, exc_info=True)
                else:
                    if annual_result is not None:
                        results[f"{key}_annual"] = annual_result

        return results

    def _compute_annual_timeseries(self, ts_target, ts_exp, base_var, level,
                                   config_data, config_data_ref,
                                   processed_key, save_fig, save_data):
        """Computes the annual rollout timeseries."""

        # Extract the variable from the time series
        target_ds = ts_target[[base_var]] if base_var in ts_target else ts_target
        exp_ds = ts_exp[[base_var]] if base_var in ts_exp else ts_exp

        try:
            annual_target = self.reader_data_ref.timmean(target_ds, freq="annual", exclude_incomplete=True)
            annual_exp = self.reader_data.timmean(exp_ds, freq="annual", exclude_incomplete=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to compute annual means for {processed_key}: {exc}") from exc

        da_target = annual_target[base_var]
        da_exp = annual_exp[base_var]

        # Drop missing values
        da_target = da_target.dropna("time", how="all")
        da_exp = da_exp.dropna("time", how="all")

        # Check if the time series is empty
        if da_target.sizes.get("time", 0) == 0:
            raise NotEnoughDataError(f"Not enough target data to compute a full annual mean for {processed_key}.")
        if da_exp.sizes.get("time", 0) == 0:
            raise NotEnoughDataError(f"Not enough experiment data to compute a full annual mean for {processed_key}.")

        # Align the time series to ensure they cover the same time range
        da_target, da_exp = xr.align(da_target, da_exp, join="inner")
        if da_target.sizes.get("time", 0) == 0:
            raise NotEnoughDataError(f"No overlapping annual periods between target and experiment for {processed_key}.")

        # Create the figure and plot the time series
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(da_target["time"].values, da_target.values, label=config_data_ref.get("label", "Target"), color="black", linewidth=2)
        ax.plot(da_exp["time"].values, da_exp.values, label=config_data.get("label", "Experiment"), color="blue")

        ax.set_xlabel("Time")
        units = da_target.attrs.get("units") or da_exp.attrs.get("units")
        ylabel = f"{base_var} ({units})" if units else base_var
        ax.set_ylabel(ylabel)

        base_title = self._build_title(base_var, level, config_data, config_data_ref).rstrip()
        title = f"{base_title} (Annual Mean)" if base_title else "Annual Mean"
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
        fig.tight_layout()

        # Create the combined dataset
        combined = xr.Dataset({"target": da_target, "experiment": da_exp})
        combined.attrs["variable"] = base_var
        combined.attrs["aggregation"] = "annual"
        if level is not None:
            combined.attrs["level"] = level

        if save_fig:
            self._save_figure(fig, f"{processed_key}_annual")
        if save_data:
            self._save_data(combined, f"{processed_key}_annual")

        plt.close(fig)

        return combined

    def _build_title(self, base_var, level,
                     config_data, config_data_ref):
        """Builds the title for the rollout plot."""
        model = config_data.get("model", "N/A")
        exp = config_data.get("exp", "N/A")
        source = config_data.get("source", "N/A")
        model_ref = config_data_ref.get("model", "N/A")
        exp_ref = config_data_ref.get("exp", "N/A")
        source_ref = config_data_ref.get("source", "N/A")

        level_part = f" at level {level}" if level is not None else ""
        return f"Rollout Comparison: {base_var}{level_part}\nModel: {model} Experiment: {exp} Source: {source} vs Model: {model_ref} Experiment: {exp_ref} Source: {source_ref}"

    # TODO: Move this to a shared utility module.
    @staticmethod
    def _normalise_levels(levels):
        if levels is None:
            return [None]
        if isinstance(levels, (int, float)):
            return [int(levels)]
        if isinstance(levels, list):
            return [int(level) for level in levels]
        raise ValueError(f"Unsupported level specification: {levels}")

    @staticmethod
    def _build_key(var_name, level):
        """Builds the key for the rollout plot."""
        return f"{var_name}_{level}" if level is not None else var_name

    # TODO: Move this to a shared utility module.
    @staticmethod
    def _parse_processed_key(key):
        """Parses the processed key into base variable name and level."""
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
        """Normalizes the extension of a file."""
        ext = extension.lower().lstrip(".")
        if not ext:
            return "pdf"
        return ext

    def _save_figure(self, fig, processed_key):
        """Saves the generated figure with a unique name based on config."""
        filename = self._generate_filename(processed_key, "figure", "rollout")
        try:
            fig.savefig(filename)
            self.logger.info("Saved rollout figure to %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save figure %s: %s", filename, exc, exc_info=True)

    def _save_data(self, data, processed_key):
        """Saves the generated data with a unique name based on config."""
        if not self.outputdir_data:
            self.logger.warning("Output data directory not configured; skipping NetCDF save for %s.", processed_key)
            return

        filename = self._generate_filename(processed_key, "netcdf", "rollout_timeseries")
        try:
            data.to_netcdf(filename)
            self.logger.info("Saved rollout data to %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save NetCDF %s: %s", filename, exc, exc_info=True)

    def _generate_filename(self, processed_key, output_type, suffix):
        """Generates a unique filename based on config."""
        data_cfg = self.config.get("data", {})
        ref_cfg = self.config.get("data_ref", {})

        filename_parts = [self._sanitize_filename_part(data_cfg.get("model", "model")),
                          self._sanitize_filename_part(data_cfg.get("exp", "exp")),
                          self._sanitize_filename_part(data_cfg.get("source", "src")),
                          "vs",
                          self._sanitize_filename_part(ref_cfg.get("model", "refmodel")),
                         self._sanitize_filename_part(ref_cfg.get("exp", "refexp")),
                         self._sanitize_filename_part(ref_cfg.get("source", "refsrc")),
                         self._sanitize_filename_part(processed_key),
                         self._sanitize_filename_part(self.startdate),
                         self._sanitize_filename_part(self.enddate),
                         suffix]

        filename = "_".join(filter(None, filename_parts))

        if output_type == "figure":
            extension = f".{self.figure_format}"
            directory = self.outputdir_fig
        elif output_type == "netcdf":
            extension = ".nc"
            directory = self.outputdir_data
        else:
            raise ValueError(f"Unsupported output type '{output_type}'.")

        if directory is None:
            directory = "."

        return os.path.join(directory, f"{filename}{extension}")


    # TODO: Move this to a shared utility module.
    @staticmethod
    def _sanitize_filename_part(value):
        """Sanitizes a string part for use in a filename."""
        text = str(value) if value is not None else ""
        for char in [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']:
            text = text.replace(char, "_")
        return text