import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.exceptions import NotEnoughDataError
from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import load_yaml
from aqua.core.fldstat.area_selection import AreaSelection


class SeasonalCycle:
    """
    Seasonal Cycle Diagnostic
    
    This class provides methods to compute the seasonal cycle of a given variable.
    It also allows for the saving of the data and figures to a specified directory.

    Args:
        config (Any): The configuration for the diagnostic.
        loglevel (str): The log level to be used. Default is "WARNING".
    """
    def __init__(self, config, loglevel = "WARNING"):
        """Initialize the Seasonal Cycle Diagnostic"""
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "SeasonalCycle")

        # Load the configuration
        if isinstance(config, str):
            self.config: Dict[str, Any] = load_yaml(config)
        else:
            self.config = config

        self._validate_config()

        # Get the start and end dates
        self.startdate = self.config["dates"]["startdate"]
        self.enddate = self.config["dates"]["enddate"]

        # Output directory for figures
        self.outputdir_fig = self.config.get("outputdir_fig", "./figs_seasonal_cycle")
        self.outputdir_data = self.config.get("outputdir_data", "./data_seasonal_cycle")
        self.figure_format = self._normalise_extension(self.config.get("figure_format", "pdf"))

        os.makedirs(self.outputdir_fig, exist_ok=True)
        if self.outputdir_data:
            os.makedirs(self.outputdir_data, exist_ok=True)

        self.data_label = self._infer_label(self.config.get("data", {}))
        self.data_ref_label = self._infer_label(self.config.get("data_ref", {}))
        self.regions = self._parse_regions(self.config.get("regions"))

        # Initialize the reader
        self._reader()

    def _validate_config(self):
        """Validate the configuration"""
        # Validate the required sections
        required = ["data", "data_ref", "dates", "variables"]
        missing = [section for section in required if section not in self.config]
        if missing:
            raise ValueError(f"Missing required configuration sections: {', '.join(missing)}")

        # Validate the dates
        for key in ("startdate", "enddate"):
            if not self.config["dates"].get(key):
                raise ValueError(f"Configuration must provide dates.{key}")

        # Validate the variables
        variables = self.config.get("variables", [])
        if not isinstance(variables, list) or not variables:
            raise ValueError("Configuration must provide a non-empty 'variables' list")

    def _reader(self):
        cfg_data = self.config.get("data", {})
        cfg_ref = self.config.get("data_ref", {})

        self.reader_data = Reader(catalog=cfg_data.get("catalog"),
                                  model=cfg_data.get("model"),
                                  exp=cfg_data.get("exp"),
                                  source=cfg_data.get("source"),
                                  regrid=cfg_data.get("regrid"),
                                  fix=cfg_data.get("fix"),
                                  areas=cfg_data.get("areas"),
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel)

        self.reader_data_ref = Reader(catalog=cfg_ref.get("catalog"),
                                      model=cfg_ref.get("model"),
                                      exp=cfg_ref.get("exp"),
                                      source=cfg_ref.get("source"),
                                      regrid=cfg_ref.get("regrid"),
                                      fix=cfg_ref.get("fix"),
                                      areas=cfg_ref.get("areas"),
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel)

    def retrieve(self):
        """Retrieve the data"""
        variables = self.config.get("variables", [])
        self.retrieved_data: Dict[str, xr.Dataset] = {}
        self.retrieved_data_ref: Dict[str, xr.Dataset] = {}

        # Retrieve the data
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

                data_ref = self._retrieve_single(self.reader_data_ref, retrieve_args, key, is_reference=True)
                data = self._retrieve_single(self.reader_data, retrieve_args, key, is_reference=False)

                if data_ref is None or data is None:
                    continue

                self.retrieved_data_ref[key] = data_ref
                self.retrieved_data[key] = data

        # Identify and drop mismatched variables
        missing = set(self.retrieved_data_ref) ^ set(self.retrieved_data)
        if missing:
            for key in missing:
                self.retrieved_data_ref.pop(key, None)
                self.retrieved_data.pop(key, None)
            if missing:
                self.logger.warning("Dropped variables with incomplete data: %s", sorted(missing))

        if not self.retrieved_data:
            self.logger.warning("No variables retrieved successfully")

    def _retrieve_single(self, reader, retrieve_args, key, is_reference):
        """Retrieve a single variable"""
        # Set the label
        label = "reference" if is_reference else "experiment"
        try:
            # Retrieve the data
            dataset = reader.retrieve(**retrieve_args)
            # Get the level
            level = retrieve_args.get("level")
            var = retrieve_args["var"]

            # TODO: Generalize this to handle other pressure levels naming conventions
            # This works now under the assumption that the pressure level is in the coordinates
            # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
            if level is not None and "plev" in dataset.coords:
                dataset = dataset.isel(plev=0, drop=True)
            tgt_grid = getattr(reader, "tgt_grid_name", None)

            # Apply regridding if configured
            # If regrid=False AQUA sets tgt_grid_name to None
            if tgt_grid:
                if hasattr(reader, "regridder"):
                    self.logger.debug("Applying regridding to %s data for %s", label, key)
                    dataset = reader.regrid(dataset)
                else:
                    self.logger.debug("Skipping regridding for %s data for %s; reader has no regridder", label, key)

            # Check if the variable is in the dataset
            if var not in dataset:
                self.logger.warning("Variable '%s' missing in retrieved %s dataset for key %s", var, label, key)
                return None
            return dataset

        except Exception as exc:
            self.logger.error("Failed to retrieve %s data for %s: %s", label, key, exc, exc_info=True)
            return None

    def compute(self, save_fig = True, save_data = False):
        """Compute the seasonal cycle"""
        if not hasattr(self, "retrieved_data") or not hasattr(self, "retrieved_data_ref"):
            raise RuntimeError("Data not retrieved. Call 'retrieve' before compute().")

        results = {}
        for key in sorted(self.retrieved_data):
            if key not in self.retrieved_data_ref:
                continue

            # Get the data and reference data
            data = self.retrieved_data[key]
            data_ref = self.retrieved_data_ref[key]
            base_var, level = self._parse_processed_key(key)

            # Check if the variable is in the data and reference data
            if base_var not in data or base_var not in data_ref:
                self.logger.warning("Variable '%s' missing for key %s; skipping", base_var, key)
                continue

            # Process each region
            for region in self.regions:
                region_key = f"{key}__{region['name']}"
                try:
                    # Compute the seasonal cycle for the region
                    result = self._compute_region_cycle(data[base_var],
                                                        data_ref[base_var],
                                                        base_var,
                                                        level,
                                                        region,
                                                        key,
                                                        save_fig,
                                                        save_data)

                except NotEnoughDataError as exc: # If the data is not enough for the seasonal cycle, skip the region
                    self.logger.warning("Skipping %s (%s): %s", key, region["name"], exc)
                    continue
                except Exception as exc:
                    self.logger.error("Failed seasonal cycle for %s (%s): %s", key, region["name"], exc, exc_info=True)
                    continue

                if result is not None:
                    results[region_key] = result

        return results

    def _compute_region_cycle(self, data, data_ref, base_var, level, region, key,
                              save_fig, save_data):
        """Compute the seasonal cycle for a given region"""
        # Compute the regional timeseries for the data and reference data
        da_data = self._compute_region_timeseries(self.reader_data, data, region)
        da_ref = self._compute_region_timeseries(self.reader_data_ref, data_ref, region)

        # Compute the monthly means for the data and reference data
        monthly_data = self.reader_data.timmean(da_data, freq="MS", exclude_incomplete=True)
        monthly_ref = self.reader_data_ref.timmean(da_ref, freq="MS", exclude_incomplete=True)

        # Drop missing values
        monthly_data = monthly_data.dropna("time", how="all")
        monthly_ref = monthly_ref.dropna("time", how="all")

        # Check if the data is enough for the seasonal cycle
        if monthly_data.sizes.get("time", 0) < 12:
            raise NotEnoughDataError("Experiment data shorter than one year of monthly means")
        if monthly_ref.sizes.get("time", 0) < 12:
            raise NotEnoughDataError("Reference data shorter than one year of monthly means")

        # Align the monthly means to ensure they cover the same time range
        monthly_data, monthly_ref = xr.align(monthly_data, monthly_ref, join="inner")
        if monthly_data.sizes.get("time", 0) < 12:
            raise NotEnoughDataError("Not enough overlapping monthly means between experiment and reference")

        # Compute the seasonal cycle for the data and reference data
        seasonal_data = monthly_data.groupby("time.month").mean("time")
        seasonal_ref = monthly_ref.groupby("time.month").mean("time")

        # Check if the seasonal cycle is enough for the data and reference data
        if seasonal_data.sizes.get("month", 0) < 12:
            raise NotEnoughDataError("Experiment seasonal cycle missing months")
        if seasonal_ref.sizes.get("month", 0) < 12:
            raise NotEnoughDataError("Reference seasonal cycle missing months")

        # Compute the anomaly for the data and reference data
        anomaly_data = seasonal_data - seasonal_data.mean("month")
        anomaly_ref = seasonal_ref - seasonal_ref.mean("month")

        result = xr.Dataset({"data": seasonal_data,
                             "data_ref": seasonal_ref,
                             "data_anomaly": anomaly_data,
                             "data_ref_anomaly": anomaly_ref})

        result.attrs["variable"] = base_var
        result.attrs["region"] = region["name"]
        if level is not None:
            result.attrs["level"] = level

        if save_fig:
            self._plot_seasonal_cycle(result, base_var, level, region, key)

        if save_data and self.outputdir_data:
            self._save_data(result, key, region)

        return result

    def _compute_region_timeseries(self, reader, data, region):
        """Compute the regional timeseries for a given region"""
        lat_limits = region.get("lat")
        lon_limits = region.get("lon")

        if lat_limits is not None or lon_limits is not None:
            data_selected = AreaSelection(loglevel=self.loglevel).select_area(data,
                                           lat=lat_limits,
                                           lon=lon_limits,
                                           drop=False)
        else:
            data_selected = data

        # Compute the field mean for the data
        averaged = reader.fldmean(data_selected)
        if isinstance(averaged, xr.Dataset):
            averaged = averaged[data.name]

        return averaged.squeeze()

    def _plot_seasonal_cycle(self, result, base_var, level, region, key):
        """Plot the seasonal cycle for a given region"""
        months = result["data"]["month"].values
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        label_lookup = dict(zip(range(1, 13), month_labels))
        tick_labels = [label_lookup.get(int(m), str(int(m))) for m in months]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(months, result["data"].values, label=self.data_label, marker="o")
        axes[0].plot(months, result["data_ref"].values, label=self.data_ref_label, marker="o")
        axes[0].set_ylabel(self._build_ylabel(result["data"]))
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(months, result["data_anomaly"].values, label=self.data_label, marker="o")
        axes[1].plot(months, result["data_ref_anomaly"].values, label=self.data_ref_label, marker="o")
        axes[1].set_ylabel(self._build_ylabel(result["data_anomaly"], default=f"{base_var} anomaly"))
        axes[1].set_xticks(months)
        axes[1].set_xticklabels(tick_labels)
        axes[1].legend()
        axes[1].grid(True)

        title = self._build_title(base_var, level, region)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        filename = self._generate_filename(key, region, output_type="figure")
        try:
            fig.savefig(filename)
            self.logger.info("Saved seasonal cycle figure to %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save figure %s: %s", filename, exc, exc_info=True)
        finally:
            plt.close(fig)

    def _save_data(self, data, key, region):
        """Save the seasonal cycle data to a file"""
        filename = self._generate_filename(key, region, output_type="netcdf")
        try:
            data.to_netcdf(filename)
            self.logger.info("Saved seasonal cycle data to %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save NetCDF %s: %s", filename, exc, exc_info=True)

    def _generate_filename(self, key, region, output_type):
        """Generate the filename for the seasonal cycle data"""
        data_cfg = self.config.get("data", {})
        ref_cfg = self.config.get("data_ref", {})

        parts = [self._sanitize_filename_part(data_cfg.get("model", "model")),
                 self._sanitize_filename_part(data_cfg.get("exp", "exp")),
                 self._sanitize_filename_part(data_cfg.get("source", "src")),
                 "vs",
                 self._sanitize_filename_part(ref_cfg.get("model", "refmodel")),
                 self._sanitize_filename_part(ref_cfg.get("exp", "refexp")),
                 self._sanitize_filename_part(key),
                 self._sanitize_filename_part(region["name"]),
                 self._sanitize_filename_part(self.startdate),
                 self._sanitize_filename_part(self.enddate),
                 "seasonal_cycle"]

        filename = "_".join(filter(None, parts))

        if output_type == "figure":
            extension = f".{self.figure_format}"
            directory = self.outputdir_fig
        elif output_type == "netcdf":
            extension = ".nc"
            directory = self.outputdir_data or "."
        else:
            raise ValueError(f"Unsupported output type '{output_type}'")

        return os.path.join(directory, f"{filename}{extension}")

    @staticmethod
    def _build_title(base_var, level, region):
        """Build the title for the seasonal cycle"""
        level_part = f", level {level}" if level is not None else ""
        region_part = region["name"]
        return f"Seasonal cycle for {base_var}{level_part} ({region_part})"

    @staticmethod
    def _build_ylabel(data, default=None):
        """Build the y-label for the seasonal cycle"""
        units = data.attrs.get("units")
        name = default or data.name or "Value"
        if units:
            return f"{name} ({units})"
        return name

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
        """Build the key for the seasonal cycle"""
        return f"{var_name}_{level}" if level is not None else var_name

    @staticmethod
    def _parse_processed_key(key):
        """Parse the processed key into base variable name and level"""
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
        """Normalise the extension"""
        ext = (extension or "").lower().lstrip(".")
        return ext or "pdf"

    @staticmethod
    def _sanitize_filename_part(value):
        """Sanitize the filename part"""
        text = str(value) if value is not None else ""
        for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", "."]:
            text = text.replace(char, "_")
        return text

    def _infer_label(self, cfg):
        """Infer the label for the seasonal cycle"""
        return cfg.get("label") or cfg.get("model") or cfg.get("exp") or "Dataset"

    def _parse_regions(self, regions_cfg):
        """Parse the regions"""
        regions = []
        if regions_cfg is None:
            return [{"name": "global", "lat": None, "lon": None}]

        if isinstance(regions_cfg, list):
            for idx, region in enumerate(regions_cfg):
                if not isinstance(region, dict):
                    continue
                name = region.get("name") or f"region_{idx+1}"
                regions.append({"name": name,
                                "lat": self._validate_bounds(region.get("lat")),
                                "lon": self._validate_bounds(region.get("lon"))})
        elif isinstance(regions_cfg, dict):
            for name, region in regions_cfg.items():
                if not isinstance(region, dict):
                    continue
                regions.append({"name": name,
                                "lat": self._validate_bounds(region.get("lat")),
                                "lon": self._validate_bounds(region.get("lon"))})

        if not regions:
            regions.append({"name": "global", "lat": None, "lon": None})
        return regions

    @staticmethod
    def _validate_bounds(bounds):
        """Validate the bounds"""
        if bounds is None:
            return None
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            return [float(bounds[0]), float(bounds[1])]
        raise ValueError(f"Region bounds must be lists of two values, got {bounds}")
