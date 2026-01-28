import os
import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from aqua.core.graphics.multiple_maps import plot_maps
from aqua.core.graphics.single_map import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import load_yaml, to_list
from aqua.core.fldstat.area_selection import AreaSelection
from aqua.core.util.sci_util import select_season

class AMO:
    """
    Atlantic Multidecadal Oscillation (AMO) diagnostic.
    

    van Oldenborgh, G. J., te Raa, L. A., Dijkstra, H. A., & Philip, S. Y. (2009).
    Frequency-or amplitude-dependent effects of the Atlantic meridional overturning on
    the tropical Pacific Ocean. Ocean science, 5(3), 293-301.

    Trenberth, K. E., & Shea, D. J. (2006). Atlantic hurricanes and natural variability
    in 2005. Geophysical research letters, 33(12).
    """

    def __init__(self, config, loglevel: str = "WARNING"):
        """
        Initialize the AMO diagnostic.

        Args:
            config (str or dict): Configuration file or dictionary.
            loglevel (str): Logging level.
        """
        # Configure the logger
        self.logger = log_configure(log_name="AMO", log_level=loglevel)
        self.loglevel = loglevel
        self.config = load_yaml(config) if isinstance(config, str) else config

        # Validate the configuration
        required = ["data", "dates"]
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing configuration sections: {missing}")

        # Get the dates
        dates = self.config["dates"]
        self.startdate = dates.get("startdate")
        self.enddate = dates.get("enddate")
        if not self.startdate or not self.enddate:
            raise ValueError("Configuration must provide dates.startdate and dates.enddate")

        # Get the data configuration
        self.data_cfg = self.config["data"]
        self.data_label = self.data_cfg.get("label") or self.data_cfg.get("model") or "Dataset"

        # Get the field to use for the AMO computation
        self.field = self.config.get("field", "tos")
        
        # AMO-specific configuration
        amo_cfg = self.config.get("amo", {})
        self.amo_region = amo_cfg.get("region", {"lat": [25, 60], "lon": [-75, -7]})
        self.apply_detrending = amo_cfg.get("detrend", True)
        self.detrending_variable = amo_cfg.get("detrend_variable", "tos")
        self.apply_smoothing = amo_cfg.get("smooth", True) # Rolling mean
        self.window = amo_cfg.get("window", 5) # Years for the rolling mean
        
        # Get the reference data configuration
        self.data_ref_cfg = self.config.get("data_ref")
        self.data_ref_label = None
        if self.data_ref_cfg:
            self.data_ref_label = self.data_ref_cfg.get("label") or self.data_ref_cfg.get("model") or "Emulator"

        # Get the maps configuration
        self.maps_cfg = self.config.get("maps", {})
        self.compute_maps = self.maps_cfg.get("run", True)
        self.map_variables = self._ensure_list(self.maps_cfg.get("variables")) or [self.field]
        self.map_seasons = self._normalize_seasons(self._ensure_list(self.maps_cfg.get("seasons")) or ["annual"])
        self.save_combined_maps = self.maps_cfg.get("save_combined_maps", True)
        self.save_maps_netcdf = self.maps_cfg.get("save_netcdf", False)
        self.maps_fig_formats = self._ensure_list(self.maps_cfg.get("figure_format", "pdf")) or ["pdf"]
        self.maps_output_subdir = self.maps_cfg.get("output_subdir", "maps")

        # Initialize the reader
        self._reader()

        # Get the output directory
        self.output_dir = self.config.get("output_dir", "./amo_output")
        self.maps_output_dir = os.path.join(self.output_dir, self.maps_output_subdir)

        # Initialize some attributes 
        self.sst = None
        self.sst_ref = None
        self.index = None
        self.index_filtered = None
        self.index_ref = None
        self.index_filtered_ref = None
        self.regression_maps = {}
        self.correlation_maps = {}
        self._data_cache = {}

    @staticmethod
    def _ensure_list(value):
        """Ensure the value is a list."""
        if value is None:
            return None
        return to_list(value)

    @staticmethod
    def _normalize_seasons(seasons):
        """Normalize the seasons."""
        if not seasons:
            return []
        normalized = []
        seen = set()
        for season in seasons:
            if season is None:
                continue
            if isinstance(season, str) and season.lower() == "annual":
                key = "annual"
            else:
                key = str(season).upper()
            if key not in seen:
                normalized.append(key)
                seen.add(key)
        return normalized

    def _reader(self):
        """Initialize the reader."""
        cfg = self.data_cfg
        self.reader = Reader(catalog=cfg.get("catalog"),
                             model=cfg.get("model"),
                             exp=cfg.get("exp"),
                             source=cfg.get("source"),
                             regrid=cfg.get("regrid"),
                             fix=cfg.get("fix"),
                             startdate=self.startdate,
                             enddate=self.enddate,
                             loglevel=self.loglevel)

        if self.data_ref_cfg:
            cfg_ref = self.data_ref_cfg
            self.reader_ref = Reader(catalog=cfg_ref.get("catalog"),
                                     model=cfg_ref.get("model"),
                                     exp=cfg_ref.get("exp"),
                                     source=cfg_ref.get("source"),
                                     regrid=cfg_ref.get("regrid"),
                                     fix=cfg_ref.get("fix"),
                                     startdate=self.startdate,
                                     enddate=self.enddate,
                                     loglevel=self.loglevel)
        else:
            self.reader_ref = None

    def retrieve(self):
        """Load data (SST) and aggregate to monthly means."""
        field = self.field
        self.logger.info("Retrieving field '%s' for %s", field, self.data_label)
        self.sst = self.reader.retrieve(var=field)[field]
        self.sst = self.reader.timmean(self.sst, freq="MS")
        self.logger.info("Retrieved %s monthly points for %s", self.sst.sizes["time"], self.data_label)

        if self.reader_ref:
            label = self.data_ref_label or "Emulator"
            self.logger.info("Retrieving field '%s' for %s", field, label)
            self.sst_ref = self.reader_ref.retrieve(var=field)[field]
            self.sst_ref = self.reader_ref.timmean(self.sst_ref, freq="MS")
            self.logger.info("Retrieved %s monthly points for %s", self.sst_ref.sizes["time"], label)
        else:
            self.sst_ref = None

    def _select_region_mean(self, data, region, reader):
        """Select a region and compute the area-weighted mean."""
        lat = region["lat"]
        lon = region["lon"]
        self.logger.debug("Selecting box lat=%s, lon=%s", lat, lon)
        subset = AreaSelection().select_area(data, lat=lat, lon=lon, drop=False)
        mean = reader.fldmean(subset)
        return mean

    def compute_index(self):
        """Compute the AMO index."""
        # Compute the index
        self.index = self._compute_amo(self.sst, self.reader, name="amo")

        # Smooth the index if specified in the configuration
        if self.apply_smoothing:
            self.index_filtered = self._apply_filter(self.index)
        else:
            self.index_filtered = self.index

        # Compute the index for the reference data (if provided)
        if self.sst_ref is not None and self.reader_ref is not None:
            self.index_ref = self._compute_amo(self.sst_ref, self.reader_ref, name="amo_ref")
            if self.apply_smoothing:
                self.index_filtered_ref = self._apply_filter(self.index_ref)
            else:
                self.index_filtered_ref = self.index_ref
        else:
            self.index_ref = None
            self.index_filtered_ref = None

        # Generate the plot for the index
        self.plot_index()

        # Compute regression and correlation maps
        if self.compute_maps:
            self._compute_maps()

    def _compute_amo(self, sst_data, reader, name="amo"):
        """Compute the AMO index from SST data.
        
        Steps:
        - Compute monthly SST anomalies
        - Average over the AMO region (75W–7W, 25N–60N)
        - (Optional) Detrend 
        """
        self.logger.info("Computing AMO index")
        
        # Compute monthly SST anomalies
        anomalies = sst_data.groupby("time.month") - sst_data.groupby("time.month").mean(dim="time")
        
        # Average over the AMO region
        amo_index = self._select_region_mean(anomalies, self.amo_region, reader)
        
        # Detrend if specified in the configuration
        if self.apply_detrending:
            self.logger.info("Detrending AMO index")
            amo_index = self._detrend_by_global_mean(amo_index, anomalies, reader)
        
        return amo_index.rename(name)

    def _detrend_by_global_mean(self, amo_index, sst_anomalies, reader):
        """Detrend the AMO index by subtracting the global mean anomalies of a chosen variable.
        
        Currently implemented by removing the globally averaged monthly anomalies of
        ``self.detrending_variable`` from the regional AMO index, following
        Trenberth, K. E., & Shea, D. J. (2006). Atlantic hurricanes and natural variability
        in 2005. Geophysical research letters, 33(12).
        """
        # Retrieve detrending variable anomalies
        self.logger.info(f"Loading {self.detrending_variable} for detrending")
        detrending_variable_data = self.reader.retrieve(var=self.detrending_variable)[self.detrending_variable]
        detrending_variable_data = self.reader.timmean(detrending_variable_data, freq="MS")
        detrending_variable_data = detrending_variable_data.groupby("time.month") - detrending_variable_data.groupby("time.month").mean(dim="time")
        detrending_variable_data = self.reader.fldmean(detrending_variable_data)
        
        # Detrend the AMO index by subtracting the anomalies of the detrending variable
        detrended = amo_index - detrending_variable_data
        
        return detrended
            
    def _apply_filter(self, index):
        """Apply smoothing filter to the index."""
        window = self.window * 12
        filtered = index.rolling(time=window, center=True, min_periods=max(1, window // 2)).mean()
        filtered_name = f"{index.name}_filtered" if index.name else "amo_filtered"
        return filtered.rename(filtered_name)

    def _prepare_data_for_stats(self, var, use_reference):
        """Prepare data for computing regression and correlation maps."""
        cache_key = (var, use_reference)
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        reader = self.reader_ref if use_reference else self.reader
        if reader is None:
            return None

        if var == self.field:
            data = self.sst_ref if use_reference else self.sst
        else:
            dataset = reader.retrieve(var=var)
            data = dataset[var]
            data = reader.timmean(data, freq="MS")

        self._data_cache[cache_key] = data
        return data

    def _compute_stat_maps(self, var, season, use_reference=False):
        """Compute regression and correlation maps for a given variable and season."""
        index = self.index_ref if use_reference else self.index
        if index is None:
            return None, None

        data = self._prepare_data_for_stats(var=var, use_reference=use_reference)
        if data is None:
            return None, None

        data, index = self._select_season_if_needed(data, index, season)
        data = data.groupby("time.month") - data.groupby("time.month").mean(dim="time")

        # Compute the regression and correlation maps
        regression = xr.cov(index, data, dim="time") / index.var(dim="time", skipna=True)
        correlation = xr.corr(index, data, dim="time")
        return regression, correlation

    def _select_season_if_needed(self, data, index, season):
        """Select season if not annual."""
        if season == "annual":
            return data, index
        data_selected = select_season(data, season)
        index_selected = select_season(index, season)
        return data_selected, index_selected

    def _compute_maps(self):
        """Compute regression and correlation maps."""
        # Clear the dictionaries
        self.regression_maps.clear()
        self.correlation_maps.clear()

        # Compute the regression and correlation maps
        for var in self.map_variables:
            for season in self.map_seasons:
                reg_model, cor_model = self._compute_stat_maps(var=var, season=season, use_reference=False)
                reg_ref, cor_ref = self._compute_stat_maps(var=var, season=season, use_reference=True)

                if reg_model is not None:
                    self.regression_maps[(var, season, "model")] = reg_model
                if cor_model is not None:
                    self.correlation_maps[(var, season, "model")] = cor_model
                if reg_ref is not None:
                    self.regression_maps[(var, season, "reference")] = reg_ref
                if cor_ref is not None:
                    self.correlation_maps[(var, season, "reference")] = cor_ref

                if any([reg_model is not None, cor_model is not None, reg_ref is not None, cor_ref is not None]):
                    self._plot_maps(var=var, season=season,
                                    reg_model=reg_model, cor_model=cor_model,
                                    reg_ref=reg_ref, cor_ref=cor_ref)

    def _plot_maps(self, var, season, reg_model=None, cor_model=None,
                   reg_ref=None, cor_ref=None):
        """Plot regression and correlation maps for AMO diagnostic."""
        if not self.save_combined_maps and not self.save_maps_netcdf:
            return

        os.makedirs(self.maps_output_dir, exist_ok=True)
        season_suffix = season.lower()
        var_suffix = var

        def _map_title(map_type, label):
            title = f"AMO {map_type} ({var}) - {label}"
            if season != "annual":
                title += f" ({season})"
            return title

        def _map_filename(map_type, label=None, fmt="pdf"):
            base = f"amo_{map_type}_{var_suffix}_{season_suffix}"
            if label:
                safe_label = label.replace(" ", "_")
                base += f"_{safe_label}"
            return f"{base}.{fmt}"

        # Combined plots (model vs emulator)
        if self.save_combined_maps:
            label_ref = self.data_ref_label or "Emulator"
            if reg_model is not None and reg_ref is not None:
                fig = plot_maps([reg_model, reg_ref], contour=True, sym=False,
                                titles=[_map_title("regression", self.data_label),
                                        _map_title("regression", label_ref)],
                                return_fig=True, loglevel=self.loglevel)
                for fmt in self.maps_fig_formats:
                    fig.savefig(os.path.join(self.maps_output_dir, _map_filename("regression", "combined", fmt)))

            if cor_model is not None and cor_ref is not None:
                fig = plot_maps([cor_model, cor_ref], contour=True, sym=True,
                                titles=[_map_title("correlation", self.data_label),
                                        _map_title("correlation", label_ref)],
                                return_fig=True, loglevel=self.loglevel)
                for fmt in self.maps_fig_formats:
                    fig.savefig(os.path.join(self.maps_output_dir, _map_filename("correlation", "combined", fmt)))

        # Save NetCDF files
        if self.save_maps_netcdf:
            label_ref = self.data_ref_label or "Emulator"
            def _nc_filename(map_type, label):
                safe_label = label.replace(" ", "_")
                return f"amo_{map_type}_{var_suffix}_{season_suffix}_{safe_label}.nc"

            if reg_model is not None:
                reg_model.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("regression", self.data_label)))
            if cor_model is not None:
                cor_model.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("correlation", self.data_label)))
            if reg_ref is not None:
                reg_ref.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("regression", label_ref)))
            if cor_ref is not None:
                cor_ref.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("correlation", label_ref)))

    def plot_index(self, title="AMO Index", output_filename="amo_index.pdf"):
        """Generate and save the AMO index time series plot."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_filename)
        
        plt.figure(figsize=(10, 4))
        series = []
        if self.index is not None:
            series.append((self.index_filtered, self.data_label))
        if self.index_ref is not None:
            label = self.data_ref_label or "Emulator"
            series.append((self.index_filtered_ref, label))

        if not series:
            raise ValueError("No index to plot")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, (index, label) in enumerate(series):
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            index.plot(label=label, color=color)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Index Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"AMO index plot saved to: {output_path}")