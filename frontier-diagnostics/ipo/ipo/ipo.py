import os
import xarray as xr
import numpy as np
from scipy.signal import cheby1, filtfilt
import matplotlib.pyplot as plt

from aqua.core.graphics.multiple_maps import plot_maps
from aqua.core.graphics.single_map import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.reader import Reader
from aqua.core.util import load_yaml, to_list
from aqua.core.fldstat.area_selection import AreaSelection
from aqua.core.util.sci_util import select_season

class IPO:
    """
    Tripole Index diagnostic for the Interdecadal Pacific Oscillation.
    
    Henley, B. J., Gergis, J., Karoly, D. J., Power, S., Kennedy, J., & Folland, C. K. (2015).
    A tripole index for the interdecadal Pacific oscillation. Climate dynamics, 45(11), 3077-3090.
    """

    def __init__(self, config, loglevel: str = "WARNING"):
        """
        Initialize the IPO diagnostic.

        Args:
            config (str or dict): Configuration file or dictionary.
            loglevel (str): Logging level.
        """
        # Configure the logger
        self.logger = log_configure(log_name="IPO", log_level=loglevel)
        self.loglevel = loglevel
        self.config = load_yaml(config) if isinstance(config, str) else config

        # Validate the configuration
        required = ["data", "regions", "dates"]
        missing = [k for k in required if k not in self.config]
        if missing:
            self.logger.error(f"Missing configuration sections: {missing}")
            raise ValueError(f"Missing configuration sections: {missing}")

        # Get the dates
        dates = self.config["dates"]
        self.startdate = dates.get("startdate")
        self.enddate = dates.get("enddate")
        if not self.startdate or not self.enddate:
            self.logger.error("Configuration must provide dates.startdate and dates.enddate")
            raise ValueError("Configuration must provide dates.startdate and dates.enddate")

        # Get the data configuration
        self.data_cfg = self.config["data"]
        self.data_label = self.data_cfg.get("label") or self.data_cfg.get("model") or "Emulator"

        # Get the field to use for the IPO computation
        self.field = self.config.get("field", "tos")
        self.window = self.config.get("window", 13) # Window (years) for the low-pass filter
        
        # Chebyshev filter configuration
        self.filter_cfg = self.config.get("filter", {})
        self.filter_type = self.filter_cfg.get("type", "chebyshev") 
        self.cutoff_freq = self.filter_cfg.get("cutoff_freq", 1.0 / (self.window * 12)) 
        self.filter_order = self.filter_cfg.get("order", 4)
        self.ripple_db = self.filter_cfg.get("ripple_db", 1.0)
        
        # Get the reference data configuration
        self.data_ref_cfg = self.config.get("data_ref")
        self.data_ref_label = None
        if self.data_ref_cfg:
            self.data_ref_label = self.data_ref_cfg.get("label") or self.data_ref_cfg.get("model") or "Reference"

        # Get the plotting configuration
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
        self.output_dir = self.config.get("output_dir", "./ipo_output")
        self.maps_output_dir = os.path.join(self.output_dir, self.maps_output_subdir)
        self.sst = None # We assume that the variable used to compute the index will be SST (just for naming consistency)
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
        """Load field data (SST) and aggregate to monthly means."""
        field = self.field
        self.logger.info("Retrieving field '%s' for %s", field, self.data_label)
        self.sst = self.reader.retrieve(var=field)[field]
        self.sst = self.reader.timmean(self.sst, freq="MS")
        self.logger.info("Retrieved %s monthly points for %s", self.sst.sizes["time"], self.data_label)

        if self.reader_ref:
            label = self.data_ref_label
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
        """Compute the IPO index."""

        # Compute the IPO index
        self.index = self._compute_tpi(self.sst, self.reader, name="tpi")
        self.index_filtered = self._apply_filter(self.index)

        # Compute the IPO index for the reference data
        if self.sst_ref is not None and self.reader_ref is not None:
            self.index_ref = self._compute_tpi(self.sst_ref, self.reader_ref, name="tpi_ref")
            self.index_filtered_ref = self._apply_filter(self.index_ref)
        else:
            self.index_ref = None
            self.index_filtered_ref = None

        # Generate the plot
        self.plot_index()
        
        # Compute regression and correlation maps
        if self.compute_maps:
            self._compute_maps()

    def _compute_tpi(self, sst_data, reader, name="tpi"):
        """Compute the IPO index from SST data using the Tripole Index method from Henley et al. (2015)."""
        # Get the regions to compute the index
        regions = self.config["regions"]
        # Compute the monthly SST anomalies
        anomalies = sst_data.groupby("time.month") - sst_data.groupby("time.month").mean(dim="time")
        # Compute the index for the southwest region
        ssta_1 = self._select_region_mean(anomalies, regions["southwest"], reader)
        ssta_2 = self._select_region_mean(anomalies, regions["equatorial"], reader)
        ssta_3 = self._select_region_mean(anomalies, regions["northwest"], reader)
        # Compute the IPO index
        return (ssta_2 - 0.5 * (ssta_1 + ssta_3)).rename(name)

    def _apply_filter(self, index):
        """Apply a smoothing filter to the index."""
        if not self.window:
            return None

        if self.filter_type == "rolling":
            # Rolling mean approach
            window = self.window * 12
            filtered = index.rolling(time=window, center=True, min_periods=max(1, window // 2)).mean()
            if filtered is None:
                return None
        elif self.filter_type == "chebyshev":
            # Chebyshev low-pass filter
            filtered = self._apply_chebyshev_filter(index)
            if filtered is None:
                return None
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        filtered_name = f"{index.name}_filtered" if index.name else "tpi_filtered"
        return filtered.rename(filtered_name)

    def _apply_chebyshev_filter(self, index):
        """Apply Chebyshev Type I low-pass filter to the index time series."""
        try:
            # Get the time series data
            data = index.values
            
            # Design Chebyshev Type I low-pass filter
            # Normalize cutoff frequency (Nyquist frequency = 0.5 for monthly data)
            nyquist = 0.5  # cycles per month
            normalized_cutoff = self.cutoff_freq / nyquist
            
            # Ensure cutoff frequency is valid
            if normalized_cutoff >= 1.0:
                self.logger.warning(f"Cutoff frequency {self.cutoff_freq} too high, using 0.9 * Nyquist")
                normalized_cutoff = 0.9
            
            # Design filter
            b, a = cheby1(self.filter_order, self.ripple_db, normalized_cutoff, btype='low')
            
            # Apply forward-backward filtering to avoid phase shift
            filtered_data = filtfilt(b, a, data)
            
            # Create new xarray DataArray with same coordinates as original
            filtered_index = xr.DataArray(filtered_data,
                                          dims=index.dims,
                                          coords=index.coords,
                                          attrs=index.attrs)
            
            self.logger.info(f"Applied Chebyshev filter: order={self.filter_order}, "
                             f"cutoff={self.cutoff_freq:.4f} cycles/month, "
                             f"ripple={self.ripple_db} dB")
            
            return filtered_index

        except Exception as e:
            self.logger.error(f"Error applying Chebyshev filter: {e}")
            return None

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
        """Select the season if needed."""
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

        for var in self.map_variables:
            for season in self.map_seasons:
                # Compute the regression and correlation maps
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
        """Plot regression and correlation maps for IPO diagnostic."""
        if not self.save_combined_maps and not self.save_maps_netcdf:
            return

        # Create the output directory
        os.makedirs(self.maps_output_dir, exist_ok=True)
        season_suffix = season.lower()
        var_suffix = var

        def _map_title(map_type, label):
            title = f"IPO {map_type} ({var}) - {label}"
            if season != "annual":
                title += f" ({season})"
            return title

        def _map_filename(map_type, label=None, fmt="pdf"):
            base = f"ipo_{map_type}_{var_suffix}_{season_suffix}"
            if label:
                safe_label = label.replace(" ", "_")
                base += f"_{safe_label}"
            return f"{base}.{fmt}"

        # Combined plots (model vs reference)
        if self.save_combined_maps:
            label_ref = self.data_ref_label or "Reference"
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
            label_ref = self.data_ref_label or "Reference"
            def _nc_filename(map_type, label):
                safe_label = label.replace(" ", "_")
                return f"ipo_{map_type}_{var_suffix}_{season_suffix}_{safe_label}.nc"

            if reg_model is not None:
                reg_model.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("regression", self.data_label)))
            if cor_model is not None:
                cor_model.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("correlation", self.data_label)))
            if reg_ref is not None:
                reg_ref.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("regression", label_ref)))
            if cor_ref is not None:
                cor_ref.to_netcdf(os.path.join(self.maps_output_dir, _nc_filename("correlation", label_ref)))

    def plot_index(self, title="IPO Tripole Index", output_filename="ipo_index.pdf"):
        """Generate and save the IPO index time series plot."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Generate the plot
        plt.figure(figsize=(10, 4))
        series = []
        if self.index is not None:
            series.append((self.index_filtered, self.data_label))
        if self.index_ref is not None:
            label = self.data_ref_label
            series.append((self.index_filtered_ref, label))

        if not series:
            self.logger.error("No index to plot")
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
        
        self.logger.info(f"IPO index plot saved to: {output_path}")