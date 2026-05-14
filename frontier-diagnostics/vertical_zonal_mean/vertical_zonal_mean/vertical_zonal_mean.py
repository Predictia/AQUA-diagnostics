import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from aqua.core.logger import log_configure
from aqua.core.util import load_yaml
from aqua.core.reader import Reader
from aqua.core.timstat import TimStat


class VerticalZonalMean:
    """Vertical Zonal Mean diagnostic that computes latitude-pressure cross-sections.

    This diagnostic computes the zonal mean (average over longitude) of variables
    with pressure levels, comparing emulator results to reference model data.
    Produces three-panel figures showing reference, emulator, and their difference.
    """

    def __init__(self, config, loglevel: str = 'WARNING'):
        """Initialize the VerticalZonalMean diagnostic class.

        Args:
            config: Configuration for the diagnostic (dict or path to YAML file).
            loglevel (str): Logging level. Defaults to 'WARNING'.

        Raises:
            ValueError: If required configuration sections or keys are missing.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'VerticalZonalMean')

        # Load configuration
        if isinstance(config, str):
            self.logger.debug("Reading configuration file %s", config)
            self.config = load_yaml(config)
        else:
            self.logger.debug("Configuration is a dictionary")
            self.config = config

        # Get start and end dates
        self.startdate = self.config.get('dates', {}).get('startdate')
        self.enddate = self.config.get('dates', {}).get('enddate')

        # Check if start and end dates are provided
        if self.startdate is None or self.enddate is None:
            self.logger.error("Both startdate and enddate must be provided")
            raise ValueError("Both startdate and enddate must be provided")

        # Output directories
        self.outputdir_fig = self.config.get('outputdir_fig', './figs_vertical_zonal_mean')
        self.outputdir_data = self.config.get('outputdir_data', './data_vertical_zonal_mean')
        self.figure_format = self._normalise_extension(self.config.get('figure_format', 'pdf'))
        os.makedirs(self.outputdir_fig, exist_ok=True)
        os.makedirs(self.outputdir_data, exist_ok=True)

        # Load plot settings
        self.plot_settings = self._load_plot_settings()

        # Initialize TimStat for temporal aggregation
        self.timstat = TimStat(loglevel=self.loglevel)

        # Parse aggregation configuration
        self._parse_aggregation_config()

        # Parse latitude binning configuration
        self.latitude_bins = self.config.get('latitude_bins', None)
        if self.latitude_bins is not None:
            self.latitude_bins = int(self.latitude_bins)
            self.logger.debug(f"Latitude binning enabled with {self.latitude_bins} bins")

        # Initialize the Reader class
        self._reader()

    def _reader(self):
        """Initializes the Reader class for both reference and main data."""
        # Get the data configuration
        config_data_ref = self.config.get('data_ref', {})
        config_data = self.config.get('data', {})

        # Check if necessary keys exist within the data configuration
        if not all(k in config_data_ref for k in ['catalog', 'model', 'exp', 'source', 'regrid', 'fix']):
            self.logger.warning("Reference data configuration ('data_ref') might be incomplete.")
        if not all(k in config_data for k in ['catalog', 'model', 'exp', 'source', 'regrid', 'fix']):
            self.logger.warning("Main data configuration ('data') might be incomplete.")

        # Initialize the Reader class for reference data
        self.reader_data_ref = Reader(catalog=config_data_ref.get('catalog'),
                                      model=config_data_ref.get('model'),
                                      exp=config_data_ref.get('exp'),
                                      source=config_data_ref.get('source'),
                                      regrid=config_data_ref.get('regrid'),
                                      fix=config_data_ref.get('fix'),
                                      areas=config_data_ref.get('areas', False),
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel)

        self.reader_data = Reader(catalog=config_data.get('catalog'),
                                  model=config_data.get('model'),
                                  exp=config_data.get('exp'),
                                  source=config_data.get('source'),
                                  regrid=config_data.get('regrid'),
                                  fix=config_data.get('fix'),
                                  areas=config_data.get('areas', False),
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel)

        self.logger.debug('Reader classes initialized for data_ref and data')

    def _load_plot_settings(self):
        """Load plot settings from configuration."""
        plot_cfg = self.config.get('plot', {})

        # Default settings
        settings = {
            'data': {
                'cmap': 'viridis',
                'vmin': None,
                'vmax': None,
            },
            'diff': {
                'cmap': 'RdBu_r',
                'vmin': None,
                'vmax': None,
            },
            'significance': {
                'show': False,
                'alpha': 0.05,
                'hatch_pattern': '....',
                'invert_mask': False,
            },
            'pressure_unit': 'hPa',
            'figure_size': [18, 6],
            'nlevels': 18,
        }

        # Update with user-provided settings for data plots
        data_cfg = plot_cfg.get('data', {})
        if 'cmap' in data_cfg:
            settings['data']['cmap'] = str(data_cfg['cmap'])
        if 'vmin' in data_cfg and data_cfg['vmin'] is not None:
            settings['data']['vmin'] = float(data_cfg['vmin'])
        if 'vmax' in data_cfg and data_cfg['vmax'] is not None:
            settings['data']['vmax'] = float(data_cfg['vmax'])

        # Update with user-provided settings for diff plots
        diff_cfg = plot_cfg.get('diff', {})
        if 'cmap' in diff_cfg:
            settings['diff']['cmap'] = str(diff_cfg['cmap'])
        if 'vmin' in diff_cfg and diff_cfg['vmin'] is not None:
            settings['diff']['vmin'] = float(diff_cfg['vmin'])
        if 'vmax' in diff_cfg and diff_cfg['vmax'] is not None:
            settings['diff']['vmax'] = float(diff_cfg['vmax'])

        # Update significance settings
        significance_cfg = plot_cfg.get('significance', {})
        if 'show' in significance_cfg:
            settings['significance']['show'] = bool(significance_cfg['show'])
        if 'alpha' in significance_cfg and significance_cfg['alpha'] is not None:
            settings['significance']['alpha'] = float(significance_cfg['alpha'])
        if 'hatch_pattern' in significance_cfg:
            settings['significance']['hatch_pattern'] = str(significance_cfg['hatch_pattern'])
        if 'invert_mask' in significance_cfg:
            settings['significance']['invert_mask'] = bool(significance_cfg['invert_mask'])

        # Other plot settings
        if 'pressure_unit' in plot_cfg:
            settings['pressure_unit'] = str(plot_cfg['pressure_unit'])
        if 'figure_size' in plot_cfg:
            settings['figure_size'] = plot_cfg['figure_size']
        if 'nlevels' in plot_cfg:
            settings['nlevels'] = int(plot_cfg['nlevels'])

        self.logger.debug(f"Plot settings loaded: {settings}")
        return settings

    def _parse_aggregation_config(self):
        """Parse and validate aggregation configuration from config file.

        Sets self.aggregation_stat and self.percentile attributes.
        Defaults to 'mean' for backward compatibility.

        Raises:
            ValueError: If percentile is out of range [0, 100] or if invalid stat is provided.
        """
        agg_cfg = self.config.get('aggregation', {})

        # Get statistic type (default to 'mean' for backward compatibility)
        self.aggregation_stat = agg_cfg.get('stat', 'mean')

        # Validate statistic
        valid_stats = ['mean', 'std', 'max', 'min', 'percentile']
        if self.aggregation_stat not in valid_stats:
            self.logger.error(f"Invalid aggregation stat '{self.aggregation_stat}'. Must be one of {valid_stats}")
            raise ValueError(f"Invalid aggregation stat '{self.aggregation_stat}'. Must be one of {valid_stats}")

        # Get percentile value (only used if stat='percentile')
        self.percentile = agg_cfg.get('percentile', 99.9)

        # Validate percentile range
        if not (0 <= self.percentile <= 100):
            self.logger.error(f"Percentile must be between 0 and 100, got {self.percentile}")
            raise ValueError(f"Percentile must be between 0 and 100, got {self.percentile}")

        self.logger.debug(f"Aggregation config: stat={self.aggregation_stat}, percentile={self.percentile}")

    def retrieve(self):
        """
        Retrieves data for all variables specified in the configuration
        from both the reference and main data sources using the initialized Reader classes.
        Stores retrieved data in self.retrieved_data_ref and self.retrieved_data dictionaries.

        This method retrieves the full 3D data (with all pressure levels) for each variable.
        """
        self.retrieved_data_ref = {}
        self.retrieved_data = {}
        variables_to_process = self.config.get('variables', [])

        if not variables_to_process:
            self.logger.warning("No variables section found in the configuration. Nothing to retrieve.")
            return

        self.logger.info("Starting data retrieval...")

        # Process each variable
        for var_info in variables_to_process:
            var_name = var_info.get('name')
            if not var_name:
                self.logger.warning("Skipping variable entry with no name: %s", var_info)
                continue

            self.logger.info(f"Retrieving variable: {var_name}")

            # Retrieve reference data (all levels)
            try:
                self.logger.debug(f"Retrieving reference data for {var_name}")
                data_ref = self.reader_data_ref.retrieve(var=var_name)
                if isinstance(data_ref, xr.Dataset):
                    data_ref = data_ref[var_name]

                # Apply regridding if configured
                if self.reader_data_ref.tgt_grid_name:
                    self.logger.debug(f"Applying regridding to reference data for {var_name}")
                    data_ref = self.reader_data_ref.regrid(data_ref)

                self.retrieved_data_ref[var_name] = data_ref
                self.logger.debug(f"Successfully retrieved reference data for: {var_name}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred retrieving reference {var_name}: {e}", exc_info=True)
                continue

            # Retrieve Main Data (all levels)
            try:
                self.logger.debug(f"Retrieving main data for {var_name}")
                data = self.reader_data.retrieve(var=var_name)
                if isinstance(data, xr.Dataset):
                    data = data[var_name]

                # Apply regridding if configured
                if self.reader_data.tgt_grid_name:
                    self.logger.debug(f"Applying regridding to main data for {var_name}")
                    data = self.reader_data.regrid(data)

                self.retrieved_data[var_name] = data
                self.logger.debug(f"Successfully retrieved main data for: {var_name}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred retrieving main {var_name}: {e}", exc_info=True)
                # Clean up ref data if main data failed
                if var_name in self.retrieved_data_ref:
                    del self.retrieved_data_ref[var_name]
                continue

        self.logger.info("Data retrieval finished.")
        if not self.retrieved_data_ref and not self.retrieved_data:
            self.logger.warning("No data was successfully retrieved for any variable.")
        else:
            # Log the keys
            self.logger.info(f"Retrieved reference data for variables: {list(self.retrieved_data_ref.keys())}")
            self.logger.info(f"Retrieved main data for variables: {list(self.retrieved_data.keys())}")

    @staticmethod
    def check_and_convert_coords(data, data_ref):
        """
        Checks if latitude and longitude coordinates are of the same type between datasets.
        Converts them to float32 if they differ.

        Args:
            data (xr.DataArray): First data array to check
            data_ref (xr.DataArray): Reference data array to check

        Returns:
            tuple: Processed data arrays with matching coordinate types
        """
        # Get coordinate names
        try:
            lat_name = [dim for dim in data.dims if 'lat' in dim.lower()][0]
            lon_name = [dim for dim in data.dims if 'lon' in dim.lower()][0]
        except IndexError:
            print(f"Warning: Could not determine lat/lon dimension names.")
            return data, data_ref

        # Check if types match
        if data[lat_name].dtype != data_ref[lat_name].dtype or data[lon_name].dtype != data_ref[lon_name].dtype:
            # Convert both coordinates to float32
            data = data.assign_coords({lat_name: data[lat_name].astype('float32'),
                                       lon_name: data[lon_name].astype('float32')})
            data_ref = data_ref.assign_coords({lat_name: data_ref[lat_name].astype('float32'),
                                               lon_name: data_ref[lon_name].astype('float32')})
        return data, data_ref

    @staticmethod
    def _find_level_coord(data):
        """Find the name of the vertical level coordinate.

        Args:
            data (xr.DataArray): Data array to inspect

        Returns:
            str: Name of the level coordinate, or None if not found
        """
        # Common names for pressure level coordinates
        possible_names = ['plev', 'level', 'lev', 'pressure', 'p']
        for name in possible_names:
            if name in data.dims or name in data.coords:
                return name
        # Try to find by checking for pressure-like values
        for coord in data.coords:
            if coord.lower() in possible_names:
                return coord
        return None

    def _find_common_levels(self, data_ref, data):
        """Find common pressure levels between reference and emulator data.

        Args:
            data_ref (xr.DataArray): Reference data array
            data (xr.DataArray): Emulator data array

        Returns:
            list: Sorted list of common pressure levels (descending order for proper plotting)
        """
        lev_name_ref = self._find_level_coord(data_ref)
        lev_name_data = self._find_level_coord(data)

        if lev_name_ref is None or lev_name_data is None:
            self.logger.warning("Could not find level coordinate in one or both datasets")
            return None, None, None

        levels_ref = set(data_ref[lev_name_ref].values)
        levels_data = set(data[lev_name_data].values)

        common_levels = sorted(levels_ref & levels_data, reverse=True)

        self.logger.debug(f"Reference levels: {len(levels_ref)}, Data levels: {len(levels_data)}, Common: {len(common_levels)}")

        return common_levels, lev_name_ref, lev_name_data

    @staticmethod
    def _percentile_func(data, q):
        """
        Compute percentile over time dimension using xarray quantile.

        Args:
            data (xr.DataArray): Data array with time dimension
            q (float): Percentile to compute (0-100 scale)

        Returns:
            xr.DataArray: Percentile values with time dimension removed
        """
        # Convert from 0-100 scale to 0-1 scale for xarray quantile
        result = data.quantile(q / 100.0, dim='time', skipna=True)
        # Drop the quantile coordinate that xarray adds
        if 'quantile' in result.coords:
            result = result.drop_vars('quantile')
        return result

    def _compute_temporal_aggregation(self, data):
        """Compute temporal aggregation of the data.

        Args:
            data (xr.DataArray): Data array with time dimension

        Returns:
            xr.DataArray: Temporally aggregated data
        """
        if self.aggregation_stat == 'percentile':
            self.logger.debug(f"Computing {self.percentile}th percentile over time dimension")
            temporal_stat = self.timstat.timstat(
                data,
                stat=self._percentile_func,
                freq=None,
                func_kwargs={'q': self.percentile}
            )
        else:
            self.logger.debug(f"Computing temporal {self.aggregation_stat}")
            temporal_stat = self.timstat.timstat(
                data,
                stat=self.aggregation_stat,
                freq=None
            )
        return temporal_stat

    def _compute_zonal_mean(self, data):
        """Compute zonal mean (average over longitude) of the data.

        Args:
            data (xr.DataArray): Data array with lon dimension

        Returns:
            xr.DataArray: Zonal mean data
        """
        # Find longitude dimension
        lon_name = [dim for dim in data.dims if 'lon' in dim.lower()][0]
        zonal_mean = data.mean(dim=lon_name)
        return zonal_mean

    def _compute_yearly_temporal_means(self, data):
        """Compute yearly means over time for significance testing."""
        return self.timstat.timstat(data, stat='mean', freq='YS')

    @staticmethod
    def _ttest_at_grid_point(model_vals, ref_vals, min_samples=3):
        """Perform a two-sample Welch t-test at one grid point."""
        model_clean = model_vals[np.isfinite(model_vals)]
        ref_clean = ref_vals[np.isfinite(ref_vals)]

        if len(model_clean) < min_samples or len(ref_clean) < min_samples:
            return np.nan

        _, p_value = stats.ttest_ind(model_clean, ref_clean, equal_var=False)
        return p_value

    def _compute_significance_ttest(self, data, data_ref, alpha=0.05, min_samples=3):
        """Compute statistical significance mask using Welch t-test on yearly means.

        Args:
            data (xr.DataArray): Emulator data with time dimension.
            data_ref (xr.DataArray): Reference data with time dimension.
            alpha (float): Significance level for p-values.
            min_samples (int): Minimum yearly samples required for each dataset.

        Returns:
            xr.DataArray: Boolean significance mask on spatial dimensions.
        """
        if 'time' not in data.dims or 'time' not in data_ref.dims:
            raise ValueError("Both data and data_ref must include a 'time' dimension for significance testing.")

        data_yearly = self._compute_yearly_temporal_means(data)
        data_ref_yearly = self._compute_yearly_temporal_means(data_ref)

        n_samples = int(data_yearly.sizes.get('time', 0))
        n_samples_ref = int(data_ref_yearly.sizes.get('time', 0))
        self.logger.info("Significance test yearly samples - model: %s, reference: %s", n_samples, n_samples_ref)

        if n_samples < min_samples or n_samples_ref < min_samples:
            self.logger.warning(
                "Insufficient yearly samples for significance test (model=%s, reference=%s, min=%s).",
                n_samples, n_samples_ref, min_samples
            )
            base = data.isel(time=0, drop=True)
            is_significant = xr.zeros_like(base, dtype=bool)
            is_significant.attrs.update({
                'long_name': 'Statistical significance of zonal-mean difference',
                'description': f'Two-sample Welch t-test with alpha={alpha}',
                'alpha': alpha,
                'n_samples_model': n_samples,
                'n_samples_reference': n_samples_ref,
                'n_significant_points': 0,
                'percent_significant': 0.0,
            })
            return is_significant

        data_yearly = data_yearly.chunk({'time': -1})
        data_ref_yearly = data_ref_yearly.chunk({'time': -1})

        time_dim = 'time'
        time_dim_ref = 'time_ref'
        data_yearly = data_yearly.rename({time_dim: time_dim})
        data_ref_yearly = data_ref_yearly.rename({time_dim: time_dim_ref})

        p_values = xr.apply_ufunc(
            self._ttest_at_grid_point,
            data_yearly,
            data_ref_yearly,
            input_core_dims=[[time_dim], [time_dim_ref]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            kwargs={'min_samples': min_samples},
        )

        is_significant = (p_values < alpha).fillna(False)
        is_significant.load()

        n_significant = int(is_significant.sum().values)
        n_total = int(np.prod(is_significant.shape))
        pct_significant = 100.0 * n_significant / n_total if n_total > 0 else 0.0
        self.logger.info(
            "Significance test complete: %s/%s points (%.1f%%) significant at alpha=%s",
            n_significant, n_total, pct_significant, alpha
        )

        is_significant.attrs.update({
            'long_name': 'Statistical significance of zonal-mean difference',
            'description': f'Two-sample Welch t-test with alpha={alpha}',
            'alpha': alpha,
            'n_samples_model': n_samples,
            'n_samples_reference': n_samples_ref,
            'n_significant_points': n_significant,
            'percent_significant': pct_significant,
        })
        return is_significant

    def _bin_latitude(self, data, n_bins):
        """Bin data to coarser latitude resolution.

        Args:
            data (xr.DataArray): Data array with lat dimension
            n_bins (int): Number of latitude bins

        Returns:
            xr.DataArray: Data binned to coarser latitude resolution
        """
        # Find latitude dimension
        lat_name = [dim for dim in data.dims if 'lat' in dim.lower()][0]

        # Create bins from -90 to 90
        lat_bins = np.linspace(-90, 90, n_bins + 1)
        lat_labels = (lat_bins[:-1] + lat_bins[1:]) / 2

        # Group by bins and compute mean
        binned = data.groupby_bins(lat_name, lat_bins, labels=lat_labels).mean()

        # Rename the binned coordinate to the original lat name
        binned_coord_name = f'{lat_name}_bins'
        if binned_coord_name in binned.dims:
            binned = binned.rename({binned_coord_name: lat_name})

        return binned

    @staticmethod
    def _sanitize_filename_part(part):
        """Sanitizes a string part for use in a filename."""
        if not isinstance(part, str):
            part = str(part)

        # Replace potentially problematic characters with underscores
        invalid_chars = [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']
        for char in invalid_chars:
            part = part.replace(char, '_')
        return part

    def _generate_filename(self, var_name, suffix):
        """
        Generates a unique filename based on config and context.

        Args:
            var_name (str): The variable name.
            suffix (str): Suffix describing the calculation.

        Returns:
            str: The full, unique path for the output file.
        """
        # Get config details with defaults and sanitize them
        data_cfg = self.config.get('data', {})
        ref_cfg = self.config.get('data_ref', {})
        dates_cfg = self.config.get('dates', {})

        model = self._sanitize_filename_part(data_cfg.get('model', 'model'))
        exp = self._sanitize_filename_part(data_cfg.get('exp', 'exp'))
        source = self._sanitize_filename_part(data_cfg.get('source', 'src'))
        model_ref = self._sanitize_filename_part(ref_cfg.get('model', 'refmodel'))
        exp_ref = self._sanitize_filename_part(ref_cfg.get('exp', 'refexp'))

        startdate = self._sanitize_filename_part(dates_cfg.get('startdate', 'nodate'))
        enddate = self._sanitize_filename_part(dates_cfg.get('enddate', 'nodate'))

        # Sanitize variable name for filename
        var_name_safe = self._sanitize_filename_part(var_name)

        # Generate suffix with aggregation statistic
        if self.aggregation_stat == 'percentile':
            stat_suffix = f'percentile_{str(self.percentile).replace(".", "_")}'
        else:
            stat_suffix = self.aggregation_stat

        # Construct the base filename from sanitized parts
        base_name_parts = [model, exp, source,
                          'vs', model_ref, exp_ref,
                          var_name_safe,
                          startdate, enddate,
                          f'{suffix}_{stat_suffix}']

        # Filter out any empty strings
        base_filename = '_'.join(filter(None, base_name_parts))

        # Determine extension
        extension = f'.{self.figure_format}'

        full_path = os.path.join(self.outputdir_fig, f"{base_filename}{extension}")
        self.logger.debug(f"Generated path for figure ({var_name}, {suffix}): {full_path}")
        return full_path

    def _save_figure(self, fig, var_name):
        """
        Saves the generated figure with a unique name based on config.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            var_name (str): The variable name.
        """
        filename = self._generate_filename(var_name, 'vertical_zonal_mean')

        try:
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

    def _save_data(self, zm_ref, zm_data, zm_diff, var_name, lev_name, significance_mask=None):
        """
        Saves the computed zonal mean data to NetCDF.

        Args:
            zm_ref (xr.DataArray): Reference zonal mean profile.
            zm_data (xr.DataArray): Emulator zonal mean profile.
            zm_diff (xr.DataArray): Difference (emulator - reference).
            var_name (str): The variable name.
            lev_name (str): Name of the level coordinate.
        """
        data_cfg = self.config.get('data', {})
        ref_cfg = self.config.get('data_ref', {})
        dates_cfg = self.config.get('dates', {})

        model = self._sanitize_filename_part(data_cfg.get('model', 'model'))
        exp = self._sanitize_filename_part(data_cfg.get('exp', 'exp'))
        source = self._sanitize_filename_part(data_cfg.get('source', 'src'))
        model_ref = self._sanitize_filename_part(ref_cfg.get('model', 'refmodel'))
        exp_ref = self._sanitize_filename_part(ref_cfg.get('exp', 'refexp'))

        startdate = self._sanitize_filename_part(dates_cfg.get('startdate', 'nodate'))
        enddate = self._sanitize_filename_part(dates_cfg.get('enddate', 'nodate'))

        var_name_safe = self._sanitize_filename_part(var_name)

        # Generate suffix with aggregation statistic
        if self.aggregation_stat == 'percentile':
            stat_suffix = f'percentile_{str(self.percentile).replace(".", "_")}'
        else:
            stat_suffix = self.aggregation_stat

        base_name_parts = [model, exp, source,
                          'vs', model_ref, exp_ref,
                          var_name_safe,
                          startdate, enddate,
                          f'vertical_zonal_mean_{stat_suffix}']

        base_filename = '_'.join(filter(None, base_name_parts))
        filepath = os.path.join(self.outputdir_data, f"{base_filename}.nc")

        try:
            # Create a dataset with all three fields
            ds = xr.Dataset({
                'zonal_mean_reference': zm_ref.rename('zonal_mean_reference'),
                'zonal_mean_emulator': zm_data.rename('zonal_mean_emulator'),
                'zonal_mean_difference': zm_diff.rename('zonal_mean_difference'),
            })
            if significance_mask is not None:
                ds['significance_mask'] = significance_mask.rename('significance_mask')

            # Add metadata
            ds.attrs['variable'] = var_name
            ds.attrs['startdate'] = str(self.startdate)
            ds.attrs['enddate'] = str(self.enddate)
            ds.attrs['model'] = data_cfg.get('model', 'unknown')
            ds.attrs['model_ref'] = ref_cfg.get('model', 'unknown')
            ds.attrs['aggregation_stat'] = self.aggregation_stat
            if self.aggregation_stat == 'percentile':
                ds.attrs['percentile'] = self.percentile
            if self.latitude_bins is not None:
                ds.attrs['latitude_bins'] = self.latitude_bins
            if significance_mask is not None:
                ds.attrs['significance_alpha'] = significance_mask.attrs.get('alpha')
                ds.attrs['significance_n_samples_model'] = significance_mask.attrs.get('n_samples_model')
                ds.attrs['significance_n_samples_reference'] = significance_mask.attrs.get('n_samples_reference')
                ds.attrs['significance_percent'] = significance_mask.attrs.get('percent_significant')

            ds.to_netcdf(filepath)
            self.logger.info(f"Data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save data {filepath}: {e}", exc_info=True)

    @staticmethod
    def _normalise_extension(extension):
        """Normalise the extension."""
        ext = extension.lower().lstrip(".")
        if not ext:
            return "pdf"
        else:
            return ext

    def _add_significance_markers(self, ax, significance_mask, lev_name,
                                  hatch_pattern='....', invert_mask=False, **kwargs):
        """Add significance hatching on lat-pressure contour plots.

        Uses contourf with hatching so that the overlay interpolates between
        discrete pressure levels, matching the filled-contour background.

        Args:
            ax: Matplotlib axes to draw on.
            significance_mask (xr.DataArray): Boolean mask (True = significant).
            lev_name (str): Name of the vertical level coordinate.
            hatch_pattern (str): Matplotlib hatch pattern for significant regions.
            invert_mask (bool): If True, hatch non-significant regions instead.
            **kwargs: Accepted for backward compatibility (marker_stride, etc.).
        """
        try:
            lat_name = [dim for dim in significance_mask.dims if 'lat' in dim.lower()][0]
        except IndexError:
            raise ValueError("Unable to determine latitude coordinate from significance mask.")
        if lev_name not in significance_mask.dims:
            raise ValueError(f"Level coordinate '{lev_name}' not found in significance mask.")

        lat_vals = significance_mask[lat_name].values
        lev_vals = significance_mask[lev_name].values

        pressure_unit = self.plot_settings['pressure_unit']
        if pressure_unit == 'hPa' and np.nanmax(lev_vals) > 2000:
            lev_vals = lev_vals / 100.0

        mask_values = significance_mask.values.astype(float)
        if invert_mask:
            mask_values = 1.0 - mask_values

        ax.contourf(lat_vals, lev_vals, mask_values,
                    levels=[0.5, 1.5], hatches=[hatch_pattern],
                    colors='none', alpha=0)

    def _plot_vertical_zonal_mean(self, zm_ref, zm_data, zm_diff, var_name, units,
                                   label_ref, label_exp, lev_name, significance_mask=None):
        """
        Create a figure with three subplots showing reference, emulator, and difference.

        Args:
            zm_ref: Zonal mean profile of reference data (lat x plev)
            zm_data: Zonal mean profile of emulator data (lat x plev)
            zm_diff: Difference (emulator - reference)
            var_name: Variable name for labels
            units: Units of the variable
            label_ref: Label for reference data
            label_exp: Label for emulator data
            lev_name: Name of the level coordinate

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get latitude coordinate
        try:
            lat_name = [dim for dim in zm_ref.dims if 'lat' in dim.lower()][0]
        except IndexError:
            raise ValueError("Unable to determine latitude coordinate for plotting.")

        lat = zm_ref[lat_name].values
        lev = zm_ref[lev_name].values

        # Convert pressure units for display if needed
        pressure_unit = self.plot_settings['pressure_unit']
        if pressure_unit == 'hPa' and lev.max() > 2000:  # Assume Pa if max > 2000
            lev_display = lev / 100.0
            ylabel = 'Pressure [hPa]'
        else:
            lev_display = lev
            ylabel = f'Pressure [{pressure_unit}]'

        # Get plot settings
        data_settings = self.plot_settings['data']
        diff_settings = self.plot_settings['diff']
        nlevels = self.plot_settings['nlevels']

        # Determine color limits for data plots
        if data_settings['vmin'] is not None and data_settings['vmax'] is not None:
            vmin_data = data_settings['vmin']
            vmax_data = data_settings['vmax']
        else:
            valid_ref = zm_ref.values[np.isfinite(zm_ref.values)]
            valid_data = zm_data.values[np.isfinite(zm_data.values)]
            if len(valid_ref) > 0 and len(valid_data) > 0:
                all_valid = np.concatenate([valid_ref, valid_data])
                vmin_data = float(np.percentile(all_valid, 2))
                vmax_data = float(np.percentile(all_valid, 98))
            else:
                vmin_data = 0
                vmax_data = 1

        # Determine color limits for difference plot (symmetric)
        if diff_settings['vmin'] is not None and diff_settings['vmax'] is not None:
            vmin_diff = diff_settings['vmin']
            vmax_diff = diff_settings['vmax']
        else:
            valid_diff = zm_diff.values[np.isfinite(zm_diff.values)]
            if len(valid_diff) > 0:
                abs_max = float(np.percentile(np.abs(valid_diff), 98))
                vmin_diff = -abs_max
                vmax_diff = abs_max
            else:
                vmin_diff = -0.1
                vmax_diff = 0.1

        # Create contour levels
        levels_data = np.linspace(vmin_data, vmax_data, nlevels)
        levels_diff = np.linspace(vmin_diff, vmax_diff, nlevels)

        # Create figure with 3 subplots
        figsize = tuple(self.plot_settings['figure_size'])
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Reference
        cf1 = axes[0].contourf(lat, lev_display, zm_ref.values,
                               levels=levels_data, cmap=data_settings['cmap'], extend='both')
        axes[0].set_title(f'Reference ({label_ref})')
        axes[0].set_xlabel('Latitude')
        axes[0].set_ylabel(ylabel)
        axes[0].invert_yaxis()
        cbar1 = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.12)
        cbar_label = f'{var_name} [{units}]' if units else var_name
        cbar1.set_label(cbar_label)

        # Plot 2: Emulator
        cf2 = axes[1].contourf(lat, lev_display, zm_data.values,
                               levels=levels_data, cmap=data_settings['cmap'], extend='both')
        axes[1].set_title(f'Emulator ({label_exp})')
        axes[1].set_xlabel('Latitude')
        axes[1].set_ylabel(ylabel)
        axes[1].invert_yaxis()
        cbar2 = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.12)
        cbar2.set_label(cbar_label)

        # Plot 3: Difference (Emulator - Reference)
        cf3 = axes[2].contourf(lat, lev_display, zm_diff.values,
                               levels=levels_diff, cmap=diff_settings['cmap'], extend='both')
        axes[2].set_title('Difference (Emulator - Reference)')
        axes[2].set_xlabel('Latitude')
        axes[2].set_ylabel(ylabel)
        axes[2].invert_yaxis()
        cbar3 = plt.colorbar(cf3, ax=axes[2], orientation='horizontal', pad=0.12)
        cbar3.set_label(cbar_label)

        significance_cfg = self.plot_settings['significance']
        if significance_cfg['show'] and significance_mask is not None:
            self._add_significance_markers(
                axes[2],
                significance_mask=significance_mask,
                lev_name=lev_name,
                hatch_pattern=significance_cfg['hatch_pattern'],
                invert_mask=significance_cfg['invert_mask'],
            )
            pct_sig = significance_mask.attrs.get('percent_significant', 0.0)
            n_samples = significance_mask.attrs.get('n_samples_model', 'unknown')
            alpha = significance_cfg['alpha']
            hatch_label = "non-significant" if significance_cfg['invert_mask'] else "significant"
            axes[2].text(
                0.99, 0.01,
                f"Hatching: {hatch_label} (p < {alpha})\n"
                f"Welch t-test, N = {n_samples}\nSignificant: {pct_sig:.1f}%",
                transform=axes[2].transAxes,
                ha='right', va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            )

        # Set overall title
        if self.aggregation_stat == 'percentile':
            stat_label = f"{self.percentile}th percentile"
        else:
            stat_label = self.aggregation_stat.title()
        fig.suptitle(f"Vertical Zonal Mean ({stat_label}): {var_name}\n"
                     f"{self.startdate} to {self.enddate}",
                     fontsize=14, y=1.02)

        fig.tight_layout()
        return fig

    def compute_vertical_profile(self, save_fig: bool = False, save_data: bool = False):
        """
        Calculates and plots the vertical zonal mean profiles for all variables retrieved.

        This method processes all common variables between the main and reference datasets,
        calculating the zonal mean (latitude-pressure profile) for each. The temporal
        aggregation statistic is applied before computing the zonal mean.

        Args:
            save_fig (bool, optional): If True, saves the generated figures to the configured
                output directory. Defaults to False.
            save_data (bool, optional): If True, saves the computed zonal mean data
                to NetCDF files. Defaults to False.

        Returns:
            dict: Dictionary of figure objects keyed by variable name.

        Note:
            - The retrieve() method must be called before this method.
            - Only variables present in both datasets will be processed.
        """
        self.logger.info('Calculating and plotting vertical zonal mean profiles for retrieved variables.')

        # Ensure data has been retrieved
        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
            self.logger.error("Data has not been retrieved yet. Call the retrieve() method first.")
            return {}

        # Find common keys retrieved in both datasets
        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys())
        if not common_keys:
            self.logger.warning("No common data keys found between reference and main retrieved data.")
            return {}

        results = {}

        # Get metadata for labels
        config_data = self.config.get('data', {})
        config_data_ref = self.config.get('data_ref', {})
        label_exp = config_data.get('label', f"{config_data.get('model', 'N/A')} {config_data.get('exp', 'N/A')}")
        label_ref = config_data_ref.get('label', f"{config_data_ref.get('model', 'N/A')} {config_data_ref.get('exp', 'N/A')}")

        for var_name in common_keys:
            self.logger.info(f"Processing vertical zonal mean profile for: {var_name}")

            data = self.retrieved_data[var_name]
            data_ref = self.retrieved_data_ref[var_name]

            try:
                # Check for pressure level coordinate
                lev_name_ref = self._find_level_coord(data_ref)
                lev_name_data = self._find_level_coord(data)

                if lev_name_ref is None or lev_name_data is None:
                    self.logger.warning(f"Variable {var_name} does not have pressure levels. Skipping.")
                    continue

                # Find common pressure levels
                common_levels, lev_name_ref, lev_name_data = self._find_common_levels(data_ref, data)
                if common_levels is None or len(common_levels) == 0:
                    self.logger.warning(f"No common pressure levels found for {var_name}. Skipping.")
                    continue

                self.logger.debug(f"Using {len(common_levels)} common pressure levels for {var_name}")

                # Select common levels
                data_ref = data_ref.sel({lev_name_ref: common_levels})
                data = data.sel({lev_name_data: common_levels})

                # Ensure consistent coordinate naming
                if lev_name_data != lev_name_ref:
                    data = data.rename({lev_name_data: lev_name_ref})

                lev_name = lev_name_ref

                # Coordinate Check/Conversion
                data, data_ref = self.check_and_convert_coords(data, data_ref)
                self.logger.debug(f"Coordinates checked/converted for {var_name}")

                significance_mask = None
                significance_cfg = self.plot_settings['significance']
                if significance_cfg['show']:
                    self.logger.info("Computing significance mask for %s", var_name)
                    zonal_ts_ref = self._compute_zonal_mean(data_ref)
                    zonal_ts_data = self._compute_zonal_mean(data)

                    if self.latitude_bins is not None:
                        zonal_ts_ref = self._bin_latitude(zonal_ts_ref, self.latitude_bins)
                        zonal_ts_data = self._bin_latitude(zonal_ts_data, self.latitude_bins)

                    significance_mask = self._compute_significance_ttest(
                        zonal_ts_data,
                        zonal_ts_ref,
                        alpha=significance_cfg['alpha'],
                    )

                # Compute temporal aggregation
                temporal_ref = self._compute_temporal_aggregation(data_ref)
                temporal_data = self._compute_temporal_aggregation(data)
                self.logger.debug(f"Temporal aggregation computed for {var_name}")

                # Compute zonal mean (average over longitude)
                zm_ref = self._compute_zonal_mean(temporal_ref)
                zm_data = self._compute_zonal_mean(temporal_data)
                self.logger.debug(f"Zonal mean computed for {var_name}")

                # Apply latitude binning if configured
                if self.latitude_bins is not None:
                    zm_ref = self._bin_latitude(zm_ref, self.latitude_bins)
                    zm_data = self._bin_latitude(zm_data, self.latitude_bins)
                    self.logger.debug(f"Latitude binning applied for {var_name}")

                # Compute difference
                zm_diff = zm_data - zm_ref

                # Get variable metadata
                units = data.attrs.get('units', '')
                long_name = data.attrs.get('long_name', var_name)

                # Transpose to ensure correct dimension order (lat, plev)
                lat_name = [dim for dim in zm_ref.dims if 'lat' in dim.lower()][0]
                zm_ref = zm_ref.transpose(lev_name, lat_name)
                zm_data = zm_data.transpose(lev_name, lat_name)
                zm_diff = zm_diff.transpose(lev_name, lat_name)
                if significance_mask is not None:
                    significance_mask = significance_mask.transpose(lev_name, lat_name)

                # Plotting
                fig = self._plot_vertical_zonal_mean(zm_ref, zm_data, zm_diff,
                                                      long_name, units,
                                                      label_ref, label_exp, lev_name,
                                                      significance_mask=significance_mask)

                self.logger.info(f"Vertical zonal mean profile plot generated for {var_name}")

                # Store results
                results[var_name] = fig

                if save_fig:
                    self._save_figure(fig, var_name)
                    plt.close(fig)

                if save_data:
                    self._save_data(zm_ref, zm_data, zm_diff, var_name, lev_name, significance_mask=significance_mask)

            except Exception as e:
                self.logger.error(f"Failed to calculate or plot vertical zonal mean for {var_name}: {e}", exc_info=True)
                if var_name in results:
                    del results[var_name]

        self.logger.info("Finished vertical zonal mean profile processing.")

        return results
