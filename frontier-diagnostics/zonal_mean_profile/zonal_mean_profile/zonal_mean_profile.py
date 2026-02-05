import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from aqua.core.logger import log_configure
from aqua.core.util import load_yaml, coord_names
from aqua.core.reader import Reader
from aqua.core.fixer import EvaluateFormula
from aqua.core.timstat import TimStat

class ZonalMeanProfile:
    """Zonal Mean Profile diagnostic that computes zonal mean (latitude profile) for variables and formulas.
    
    This diagnostic computes the zonal mean (average over longitude) of variables or formulas
    (e.g., precipitation - evaporation), comparing emulator results to reference model data.
    """

    def __init__(self, config, loglevel: str = 'WARNING'):
        """Initialize the ZonalMeanProfile diagnostic class.

        Args:
            config: Configuration for the diagnostic (dict or path to YAML file).
            loglevel (str): Logging level. Defaults to 'WARNING'.

        Raises:
            ValueError: If required configuration sections or keys are missing.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'ZonalMeanProfile')

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
        self.outputdir_fig = self.config.get('outputdir_fig', './figs_zonal_mean_profile')
        self.outputdir_data = self.config.get('outputdir_data', './data_zonal_mean_profile')
        self.figure_format = self._normalise_extension(self.config.get('figure_format', 'pdf'))
        os.makedirs(self.outputdir_fig, exist_ok=True)
        os.makedirs(self.outputdir_data, exist_ok=True)

        # Load plot settings
        self.plot_settings = self._load_plot_settings()

        # Initialize TimStat for temporal aggregation
        self.timstat = TimStat(loglevel=self.loglevel)

        # Parse aggregation configuration
        self._parse_aggregation_config()

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
                                      areas=False,
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel)

        self.reader_data = Reader(catalog=config_data.get('catalog'),
                                  model=config_data.get('model'),
                                  exp=config_data.get('exp'),
                                  source=config_data.get('source'),
                                  regrid=config_data.get('regrid'),
                                  fix=config_data.get('fix'),
                                  areas=False,
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel)

        self.logger.debug('Reader classes initialized for data_ref and data')

    def _load_plot_settings(self):
        """Load plot settings from configuration"""
        plot_cfg = self.config.get('plot', {})
        
        # Default settings
        settings = {
            'figure_size': [10, 6],
            'linewidth': 1.5,
        }
        
        # Update with user-provided settings
        if 'figure_size' in plot_cfg:
            settings['figure_size'] = plot_cfg['figure_size']
        if 'linewidth' in plot_cfg:
            settings['linewidth'] = float(plot_cfg['linewidth'])
        
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
        Retrieves data for all variables/formulas specified in the configuration
        from both the reference and main data sources using the initialized Reader classes.
        Stores retrieved data in self.retrieved_data_ref and self.retrieved_data dictionaries.
        Handles both simple variables and formulas.
        """
        self.retrieved_data_ref = {}
        self.retrieved_data = {}
        variables_to_process = self.config.get('variables', [])

        if not variables_to_process:
            self.logger.warning("No variables section found in the configuration. Nothing to retrieve.")
            return

        self.logger.info("Starting data retrieval...")

        # Process each variable or formula
        for var_info in variables_to_process:
            var_name = var_info.get('name')
            if not var_name:
                self.logger.warning("Skipping variable entry with no name: %s", var_info)
                continue

            is_formula = var_info.get('formula', False)
            units = var_info.get('units', None)
            long_name = var_info.get('long_name', None)
            standard_name = var_info.get('standard_name', var_name)

            # Use variable name or formula string as the key
            data_key = var_name

            if is_formula:
                self.logger.info(f"Processing formula: {var_name}")
                # Retrieve all data needed for the formula
                try:
                    # For formulas, we need to retrieve all required variables first
                    # The Reader will handle this when we retrieve with fix=True
                    data_ref = self.reader_data_ref.retrieve()
                    data = self.reader_data.retrieve()

                    # Apply regridding if configured
                    if self.reader_data_ref.tgt_grid_name:
                        self.logger.debug(f"Applying regridding to reference data")
                        for var in data_ref.data_vars:
                            data_ref[var] = self.reader_data_ref.regrid(data_ref[var])

                    if self.reader_data.tgt_grid_name:
                        self.logger.debug(f"Applying regridding to main data")
                        for var in data.data_vars:
                            data[var] = self.reader_data.regrid(data[var])

                    # Evaluate the formula for reference data
                    self.logger.debug(f"Evaluating formula '{var_name}' for reference data")
                    evaluated_ref = EvaluateFormula(
                        data=data_ref,
                        formula=var_name,
                        long_name=long_name,
                        short_name=standard_name,
                        units=units,
                        loglevel=self.loglevel
                    ).evaluate()

                    if evaluated_ref is None:
                        raise ValueError(f'Error evaluating formula {var_name} for reference data')

                    # Evaluate the formula for main data
                    self.logger.debug(f"Evaluating formula '{var_name}' for main data")
                    evaluated_data = EvaluateFormula(
                        data=data,
                        formula=var_name,
                        long_name=long_name,
                        short_name=standard_name,
                        units=units,
                        loglevel=self.loglevel
                    ).evaluate()

                    if evaluated_data is None:
                        raise ValueError(f'Error evaluating formula {var_name} for main data')

                    self.retrieved_data_ref[data_key] = evaluated_ref
                    self.retrieved_data[data_key] = evaluated_data
                    self.logger.debug(f"Successfully evaluated formula: {data_key}")

                except Exception as e:
                    self.logger.error(f"An unexpected error occurred evaluating formula {var_name}: {e}", exc_info=True)
                    continue

            else:
                # Simple variable retrieval
                self.logger.info(f"Retrieving variable: {var_name}")
                
                # Retrieve reference data
                try:
                    self.logger.debug(f"Retrieving reference data for {var_name}")
                    data_ref = self.reader_data_ref.retrieve(var=var_name)
                    if isinstance(data_ref, xr.Dataset):
                        data_ref = data_ref[var_name]

                    # Apply regridding if configured
                    if self.reader_data_ref.tgt_grid_name:
                        self.logger.debug(f"Applying regridding to reference data for {var_name}")
                        data_ref = self.reader_data_ref.regrid(data_ref)

                    self.retrieved_data_ref[data_key] = data_ref
                    self.logger.debug(f"Successfully retrieved reference data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving reference {var_name}: {e}", exc_info=True)
                    continue

                # Retrieve Main Data
                try:
                    self.logger.debug(f"Retrieving main data for {var_name}")
                    data = self.reader_data.retrieve(var=var_name)
                    if isinstance(data, xr.Dataset):
                        data = data[var_name]

                    # Apply regridding if configured
                    if self.reader_data.tgt_grid_name:
                        self.logger.debug(f"Applying regridding to main data for {var_name}")
                        data = self.reader_data.regrid(data)

                    self.retrieved_data[data_key] = data
                    self.logger.debug(f"Successfully retrieved main data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving main {var_name}: {e}", exc_info=True)
                    # Clean up ref data if main data failed
                    if data_key in self.retrieved_data_ref:
                        del self.retrieved_data_ref[data_key]
                    continue

        self.logger.info("Data retrieval finished.")
        if not self.retrieved_data_ref and not self.retrieved_data:
             self.logger.warning("No data was successfully retrieved for any variable.")
        else:
             # Log the keys
             self.logger.info(f"Retrieved reference data for keys: {list(self.retrieved_data_ref.keys())}")
             self.logger.info(f"Retrieved main data for keys: {list(self.retrieved_data.keys())}")

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
    def _percentile_func(data, q):
        """
        Compute percentile over time dimension using xarray quantile.
        
        This is a static method designed to be used as a callable function with TimStat.
        
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

    def compute_zonal_mean(self, data):
        """
        Compute zonal mean (average over longitude) of the data after temporal aggregation.
        
        The temporal aggregation is controlled by self.aggregation_stat and can be:
        - 'mean': temporal mean
        - 'std': temporal standard deviation
        - 'max': temporal maximum
        - 'min': temporal minimum
        - 'percentile': temporal percentile (value set by self.percentile)
        
        Args:
            data (xr.DataArray): Data array with time, lat, lon dimensions
            
        Returns:
            xr.DataArray: Zonal mean (latitude profile) of the temporally aggregated data
        """
        # First compute temporal aggregation using TimStat
        if self.aggregation_stat == 'percentile':
            # For percentile, use custom function with func_kwargs
            self.logger.debug(f"Computing {self.percentile}th percentile over time dimension")
            temporal_stat = self.timstat.timstat(
                data, 
                stat=self._percentile_func,
                freq=None,  # None means aggregate over entire time period
                func_kwargs={'q': self.percentile}
            )
        else:
            # For built-in stats (mean, std, max, min)
            self.logger.debug(f"Computing temporal {self.aggregation_stat}")
            temporal_stat = self.timstat.timstat(
                data,
                stat=self.aggregation_stat,
                freq=None  # None means aggregate over entire time period
            )
        
        # Then compute zonal mean (average over longitude)
        lon_name = [dim for dim in temporal_stat.dims if 'lon' in dim.lower()][0]
        zonal_mean = temporal_stat.mean(dim=lon_name)
        
        # Squeeze to remove singleton dimensions (e.g., plev with size 1)
        zonal_mean = zonal_mean.squeeze()
        
        return zonal_mean

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

    def _generate_filename(self, var_key, suffix):
        """
        Generates a unique filename based on config and context.

        Args:
            var_key (str): The variable key.
            suffix (str): Suffix describing the calculation (e.g., 'zonal_mean_profile').

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

        # Sanitize variable key for filename
        var_key_safe = self._sanitize_filename_part(var_key)

        # Construct the base filename from sanitized parts
        base_name_parts = [model, exp, source,
                           'vs', model_ref, exp_ref,
                           var_key_safe,
                           startdate, enddate,
                           suffix]

        # Filter out any empty strings
        base_filename = '_'.join(filter(None, base_name_parts))

        # Determine extension
        extension = f'.{self.figure_format}'

        full_path = os.path.join(self.outputdir_fig, f"{base_filename}{extension}")
        self.logger.debug(f"Generated path for figure ({var_key}, {suffix}): {full_path}")
        return full_path

    def _save_figure(self, fig, var_key):
        """
        Saves the generated figure with a unique name based on config.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            var_key (str): The variable key.
        """
        # Generate suffix with aggregation statistic
        if self.aggregation_stat == 'percentile':
            stat_suffix = f'percentile_{str(self.percentile).replace(".", "_")}'
        else:
            stat_suffix = self.aggregation_stat
        suffix = f'zonal_mean_{stat_suffix}_profile'
        
        filename = self._generate_filename(var_key, suffix)

        try:
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

    def _save_data(self, zonal_mean_ref, zonal_mean_data, var_key):
        """
        Saves the computed zonal mean data to NetCDF.

        Args:
            zonal_mean_ref (xr.DataArray): Reference zonal mean profile.
            zonal_mean_data (xr.DataArray): Experiment zonal mean profile.
            var_key (str): The variable key.
        """
        # Generate filename using similar pattern to figures
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

        var_key_safe = self._sanitize_filename_part(var_key)

        # Generate suffix with aggregation statistic
        if self.aggregation_stat == 'percentile':
            stat_suffix = f'percentile_{str(self.percentile).replace(".", "_")}'
        else:
            stat_suffix = self.aggregation_stat
        suffix = f'zonal_mean_{stat_suffix}_profile'

        base_name_parts = [model, exp, source,
                          'vs', model_ref, exp_ref,
                          var_key_safe,
                          startdate, enddate,
                          suffix]

        base_filename = '_'.join(filter(None, base_name_parts))
        filepath = os.path.join(self.outputdir_data, f"{base_filename}.nc")

        try:
            # Create a dataset with both profiles
            ds = xr.Dataset({
                'zonal_mean_experiment': zonal_mean_data.rename('zonal_mean_experiment'),
                'zonal_mean_reference': zonal_mean_ref.rename('zonal_mean_reference'),
            })
            
            # Add metadata
            ds.attrs['variable'] = var_key
            ds.attrs['startdate'] = str(self.startdate)
            ds.attrs['enddate'] = str(self.enddate)
            ds.attrs['model'] = data_cfg.get('model', 'unknown')
            ds.attrs['model_ref'] = ref_cfg.get('model', 'unknown')
            ds.attrs['aggregation_stat'] = self.aggregation_stat
            if self.aggregation_stat == 'percentile':
                ds.attrs['percentile'] = self.percentile
            
            ds.to_netcdf(filepath)
            self.logger.info(f"Data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save data {filepath}: {e}", exc_info=True)

    @staticmethod
    def _normalise_extension(extension):
        """Normalise the extension"""
        ext = extension.lower().lstrip(".")
        if not ext:
            return "pdf"
        else:
            return ext

    def compute_profile(self, save_fig: bool = False, save_data: bool = False):
        """
        Calculates and plots the zonal mean profiles for all variables retrieved.

        This method processes all common variables between the main and reference datasets,
        calculating the zonal mean (latitude profile) for each. The temporal aggregation
        statistic (mean, std, max, min, or percentile) is applied before computing the
        zonal mean, as configured in the aggregation section of the config file.

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
            - The temporal aggregation statistic is controlled by self.aggregation_stat
              and self.percentile attributes.
        """
        self.logger.info('Calculating and plotting zonal mean profiles for retrieved variables.')

        # Ensure data has been retrieved
        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
             self.logger.error("Data has not been retrieved yet. Call the retrieve() method first.")
             return {}

        # Find common keys retrieved in both datasets
        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys())
        if not common_keys:
            self.logger.warning("No common data keys found between reference and main retrieved data. Cannot calculate profiles.")
            return {}

        results = {}

        # Get metadata for labels
        config_data = self.config.get('data', {})
        config_data_ref = self.config.get('data_ref', {})
        label_exp = config_data.get('label', f"{config_data.get('model', 'N/A')} {config_data.get('exp', 'N/A')}")
        label_ref = config_data_ref.get('label', f"{config_data_ref.get('model', 'N/A')} {config_data_ref.get('exp', 'N/A')}")

        for var_key in common_keys:
            self.logger.info(f"Processing zonal mean profile for: {var_key}")

            data = self.retrieved_data[var_key]
            data_ref = self.retrieved_data_ref[var_key]

            try:
                # Coordinate Check/Conversion
                data, data_ref = self.check_and_convert_coords(data, data_ref)
                self.logger.debug(f"Coordinates checked/converted for {var_key}")

                # Compute zonal mean profiles
                zonal_mean_data = self.compute_zonal_mean(data)
                zonal_mean_ref = self.compute_zonal_mean(data_ref)
                
                self.logger.debug(f"Zonal mean profiles calculated for {var_key}")

                # Get variable metadata
                units = data.attrs.get('units', '')
                long_name = data.attrs.get('long_name', var_key)

                # Plotting
                fig = self._plot_zonal_profile(zonal_mean_ref, zonal_mean_data,
                                               long_name, units,
                                               label_ref, label_exp, var_key)

                self.logger.info(f"Zonal mean profile plot generated for {var_key}")

                # Store results
                results[var_key] = fig

                if save_fig:
                    self._save_figure(fig, var_key)
                    plt.close(fig)
                
                if save_data:
                    self._save_data(zonal_mean_ref, zonal_mean_data, var_key)

            except Exception as e:
                self.logger.error(f"Failed to calculate or plot zonal mean profile for {var_key}: {e}", exc_info=True)
                if var_key in results: 
                    del results[var_key]

        self.logger.info("Finished zonal mean profile processing.")

        return results

    def _plot_zonal_profile(self, data_ref, data, var_name, units, label_ref, label_exp, var_key):
        """
        Create a figure showing zonal mean profiles comparing reference and experiment data.

        Args:
            data_ref: Zonal mean profile of reference data
            data: Zonal mean profile of main data
            var_name: Variable name for labels
            units: Units of the variable
            label_ref: Label for reference data
            label_exp: Label for experiment data
            var_key: Variable key for title

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get latitude coordinate
        try:
            lat_name = [dim for dim in data_ref.dims if 'lat' in dim.lower()][0]
        except IndexError:
            raise ValueError("Unable to determine latitude coordinate for plotting.")

        lat = data_ref[lat_name].values

        # Create figure
        figsize = tuple(self.plot_settings['figure_size'])
        linewidth = self.plot_settings['linewidth']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot profiles
        ax.plot(lat, data_ref.values, label=label_ref, linewidth=linewidth, color='blue')
        ax.plot(lat, data.values, label=label_exp, linewidth=linewidth, color='red')
        
        # Formatting
        ax.set_xlabel('Latitude', fontsize=12)
        ylabel = f'{var_name}'
        if units:
            ylabel += f' [{units}]'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)
        
        # Set title with aggregation statistic
        if self.aggregation_stat == 'percentile':
            stat_label = f"{self.percentile}th percentile"
        else:
            stat_label = self.aggregation_stat
        title = f"Zonal Mean {stat_label.title()}: {var_name}\n{self.startdate} to {self.enddate}"
        ax.set_title(title, fontsize=14)
        
        fig.tight_layout()
        return fig
