import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from smmregrid import GridInspector
from aqua.core.logger import log_configure
from aqua.core.util import load_yaml
from aqua.core.reader import Reader
from aqua.core.graphics import plot_single_map

class ACC: 
    """ACC diagnostic"""

    def __init__(self, config, loglevel: str = 'WARNING'):
        """Initialize the ACC diagnostic class.

        Args:
            config: Configuration for the ACC diagnostic
            loglevel (str, optional): Logging level. Defaults to 'WARNING'.

        Raises:
            ValueError: If required configuration sections or keys are missing.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'ACC')

        # Load configuration
        if isinstance(config, str):
            self.logger.debug("Reading configuration file %s", config)
            self.config = load_yaml(config)
        else:
            self.logger.debug("Configuration is a dictionary")
            self.config = config

        # Basic validation for required sections (besides the dates and climatology dates)
        required_sections = ['data', 'data_ref', 'climatology_data', 'dates', 'variables']
        missing_sections = [s for s in required_sections if s not in self.config]
        if missing_sections:
            msg = f"Missing required configuration sections: {', '.join(missing_sections)}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Evaluation dates
        self.startdate = self.config['dates'].get('startdate')
        self.enddate = self.config['dates'].get('enddate')
        if not self.startdate or not self.enddate:
            raise ValueError("Evaluation startdate and enddate must be provided in dates.")

        # Climatology dates
        self.clim_startdate = self.config['climatology_data'].get('startdate')
        self.clim_enddate = self.config['climatology_data'].get('enddate')
        if not self.clim_startdate or not self.clim_enddate:
            raise ValueError("Climatology startdate and enddate must be provided in climatology_data.")

        # Initialize the reader classes
        self._reader()

        # Initialize cache for climatology means
        # (to avoid recalculating the climatology means for each variable)
        self._climatology_means_cache = None

    def _reader(self):
        """Initializes the Reader class for both reference and main data"""
        # Get the data configurations
        config_data_ref = self.config.get('data_ref', {})
        config_data = self.config.get('data', {})
        config_climatology = self.config.get('climatology_data', {})

        # Check if necessary keys exist within the data configurations
        if not all(k in config_data_ref for k in ['catalog', 'model', 'exp', 'source', 'regrid', 'fix']):
             self.logger.warning("Reference data configuration ('data_ref') might be incomplete.")
        if not all(k in config_data for k in ['catalog', 'model', 'exp', 'source', 'regrid', 'fix']):
             self.logger.warning("Main data configuration ('data') might be incomplete.")

        # Initialize the reader classes
        self.reader_data_ref = Reader(
            catalog=config_data_ref.get('catalog'),
            model=config_data_ref.get('model'),
            exp=config_data_ref.get('exp'),
            source=config_data_ref.get('source'),
            regrid=config_data_ref.get('regrid'),
            fix=config_data_ref.get('fix'),
            startdate=self.startdate,
            enddate=self.enddate,
            loglevel=self.loglevel
        )

        self.reader_data = Reader(
            catalog=config_data.get('catalog'),
            model=config_data.get('model'),
            exp=config_data.get('exp'),
            source=config_data.get('source'),
            regrid=config_data.get('regrid'),
            fix=config_data.get('fix'),
            startdate=self.startdate,
            enddate=self.enddate,
            loglevel=self.loglevel
        )

        self.reader_climatology = Reader(
            catalog=config_climatology.get('catalog'),
            model=config_climatology.get('model'),
            exp=config_climatology.get('exp'),
            source=config_climatology.get('source'),
            regrid=config_climatology.get('regrid'),
            fix=config_climatology.get('fix'),
            startdate=self.clim_startdate,
            enddate=self.clim_enddate,
            loglevel=self.loglevel
        )

        self.logger.debug('Reader classes initialized for data_ref, data, and climatology')

    def retrieve(self):
        """
        Retrieves data for all variables specified in the configuration using the initialized readers.
        Handles single levels, lists of levels, and surface variables.
        """
        # Reset the climatology means cache when retrieving new data
        self.logger.debug("Resetting climatology means cache.")
        self._climatology_means_cache = None

        self.retrieved_data_ref = {}
        self.retrieved_data = {}
        self.retrieved_climatology_data = {}
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

            levels = var_info.get('level')

            # Ensure levels is a list to simplify iteration
            if levels is None:
                levels_to_iterate = [None]
            elif isinstance(levels, (int, float)):
                levels_to_iterate = [int(levels)]
            elif isinstance(levels, list):
                levels_to_iterate = [int(lvl) for lvl in levels]
            else:
                self.logger.warning(f"Invalid level format for variable {var_name}: {levels}. Skipping.")
                continue

            # Process each level
            for level in levels_to_iterate:
                # Determine the key for storing data (e.g., 'q_85000' or '2t')
                data_key = f"{var_name}_{level}" if level is not None else var_name
                log_msg_suffix = f" at level {level}" if level is not None else " (surface)"

                # Arguments for Reader.retrieve
                retrieve_args = {'var': var_name}
                if level is not None:
                    retrieve_args['level'] = level

                # Check the status of the data to load (reference, main, and climatology)
                ref_ok, data_ok, clim_ok = False, False, False

                # Retrieve reference data
                try:
                    self.logger.debug(f"Retrieving reference data for {var_name}{log_msg_suffix}")
                    data_ref = self.reader_data_ref.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                    if level is not None and 'plev' in data_ref.coords:
                        data_ref = data_ref.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_data_ref.tgt_grid_name is not None:
                        self.logger.debug(f"Applying regridding to reference data for {var_name}{log_msg_suffix}")
                        data_ref = self.reader_data_ref.regrid(data_ref)
                    self.retrieved_data_ref[data_key] = data_ref

                    ref_ok = True # Reference data retrieved successfully
                    self.logger.debug(f"Successfully retrieved reference data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving reference {var_name}{log_msg_suffix}: {e}", exc_info=True)
                    continue

                # Retrieve main data
                try:
                    self.logger.debug(f"Retrieving main data for {var_name}{log_msg_suffix}")
                    data = self.reader_data.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                    if level is not None and 'plev' in data.coords:
                        data = data.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_data.tgt_grid_name is not None:
                        self.logger.debug(f"Applying regridding to main data for {var_name}{log_msg_suffix}")
                        data = self.reader_data.regrid(data)
                    self.retrieved_data[data_key] = data

                    data_ok = True # Main data retrieved successfully
                    self.logger.debug(f"Successfully retrieved main data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving main {var_name}{log_msg_suffix}: {e}", exc_info=True)
                     # Clean up ref data if main data failed for this level
                    if data_key in self.retrieved_data_ref:
                        del self.retrieved_data_ref[data_key]
                    continue

                # Retrieve climatology data
                try:
                    self.logger.debug(f"Retrieving climatology data for {var_name}{log_msg_suffix}")
                    data_clim = self.reader_climatology.retrieve(**retrieve_args)
                    if level is not None and 'plev' in data_clim.coords:
                        data_clim = data_clim.isel(plev=0, drop=True)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                    if level is not None and 'plev' in data_clim.coords:
                        data_clim = data_clim.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_climatology.tgt_grid_name is not None:
                        self.logger.debug(f"Applying regridding to climatology data for {var_name}{log_msg_suffix}")
                        data_clim = self.reader_climatology.regrid(data_clim)
                    self.retrieved_climatology_data[data_key] = data_clim

                    clim_ok = True # Climatology data retrieved successfully
                    self.logger.debug(f"Successfully retrieved climatology data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"Failed retrieving MANDATORY climatology {data_key}: {e}", exc_info=False)
                    self.logger.error(f"Cannot compute ACC for {data_key} without climatology data.")
                    
                    # Clean up potentially retrieved main/ref data for this key if clim failed
                    if data_key in self.retrieved_data: del self.retrieved_data[data_key]
                    if data_key in self.retrieved_data_ref: del self.retrieved_data_ref[data_key]
                    continue

                # Ensure all were retrieved successfully if climatology was needed
                if not (ref_ok and data_ok and clim_ok):
                     self.logger.warning(f"Skipping {data_key} due to incomplete data retrieval (ref: {ref_ok}, main: {data_ok}, clim: {clim_ok}).")
                     
                     # Clean up partial retrievals for this key
                     if data_key in self.retrieved_data: del self.retrieved_data[data_key]
                     if data_key in self.retrieved_data_ref: del self.retrieved_data_ref[data_key]
                     if data_key in self.retrieved_climatology_data: del self.retrieved_climatology_data[data_key]

         # Keys for which all data
        valid_keys = list(self.retrieved_climatology_data.keys())
        self.logger.info(f"Data retrieval finished. Valid keys for ACC: {valid_keys}")

        if self.reader_data_ref.tgt_grid_name is not None:
            self.logger.info(f"Reference data regridded to: {self.reader_data_ref.tgt_grid_name}")
        if self.reader_data.tgt_grid_name is not None:
            self.logger.info(f"Main data regridded to: {self.reader_data.tgt_grid_name}")
        if self.reader_climatology.tgt_grid_name is not None:
            self.logger.info(f"Climatology data regridded to: {self.reader_climatology.tgt_grid_name}")
        if not valid_keys:
             self.logger.warning("No variables/levels had data successfully retrieved from all three sources (ref, main, clim). ACC cannot be computed.")

    @staticmethod
    def _ensure_float32_coords(array):
        """
        Ensures the lat/lon coordinates of a DataArray are float32.

        Args:
            array (xr.DataArray): The DataArray to check and potentially convert.

        Returns:
            xr.DataArray: The DataArray with float32 lat/lon coordinates.

        Note: I am aware this is similar to the check_and_convert_coords method in the RMSE diagnostic.
        TODO: Move this logic to a shared utility module.
        """
        if not isinstance(array, xr.DataArray):
            print(f"Warning: Input to _ensure_float32_coords is not a DataArray. Skipping.")
            return array

        # Get coordinate names for lat and lon using the base variable name
        # TODO: Generalize this to handle other coordinate names
        lat_names = [dim for dim in array.coords if 'lat' in dim.lower()]
        lon_names = [dim for dim in array.coords if 'lon' in dim.lower()]

        if not lat_names or not lon_names:
            print(f"Could not find lat/lon coordinates. Skipping conversion.")
            return array

        # Check if conversion is needed (if the coordinates are not already float32)
        lat_name = lat_names[0]
        lon_name = lon_names[0]

        needs_conversion = False
        if array[lat_name].dtype != np.float32:
            needs_conversion = True
        if array[lon_name].dtype != np.float32:
            needs_conversion = True

        if needs_conversion:
            print(f"Converting lat/lon coordinate types to float32")
            try:
                array = array.assign_coords({lat_name: array[lat_name].astype('float32'),
                                             lon_name: array[lon_name].astype('float32')})
            except Exception as e:
                print(f"Failed to convert coordinate types: {e}")

        return array

    # TODO: Move this logic to a shared utility module.
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

    # TODO: Move this logic to a shared utility module.
    def _generate_filename(self, processed_key, output_type, suffix):
        """
        Generates a unique filename based on config and context.

        Args:
            processed_key (str): The variable key (e.g., 'q_85000', '2t').
            output_type (str): Type of output ('figure' or 'netcdf').
            suffix (str): Suffix describing the calculation (e.g., 'spatial_acc', 'temporal_acc').

        Returns:
            str: The full, unique path for the output file.
        """
        # Get config details with defaults and sanitize them
        data_cfg = self.config.get('data', {})
        ref_cfg = self.config.get('data_ref', {})
        dates_cfg = self.config.get('dates', {})
        clim_cfg = self.config.get('climatology_data', {})

        model = self._sanitize_filename_part(data_cfg.get('model', 'model'))
        exp = self._sanitize_filename_part(data_cfg.get('exp', 'exp'))
        source = self._sanitize_filename_part(data_cfg.get('source', 'src'))
        
        model_ref = self._sanitize_filename_part(ref_cfg.get('model', 'refmodel'))
        exp_ref = self._sanitize_filename_part(ref_cfg.get('exp', 'refexp'))
        source_ref = self._sanitize_filename_part(ref_cfg.get('source', 'refsrc'))

        startdate = self._sanitize_filename_part(dates_cfg.get('startdate', 'nodate'))
        enddate = self._sanitize_filename_part(dates_cfg.get('enddate', 'nodate'))
        
        clim_startdate = self._sanitize_filename_part(clim_cfg.get('startdate', 'nodate'))
        clim_enddate = self._sanitize_filename_part(clim_cfg.get('enddate', 'nodate'))

        # Construct the base filename from sanitized parts
        base_name_parts = [
            model, exp, source,
            'vs', model_ref, exp_ref, source_ref,
            clim_startdate, clim_enddate,
            processed_key,
            suffix
        ]

        # Filter out any empty strings that might result from missing config values
        base_filename = '_'.join(filter(None, base_name_parts))

        # Determine directory and extension based on output_type
        if output_type == 'figure':
            output_dir = self.config.get('outputdir_fig', './figs')
            extension = '.pdf'
        elif output_type == 'netcdf':
            output_dir = self.config.get('outputdir_data', './output')
            extension = '.nc'
        else:
            self.logger.error(f"Unknown output_type requested for filename generation: {output_type}")

        # Ensure the output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Could not create output directory {output_dir}: {e}")
            output_dir = '.'

        full_path = os.path.join(output_dir, f"{base_filename}{extension}")
        self.logger.debug(f"Generated path for {output_type} ({processed_key}, {suffix}): {full_path}")
        return full_path

    # TODO: Move this logic to a shared utility module.
    def _save_figure(self, fig, processed_key, plot_type):
        """
        Saves the generated figure with a unique name based on config.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            processed_key (str): The variable key (e.g., 'q_85000', '2t').
            plot_type (str): The type of plot ('spatial' or 'temporal').
        """
        filename = self._generate_filename(processed_key, 'figure', f'{plot_type}_acc')
        try:
            fig.savefig(filename)
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

    def _save_netcdf(self, data, processed_key, data_type):
        """
        Saves the generated netcdf file with a unique name based on config.

        Args:
            data (xr.Dataset): The data to save.
            processed_key (str): The variable key (e.g., 'q_85000', '2t').
            data_type (str): The type of data ('spatial' or 'temporal').
        """
        filename = self._generate_filename(processed_key, 'netcdf', f'{data_type}_acc')
        try:
            data.to_netcdf(filename)
            self.logger.info(f"NetCDF data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save NetCDF {filename}: {e}", exc_info=True)

    # TODO: Move this logic to a shared utility module.
    @staticmethod
    def _parse_processed_key(processed_key):
        """Parses the processed key into base variable name and level."""
        parts = processed_key.split('_')
        level = None
        base_var_name = processed_key

        # Check if the last part is likely a level (numeric)
        if len(parts) > 1 and parts[-1].isdigit():
            try:
                level = int(parts[-1])
                base_var_name = '_'.join(parts[:-1])
            except ValueError:
                 pass

        return base_var_name, level

    def _calculate_anomalies(self, data_array, climatology_mean):
        """Calculates temporal anomalies by subtracting the provided climatology mean.

        Args:
            data_array (xr.DataArray): The data to calculate anomalies for.
            climatology_mean (xr.DataArray): The pre-calculated climatology mean
                                             (spatial map) from the external dataset.

        Returns:
            xr.DataArray: The calculated anomalies.
        """
        if climatology_mean is None:
            msg = f"'Climatology_mean' was None for {data_array.name}."
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(data_array, xr.DataArray) or not isinstance(climatology_mean, xr.DataArray):
             msg = "Inputs must be xarray DataArrays."
             self.logger.error(msg)
             raise TypeError(msg)

        try:
            anomalies = data_array - climatology_mean
            anomalies.attrs = data_array.attrs
            anomalies.name = f"{data_array.name}_anom"
            return anomalies
        except Exception as e:
            msg = f"Error subtracting climatology mean for {data_array.name}. Check dimensions/coords compatibility. Error: {e}"
            self.logger.error(msg, exc_info=True)
            raise ValueError(msg) from e

    def _prepare_climatology_means(self):
        """Calculates and returns a dictionary of climatology means, using a cache."""
        # Check cache first
        if self._climatology_means_cache is not None:
            self.logger.info("Using cached climatology means.")
            return self._climatology_means_cache

        # Proceed with calculation if not cached
        self.logger.info("Calculating climatology means...")
        climatology_means = {}

        # Check if the climatology data dictionary is, for some unknown reason, missing or empty
        if not self.retrieved_climatology_data:
             self.logger.error("Climatology data dictionary is missing or empty. Cannot calculate means.")
             self._climatology_means_cache = {}
             return self._climatology_means_cache

        # Compute the climatology means for each key
        for key, data_clim_ds in self.retrieved_climatology_data.items():
             base_var_name, _ = self._parse_processed_key(key)
             if base_var_name in data_clim_ds:
                 da_clim = data_clim_ds[base_var_name]
                 try:
                     if 'time' in da_clim.dims:
                         climatology_means[key] = self.reader_climatology.timmean(da_clim)
                         self.logger.debug(f"Calculated climatology mean for key: {key}")
                     else:
                         self.logger.warning(f"Climatology data for key '{key}' (variable '{base_var_name}') is missing 'time' dimension. Cannot calculate mean. Storing as is.")
                         climatology_means[key] = da_clim # Storing the data as it is (not time-varying)
                 except Exception as e:
                     self.logger.error(f"Could not calculate climatology mean for key {key}: {e}. Skipping this key.", exc_info=True)
             else:
                 self.logger.error(f"Variable '{base_var_name}' not found in retrieved climatology data for key '{key}'. Skipping mean calculation for this key.")

        # Cache the computed climatology means
        self.logger.info("Caching computed climatology means.")
        self._climatology_means_cache = climatology_means
        return climatology_means

    # TODO: Remove code duplication between spatial_acc and temporal_acc by using a shared function
    def spatial_acc(self, save_fig: bool = False, save_netcdf: bool = False):
        """
        Calculates and plots the spatial map of temporal Anomaly Correlation Coefficients (ACC)
        for all variables and levels retrieved.

        This method computes the correlation between the temporal anomaly time series
        of the main data and the reference data at each grid point.

        Args:
            save_fig (bool, optional): If True, saves the generated figures. Defaults to False.
            save_netcdf (bool, optional): If True, saves the ACC results as NetCDF files. Defaults to False.
        """
        self.logger.info('Calculating and plotting spatial ACC maps.')
        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
             self.logger.error("Data not retrieved. Call retrieve() first.")
             return {}

        climatology_means = self._prepare_climatology_means()
        if not climatology_means:
             self.logger.error("Cannot proceed without successfully calculated climatology means.")
             return {}

        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys()) & set(climatology_means.keys())
        if not common_keys:
            self.logger.warning("No common data keys found with valid climatology means. Cannot calculate ACC.")
            return {}

        self.logger.info(f"Processing spatial ACC for keys: {list(common_keys)}")

        results = {}

        config_data = self.config.get('data', {})
        config_data_ref = self.config.get('data_ref', {})
        model = config_data.get('model', 'N/A'); exp = config_data.get('exp', 'N/A'); source = config_data.get('source', 'N/A')
        model_ref = config_data_ref.get('model', 'N/A'); exp_ref = config_data_ref.get('exp', 'N/A'); source_ref = config_data_ref.get('source', 'N/A')

        # Compute the spatial ACC map for each key
        for processed_key in common_keys:
            base_var_name, level = self._parse_processed_key(processed_key)
            self.logger.info(f"Processing spatial ACC for key: {processed_key}")

            data_ds = self.retrieved_data[processed_key]
            data_ref_ds = self.retrieved_data_ref[processed_key]
            clim_mean_da = climatology_means[processed_key]

            # Check if the base variable name exists in the datasets
            if base_var_name not in data_ds or base_var_name not in data_ref_ds:
                self.logger.warning(f"Var '{base_var_name}' not in datasets for key '{processed_key}'. Skipping.")
                continue

            # Check if the climatology mean is a DataArray
            if not isinstance(clim_mean_da, xr.DataArray):
                 self.logger.warning(f"Climatology mean for key '{processed_key}' is not a DataArray. Skipping.")
                 continue

            try:
                # Get the DataArrays
                da = data_ds[base_var_name]
                da_ref = data_ref_ds[base_var_name]
                
                # Ensure all arrays have float32 coordinates
                da = self._ensure_float32_coords(da)
                da_ref = self._ensure_float32_coords(da_ref)
                clim_mean_da = self._ensure_float32_coords(clim_mean_da)

                # Calculate anomalies
                anom = self._calculate_anomalies(da, clim_mean_da)
                anom_ref = self._calculate_anomalies(da_ref, clim_mean_da)

                # Calculate spatial ACC map
                spatial_acc_map = xr.corr(anom, anom_ref, dim='time')
                spatial_acc_map.name = base_var_name

                # Plot the spatial ACC map
                vmin, vmax, cmap = -1, 1, 'RdBu_r'
                title_level_part = f" at {int(level)}" if level is not None else ""
                title = (f"Spatial ACC Map: {base_var_name}{title_level_part}\n"
                         f"{model} {exp} ({source}) vs {model_ref} {exp_ref} ({source_ref})\n"
                         f"{self.startdate} to {self.enddate} (Clim: {self.clim_startdate}-{self.clim_enddate})")
                fig, ax = plot_single_map(data=spatial_acc_map, return_fig=True, contour=True,
                                          title=title, sym=True, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

                # Set the colorbar label
                cbar_label = 'Anomaly Correlation Coefficient'
                if ax.collections: cbar = getattr(ax.collections[0], 'colorbar', None); cbar.set_label(cbar_label) if cbar else None
                elif ax.images: cbar = getattr(ax.images[0], 'colorbar', None); cbar.set_label(cbar_label) if cbar else None

                self.logger.info(f"Spatial ACC plot generated for {processed_key}")

                # Store the results
                results[processed_key] = (fig, ax, xr.Dataset({base_var_name: spatial_acc_map}))

            except Exception as e:
                self.logger.error(f"Failed processing spatial ACC for key {processed_key}: {e}", exc_info=True)
                if processed_key in results: del results[processed_key]

        self.logger.info("Finished spatial ACC processing.")
        if save_fig:
             self.logger.info("Saving spatial ACC figures...")
             for key, res in results.items():
                 if res[0]: self._save_figure(res[0], key, 'spatial')
        if save_netcdf:
             self.logger.info("Saving spatial ACC netcdf files...")
             for key, res in results.items():
                 if res[2]: self._save_netcdf(res[2], key, 'spatial')

        return None

    def temporal_acc(self, save_fig: bool = False, save_netcdf: bool = False):
        """
        Calculates and plots the temporal evolution of the spatial pattern
        Anomaly Correlation Coefficient (ACC) for all variables and levels retrieved.

        This method computes the spatial correlation between the anomaly patterns
        of the main data and the reference data for each time step.

        Args:
            save_fig (bool, optional): If True, saves the generated figures. Defaults to False.
            save_netcdf (bool, optional): If True, saves the ACC results as NetCDF files. Defaults to False.
        """
        self.logger.info('Calculating and plotting temporal evolution of spatial ACC.')

        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
             self.logger.error("Data not retrieved. Call retrieve() first.")
             return {}

        climatology_means = self._prepare_climatology_means()
        if not climatology_means:
             self.logger.error("Cannot proceed without successfully calculated climatology means.")
             return {}

        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys()) & set(climatology_means.keys())
        if not common_keys:
            self.logger.warning("No common data keys found with valid climatology means. Cannot calculate ACC.")
            return {}

        self.logger.info(f"Processing temporal ACC for keys: {list(common_keys)}")

        results = {}

        config_data = self.config.get('data', {})
        config_data_ref = self.config.get('data_ref', {})
        model = config_data.get('model', 'N/A'); exp = config_data.get('exp', 'N/A'); source = config_data.get('source', 'N/A')
        model_ref = config_data_ref.get('model', 'N/A'); exp_ref = config_data_ref.get('exp', 'N/A'); source_ref = config_data_ref.get('source', 'N/A')

        # Compute the temporal ACC for each key
        for processed_key in common_keys:
            base_var_name, level = self._parse_processed_key(processed_key)
            self.logger.info(f"Processing temporal ACC for key: {processed_key}")

            data_ds = self.retrieved_data[processed_key]
            data_ref_ds = self.retrieved_data_ref[processed_key]
            clim_mean_da = climatology_means[processed_key]

            # Check if the base variable name exists in the datasets
            if base_var_name not in data_ds or base_var_name not in data_ref_ds:
                self.logger.warning(f"Var '{base_var_name}' not in datasets for key '{processed_key}'. Skipping.")
                continue
            if not isinstance(clim_mean_da, xr.DataArray):
                 self.logger.warning(f"Climatology mean for key '{processed_key}' is not a DataArray. Skipping.")
                 continue

            try:
                # Get the DataArrays
                da = data_ds[base_var_name]
                da_ref = data_ref_ds[base_var_name]

                # Ensure all arrays have float32 coordinates
                da = self._ensure_float32_coords(da)
                da_ref = self._ensure_float32_coords(da_ref)
                clim_mean_da = self._ensure_float32_coords(clim_mean_da)
                self.logger.debug(f"Coordinates ensured float32 for {processed_key}")

                # Calculate Anomalies
                anom = self._calculate_anomalies(da, clim_mean_da)
                anom_ref = self._calculate_anomalies(da_ref, clim_mean_da)
                self.logger.debug(f"Anomalies calculated for {processed_key}")

                # Determine spatial dimensions using GridInspector
                spatial_dims = []
                try:
                    grid_types = GridInspector(anom).get_gridtype()
                    if grid_types:
                        spatial_dims = [dim for dim in grid_types[0].horizontal_dims if dim in anom.dims]
                        if not spatial_dims:
                            self.logger.warning(f"GridInspector did not return usable horizontal dims for {processed_key}. Falling back to non-time dims.")
                except Exception as exc:
                    self.logger.warning(f"Failed to infer spatial dims via GridInspector for {processed_key}: {exc}")

                # If no spatial dimensions are found, use all non-time dimensions
                if not spatial_dims:
                    spatial_dims = [dim for dim in anom.dims if dim.lower() != 'time']

                # Check if no spatial dimensions are found
                if not spatial_dims:
                    self.logger.error(f"Could not determine spatial dims for {processed_key}. Skipping.")
                    continue

                # Calculate temporal ACC time series
                temporal_acc_series = xr.corr(anom, anom_ref, dim=spatial_dims)
                temporal_acc_series.name = base_var_name

                # Plot the temporal ACC time series
                fig, ax = plt.subplots(figsize=(12, 6))
                temporal_acc_series.plot(ax=ax)
                time_coords = temporal_acc_series['time'].values
                if len(time_coords) > 1:
                    try:
                        inferred_freq = xr.infer_freq(temporal_acc_series.time)
                        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M') if inferred_freq and any(f in inferred_freq for f in ['H','T','S']) else mdates.DateFormatter('%Y-%m-%d')
                    except:
                        if np.issubdtype(time_coords.dtype, np.datetime64):
                            time_diff_seconds = (time_coords[1] - time_coords[0]) / np.timedelta64(1, 's')
                            date_format = mdates.DateFormatter('%Y-%m-%d %H:%M') if time_diff_seconds < (24*3600) else mdates.DateFormatter('%Y-%m-%d')
                        else: date_format = mdates.DateFormatter('%Y-%m-%d')
                    ax.xaxis.set_major_formatter(date_format)
                    fig.autofmt_xdate()
                elif len(time_coords) == 1: self.logger.debug(f"Only one time point for {processed_key}.")
                ax.set_ylabel("Spatial Pattern ACC")
                ax.set_xlabel("Time")
                ax.set_ylim(-1, 1)
                ax.grid(True)
                title_level_part = f" at {int(level)}" if level is not None else ""
                title = (f"Temporal Evolution of Spatial ACC: {base_var_name}{title_level_part}\n"
                         f"{model} {exp} ({source}) vs {model_ref} {exp_ref} ({source_ref})\n"
                         f"{self.startdate} to {self.enddate} (Clim: {self.clim_startdate}-{self.clim_enddate})")
                ax.set_title(title)
                fig.tight_layout()
                self.logger.info(f"Temporal ACC plot generated for {processed_key}")
                results[processed_key] = (fig, ax, xr.Dataset({base_var_name: temporal_acc_series}))

            except Exception as e:
                self.logger.error(f"Failed processing temporal ACC for key {processed_key}: {e}", exc_info=True)
                if processed_key in results: del results[processed_key]

        self.logger.info("Finished temporal ACC processing.")
        if save_fig:
             self.logger.info("Saving temporal ACC figures...")
             for key, res in results.items():
                 if res[0]: self._save_figure(res[0], key, 'temporal')
        if save_netcdf:
             self.logger.info("Saving temporal ACC netcdf files...")
             for key, res in results.items():
                 if res[2]: self._save_netcdf(res[2], key, 'temporal')

        return None
