import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from aqua.core.logger import log_configure
from aqua.core.util import load_yaml, coord_names, cbar_get_label
from aqua.core.reader import Reader

class Variability:
    """Variability diagnostic that computes standard deviation of target and emulator."""

    def __init__(self, config, loglevel: str = 'WARNING'):
        """Initialize the Variability diagnostic class.

        Args:
            config: Configuration for the Variability diagnostic.
            loglevel (str): Logging level. Defaults to 'WARNING'.

        Raises:
            ValueError: If required configuration sections or keys are missing.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'Variability')

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
        self.outputdir_fig = self.config.get('outputdir_fig', './figs_variability')
        self.figure_format = self._normalise_extension(self.config.get('figure_format', 'pdf'))
        os.makedirs(self.outputdir_fig, exist_ok=True)

        # Load plot settings
        self.plot_settings = self._load_plot_settings()

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
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel)

        self.reader_data = Reader(catalog=config_data.get('catalog'),
                                  model=config_data.get('model'),
                                  exp=config_data.get('exp'),
                                  source=config_data.get('source'),
                                  regrid=config_data.get('regrid'),
                                  fix=config_data.get('fix'),
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel)

        self.logger.debug('Reader classes initialized for data_ref and data')

    def _load_plot_settings(self):
        """Load plot settings from configuration"""
        plot_cfg = self.config.get('plot', {})
        
        # Default settings
        settings = {
            'std': {
                'cmap': 'viridis',
                'vmin': None,
                'vmax': None,
                'discrete_colorbar': False,
                'n_colorbar_levels': 10
            },
            'diff': {
                'cmap': 'RdBu_r',
                'vmin': None,
                'vmax': None,
                'discrete_colorbar': False,
                'n_colorbar_levels': 10
            }
        }
        
        # Update with user-provided settings for std plots
        std_cfg = plot_cfg.get('std', {})
        if 'cmap' in std_cfg:
            settings['std']['cmap'] = str(std_cfg['cmap'])
        if 'vmin' in std_cfg and std_cfg['vmin'] is not None:
            settings['std']['vmin'] = float(std_cfg['vmin'])
        if 'vmax' in std_cfg and std_cfg['vmax'] is not None:
            settings['std']['vmax'] = float(std_cfg['vmax'])
        if 'discrete_colorbar' in std_cfg:
            settings['std']['discrete_colorbar'] = bool(std_cfg['discrete_colorbar'])
        if 'n_colorbar_levels' in std_cfg:
            settings['std']['n_colorbar_levels'] = int(std_cfg['n_colorbar_levels'])
        
        # Update with user-provided settings for diff plots
        diff_cfg = plot_cfg.get('diff', {})
        if 'cmap' in diff_cfg:
            settings['diff']['cmap'] = str(diff_cfg['cmap'])
        if 'vmin' in diff_cfg and diff_cfg['vmin'] is not None:
            settings['diff']['vmin'] = float(diff_cfg['vmin'])
        if 'vmax' in diff_cfg and diff_cfg['vmax'] is not None:
            settings['diff']['vmax'] = float(diff_cfg['vmax'])
        if 'discrete_colorbar' in diff_cfg:
            settings['diff']['discrete_colorbar'] = bool(diff_cfg['discrete_colorbar'])
        if 'n_colorbar_levels' in diff_cfg:
            settings['diff']['n_colorbar_levels'] = int(diff_cfg['n_colorbar_levels'])
        
        self.logger.debug(f"Plot settings loaded: {settings}")
        return settings

    def retrieve(self):
        """
        Retrieves data for all variables specified in the configuration
        from both the reference and main data sources using the initialized Reader classes.
        Stores retrieved data in self.retrieved_data_ref and self.retrieved_data dictionaries.
        Handles single levels, lists of levels, and surface variables.
        
        If regridding is configured in the Reader instances (via the 'regrid' parameter),
        the data will be automatically regridded to the target grid after retrieval.
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

            levels = var_info.get('level') # Can be None, int, or list

            # Ensure levels is a list to simplify iteration, handle None case
            if levels is None:
                levels_to_iterate = [None] # Treat surface variables as a list with one None element
            elif isinstance(levels, (int, float)):
                levels_to_iterate = [int(levels)] # Single level
            elif isinstance(levels, list):
                levels_to_iterate = [int(lvl) for lvl in levels] # List of levels
            else:
                self.logger.warning(f"Invalid level format for variable {var_name}: {levels}. Skipping.")
                continue

            for level in levels_to_iterate:
                # Determine the key (e.g., 'q_85000' or '2t')
                data_key = f"{var_name}_{level}" if level is not None else var_name
                log_msg_suffix = f" at level {level}" if level is not None else " (surface)"

                # Arguments for Reader.retrieve
                retrieve_args = {'var': var_name}
                if level is not None:
                    retrieve_args['level'] = level

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
                    self.logger.debug(f"Successfully retrieved reference data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving reference {var_name}{log_msg_suffix}: {e}", exc_info=True)
                    continue # Skip this level/variable on error

                # Retrieve Main Data
                try:
                    self.logger.debug(f"Retrieving main data for {var_name}{log_msg_suffix}")
                    data = self.reader_data.retrieve(**retrieve_args)
                    
                    if level is not None and 'plev' in data.coords:
                        data = data.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if self.reader_data.tgt_grid_name is not None:
                        self.logger.debug(f"Applying regridding to main data for {var_name}{log_msg_suffix}")
                        data = self.reader_data.regrid(data)

                    self.retrieved_data[data_key] = data
                    self.logger.debug(f"Successfully retrieved main data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred retrieving main {var_name}{log_msg_suffix}: {e}", exc_info=True)
                     # Clean up ref data if main data failed for this level
                    if data_key in self.retrieved_data_ref:
                        del self.retrieved_data_ref[data_key]
                    continue # Skip this level/variable on error

        self.logger.info("Data retrieval finished.")
        if not self.retrieved_data_ref and not self.retrieved_data:
             self.logger.warning("No data was successfully retrieved for any variable.")
        else:
             # Log the keys
             self.logger.info(f"Retrieved reference data for keys: {list(self.retrieved_data_ref.keys())}")
             self.logger.info(f"Retrieved main data for keys: {list(self.retrieved_data.keys())}")
             
             # Log regridding information
             if self.reader_data_ref.tgt_grid_name is not None:
                 self.logger.info(f"Reference data regridded to: {self.reader_data_ref.tgt_grid_name}")
             if self.reader_data.tgt_grid_name is not None:
                 self.logger.info(f"Main data regridded to: {self.reader_data.tgt_grid_name}")

    @staticmethod
    def check_and_convert_coords(data, data_ref, base_var_name):
        """
        Checks if latitude and longitude coordinates are of the same type between datasets.
        Converts them to float32 if they differ.

        Args:
            data (xr.Dataset): First dataset to check
            data_ref (xr.Dataset): Reference dataset to check
            base_var_name (str): The variable name within the datasets (e.g., 'q', '2t')

        Returns:
            tuple: Processed datasets with matching coordinate types
        """
        # Check if the base variable exists in both datasets before proceeding
        if base_var_name not in data or base_var_name not in data_ref:
             print(f"Warning: Base variable '{base_var_name}' not found in one or both datasets during coordinate check.")
             return data, data_ref # Return original data if var not found

        # Get coordinate names for lat and lon using the base variable name
        # TODO: Generalize this to handle other coordinate names
        try:
            lat_name = [dim for dim in data[base_var_name].dims if 'lat' in dim.lower()][0]
            lon_name = [dim for dim in data[base_var_name].dims if 'lon' in dim.lower()][0]
        except IndexError:
             print(f"Warning: Could not determine lat/lon dimension names for variable '{base_var_name}'.")
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
    def _sanitize_filename_part(part):
        """Sanitizes a string part for use in a filename."""
        if not isinstance(part, str):
            part = str(part)

        # Replace potentially problematic characters with underscores
        invalid_chars = [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.'] 
        for char in invalid_chars:
            part = part.replace(char, '_')
        return part

    def _generate_filename(self, processed_key, suffix):
        """
        Generates a unique filename based on config and context.

        Args:
            processed_key (str): The variable key (e.g., 'q_85000', '2t').
            suffix (str): Suffix describing the calculation (e.g., 'variability').

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

        # Construct the base filename from sanitized parts
        base_name_parts = [model, exp, source,
                           'vs', model_ref, exp_ref,
                           processed_key,
                           startdate, enddate,
                           suffix]

        # Filter out any empty strings that might result from missing config values
        base_filename = '_'.join(filter(None, base_name_parts))

        # Determine extension
        extension = f'.{self.figure_format}'

        full_path = os.path.join(self.outputdir_fig, f"{base_filename}{extension}")
        self.logger.debug(f"Generated path for figure ({processed_key}, {suffix}): {full_path}")
        return full_path

    def _save_figure(self, fig, processed_key):
        """
        Saves the generated figure with a unique name based on config.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            processed_key (str): The variable key (e.g., 'q_85000', '2t').
        """
        filename = self._generate_filename(processed_key, 'variability')

        try:
            fig.savefig(filename)
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

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

    @staticmethod
    def _normalise_extension(extension):
        """Normalise the extension"""
        ext = extension.lower().lstrip(".")
        if not ext:
            return "pdf"
        else:
            return ext

    def compute_variability(self, save_fig: bool = False):
        """
        Calculates and plots the standard deviation for all variables and levels retrieved.

        This method processes all common variables between the main and reference datasets,
        calculating the temporal standard deviation spatially for each grid point.

        Args:
            save_fig (bool, optional): If True, saves the generated figures to the configured
                output directory. Defaults to False.

        Note:
            - The retrieve() method must be called before this method.
            - Only variables/levels present in both datasets will be processed.
        """
        self.logger.info('Calculating and plotting variability for retrieved variables/levels.')

        # Ensure data has been retrieved
        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
             self.logger.error("Data has not been retrieved yet. Call the retrieve() method first.")
             return {}

        # Find common keys (e.g., 'q_85000', '2t') retrieved in both datasets
        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys())
        if not common_keys:
            self.logger.warning("No common data keys found between reference and main retrieved data. Cannot calculate variability.")
            return {}

        results = {}

        # Get metadata for title from config
        config_data = self.config.get('data', {})
        config_data_ref = self.config.get('data_ref', {})
        model = config_data.get('model', 'N/A')
        exp = config_data.get('exp', 'N/A')
        source = config_data.get('source', 'N/A')
        model_ref = config_data_ref.get('model', 'N/A')
        exp_ref = config_data_ref.get('exp', 'N/A')
        source_ref = config_data_ref.get('source', 'N/A')

        for processed_key in common_keys:
            base_var_name, level = self._parse_processed_key(processed_key)
            self.logger.info(f"Processing variability for key: {processed_key} (Variable: {base_var_name}, Level: {level})")

            data = self.retrieved_data[processed_key]
            data_ref = self.retrieved_data_ref[processed_key]

            # Check if base_var_name actually exists in the datasets after retrieval
            if base_var_name not in data or base_var_name not in data_ref:
                self.logger.warning(f"Variable '{base_var_name}' not found in data for key '{processed_key}'. Skipping.")
                continue

            try:
                # Coordinate Check/Conversion - Pass base_var_name
                data, data_ref = self.check_and_convert_coords(data, data_ref, base_var_name=base_var_name)
                self.logger.debug(f"Coordinates checked/converted for {processed_key}")

                # Calculate standard deviation over time
                std_data = data[base_var_name].std(dim='time')
                std_data_ref = data_ref[base_var_name].std(dim='time')
                
                # Calculate difference
                std_diff = std_data - std_data_ref
                
                self.logger.debug(f"Standard deviation calculated for {processed_key}")

                # Plotting - create 3 subplots
                fig = self._plot_variability(std_data_ref, std_data, std_diff,
                                             base_var_name, level,
                                             model, exp, source,
                                             model_ref, exp_ref, source_ref)

                self.logger.info(f"Variability plot generated for {processed_key}")

                # Store results using the processed key
                results[processed_key] = fig

                if save_fig:
                    self._save_figure(fig, processed_key)
                    plt.close(fig)

            except Exception as e:
                self.logger.error(f"Failed to calculate or plot variability for key {processed_key}: {e}", exc_info=True)
                if processed_key in results: 
                    del results[processed_key] # Clear potentially incomplete results for this key

        self.logger.info("Finished variability processing.")

        return results

    def _plot_variability(self, std_ref, std_data, std_diff, 
                         base_var_name, level,
                         model, exp, source,
                         model_ref, exp_ref, source_ref):
        """
        Create a figure with three subplots showing reference std, data std, and their difference.

        Args:
            std_ref: Standard deviation of reference data
            std_data: Standard deviation of main data
            std_diff: Difference between std_data and std_ref
            base_var_name: Variable name
            level: Pressure level (or None)
            model, exp, source: Main data metadata
            model_ref, exp_ref, source_ref: Reference data metadata

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get coordinate names
        lon_name, lat_name = coord_names(std_ref)
        if not lon_name or not lat_name:
            raise ValueError("Unable to determine lat/lon coordinates for plotting.")

        # Prepare data for plotting
        lon = std_ref[lon_name].values
        lat = std_ref[lat_name].values
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Get plot settings
        std_settings = self.plot_settings['std']
        diff_settings = self.plot_settings['diff']

        # Determine color limits for std plots (non-negative)
        if std_settings['vmin'] is not None:
            vmin_std = std_settings['vmin']
        else:
            vmin_std = 0
        
        if std_settings['vmax'] is not None:
            vmax_std = std_settings['vmax']
        else:
            valid_values = np.concatenate([std_ref.values[np.isfinite(std_ref.values)],
                                           std_data.values[np.isfinite(std_data.values)]])
            if len(valid_values) > 0:
                vmax_std = float(np.percentile(valid_values, 98))
            else:
                vmax_std = 1.0

        # Determine color limits for difference plot (symmetric)
        if diff_settings['vmin'] is not None and diff_settings['vmax'] is not None:
            vmin_diff = diff_settings['vmin']
            vmax_diff = diff_settings['vmax']
        else:
            valid_diff = std_diff.values[np.isfinite(std_diff.values)]
            if len(valid_diff) > 0:
                abs_max = float(np.percentile(np.abs(valid_diff), 98))
                vmin_diff = -abs_max
                vmax_diff = abs_max
            else:
                vmin_diff = -0.1
                vmax_diff = 0.1

        # Create normalization for discrete colorbars if requested
        if std_settings['discrete_colorbar']:
            levels_std = np.linspace(vmin_std, vmax_std, std_settings['n_colorbar_levels'])
            norm_std = mcolors.BoundaryNorm(levels_std, ncolors=256)
        else:
            norm_std = None

        if diff_settings['discrete_colorbar']:
            levels_diff = np.linspace(vmin_diff, vmax_diff, diff_settings['n_colorbar_levels'])
            norm_diff = mcolors.BoundaryNorm(levels_diff, ncolors=256)
        else:
            norm_diff = None

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot 1: Reference standard deviation
        pcolormesh_kwargs_std = {
            'transform': ccrs.PlateCarree(),
            'cmap': std_settings['cmap'],
            'shading': 'auto'
        }
        if norm_std is not None:
            pcolormesh_kwargs_std['norm'] = norm_std
        else:
            pcolormesh_kwargs_std['vmin'] = vmin_std
            pcolormesh_kwargs_std['vmax'] = vmax_std

        mesh1 = axes[0].pcolormesh(lon2d, lat2d, std_ref.values, **pcolormesh_kwargs_std)
        axes[0].coastlines()
        axes[0].set_title(f'Reference ({model_ref} {exp_ref})')
        cbar1 = plt.colorbar(mesh1, ax=axes[0], orientation='horizontal', pad=0.05)
        cbar_label = cbar_get_label(std_ref)
        if cbar_label:
            cbar1.set_label(cbar_label)

        # Plot 2: Main data standard deviation
        mesh2 = axes[1].pcolormesh(lon2d, lat2d, std_data.values, **pcolormesh_kwargs_std)
        axes[1].coastlines()
        axes[1].set_title(f'Experiment ({model} {exp})')
        cbar2 = plt.colorbar(mesh2, ax=axes[1], orientation='horizontal', pad=0.05)
        if cbar_label:
            cbar2.set_label(cbar_label)

        # Plot 3: Difference (Experiment - Reference)
        pcolormesh_kwargs_diff = {
            'transform': ccrs.PlateCarree(),
            'cmap': diff_settings['cmap'],
            'shading': 'auto'
        }
        if norm_diff is not None:
            pcolormesh_kwargs_diff['norm'] = norm_diff
        else:
            pcolormesh_kwargs_diff['vmin'] = vmin_diff
            pcolormesh_kwargs_diff['vmax'] = vmax_diff

        mesh3 = axes[2].pcolormesh(lon2d, lat2d, std_diff.values, **pcolormesh_kwargs_diff)
        axes[2].coastlines()
        axes[2].set_title('Difference (Experiment - Reference)')
        cbar3 = plt.colorbar(mesh3, ax=axes[2], orientation='horizontal', pad=0.05)
        if cbar_label:
            cbar3.set_label(cbar_label)

        # Set overall title
        level_part = f" at level {level}" if level is not None else ""
        fig.suptitle(f"Temporal Standard Deviation: {base_var_name}{level_part}\n"
                    f"{self.startdate} to {self.enddate}",
                    fontsize=14, y=1.02)

        fig.tight_layout()
        return fig