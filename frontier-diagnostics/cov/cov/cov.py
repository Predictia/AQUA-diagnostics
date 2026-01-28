import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from aqua.core.logger import log_configure
from aqua.core.util import load_yaml
from aqua.core.reader import Reader
from aqua.core.graphics import plot_single_map
from smmregrid import GridInspector

class COV:
    """Covariance diagnostic"""

    def __init__(self, config, loglevel: str = 'WARNING'):
        """
        Initialize the COV diagnostic class.

        Args:
            config: Configuration for the COV diagnostic.
            loglevel (str): Logging level. Defaults to 'WARNING'.
        """
        # Configure logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'COV')

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

        if self.startdate is None or self.enddate is None:
            self.logger.error("Both startdate and enddate must be provided")
            raise ValueError("Both startdate and enddate must be provided")

        # Initialize the readers
        self._reader()

    def _reader(self):
        """Initializes the Reader class for both reference and main data."""
        # Get the data configurations
        config_data_ref = self.config.get('data_ref', {})
        config_data = self.config.get('data', {})

        # Initialize the readers
        self.reader_data_ref = Reader(catalog=config_data_ref.get('catalog'),
                                      model=config_data_ref.get('model'),
                                      exp=config_data_ref.get('exp'),
                                      source=config_data_ref.get('source'),
                                      regrid=config_data_ref.get('regrid'),
                                      fix=config_data_ref.get('fix'),
                                      startdate=self.startdate,
                                      enddate=self.enddate,
                                      loglevel=self.loglevel,
                                      preproc=COV._preprocess_coords_to_float32)
        self.reader_data = Reader(catalog=config_data.get('catalog'),
                                  model=config_data.get('model'),
                                  exp=config_data.get('exp'),
                                  source=config_data.get('source'),
                                  regrid=config_data.get('regrid'),
                                  fix=config_data.get('fix'),
                                  startdate=self.startdate,
                                  enddate=self.enddate,
                                  loglevel=self.loglevel,
                                  preproc=COV._preprocess_coords_to_float32)
        self.logger.debug('Reader classes initialized for data_ref and data')

    def retrieve(self):
        """
        Retrieves data for all unique variables specified in the configuration's variable_pairs.
        If regridding is configured, data will be automatically regridded after retrieval.
        """
        # Initialize the retrieved data
        self.retrieved_data_ref = {}
        self.retrieved_data = {}
        variable_pairs = self.config.get('variable_pairs', [])

        if not variable_pairs:
            self.logger.warning("No 'variable_pairs' section found in config. Nothing to retrieve.")
            return

        # Collect all unique variables to avoid retrieving the same data multiple times
        unique_vars = {}
        for pair in variable_pairs:
            for var_key in ['var1', 'var2']:
                var_info = pair.get(var_key)
                if not var_info or 'name' not in var_info:
                    continue
                
                level = var_info.get('level')
                data_key = f"{var_info['name']}_{int(level)}" if level is not None else var_info['name']
                
                if data_key not in unique_vars:
                    unique_vars[data_key] = var_info

        self.logger.info(f"Found unique variables to retrieve: {list(unique_vars.keys())}")

        # Retrieve the data
        for data_key, var_info in unique_vars.items():
            var_name = var_info['name']
            level = var_info.get('level')
            
            retrieve_args = {'var': var_name}
            if level is not None:
                retrieve_args['level'] = int(level)

            try:
                # Retrieve Reference Data
                self.logger.debug(f"Retrieving reference data for {data_key}")
                data_ref = self.reader_data_ref.retrieve(**retrieve_args)

                # TODO: Generalize this to handle other pressure levels naming conventions
                # This works now under the assumption that the pressure level is in the coordinates
                # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)
                if level is not None and 'plev' in data_ref.coords:
                    data_ref = data_ref.isel(plev=0, drop=True)

                # Apply regridding if configured
                # If regrid=False AQUA sets tgt_grid_name to None
                if self.reader_data_ref.tgt_grid_name is not None:
                    self.logger.debug(f"Applying regridding to reference data for {data_key}")
                    data_ref = self.reader_data_ref.regrid(data_ref)
                self.retrieved_data_ref[data_key] = data_ref

                # Retrieve Main Data
                self.logger.debug(f"Retrieving main data for {data_key}")
                data = self.reader_data.retrieve(**retrieve_args)

                # TODO: Generalize this to handle other pressure levels naming conventions
                # This works now under the assumption that the pressure level is in the coordinates
                # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)   
                if level is not None and 'plev' in data.coords:
                    data = data.isel(plev=0, drop=True)

                # Apply regridding if configured
                # If regrid=False AQUA sets tgt_grid_name to None
                if self.reader_data.tgt_grid_name is not None:
                    self.logger.debug(f"Applying regridding to main data for {data_key}")
                    data = self.reader_data.regrid(data)
                self.retrieved_data[data_key] = data

            except Exception as e:
                self.logger.error(f"Failed to retrieve data for {data_key}: {e}", exc_info=True)
                if data_key in self.retrieved_data_ref: del self.retrieved_data_ref[data_key]
                if data_key in self.retrieved_data: del self.retrieved_data[data_key]

        self.logger.info("Data retrieval finished.")
        self.logger.info(f"Retrieved reference data for keys: {list(self.retrieved_data_ref.keys())}")
        self.logger.info(f"Retrieved main data for keys: {list(self.retrieved_data.keys())}")

        # Log regridding information
        if self.reader_data_ref.tgt_grid_name is not None:
            self.logger.info(f"Reference data regridded to: {self.reader_data_ref.tgt_grid_name}")
        if self.reader_data.tgt_grid_name is not None:
            self.logger.info(f"Main data regridded to: {self.reader_data.tgt_grid_name}")

    # TODO: Move this to a shared utility module.
    @staticmethod
    def _preprocess_coords_to_float32(ds):
        """
        Converts 'lat' and 'lon' coordinates in an xarray.Dataset to float32
        if they exist and are currently float64.
        This function is intended to be passed to the 'preproc' argument 
        of the AQUA Reader.
        
        Args:
            ds (xr.Dataset): The Dataset to process.
            
        Returns:
            xr.Dataset: The Dataset with float32 lat/lon coordinates.
        """
        modified_coords = {}
        if 'lat' in ds.coords and ds['lat'].dtype == 'float64':
            modified_coords['lat'] = ds['lat'].astype('float32')
        
        if 'lon' in ds.coords and ds['lon'].dtype == 'float64':
            modified_coords['lon'] = ds['lon'].astype('float32')

        if modified_coords:
            ds = ds.assign_coords(**modified_coords)
        return ds

    @staticmethod
    def _calculate_correlation(da1, da2, reader):
        """
        Calculates Pearson correlation between two DataArrays.
        
        Args:
            da1 (xr.DataArray): First DataArray.
            da2 (xr.DataArray): Second DataArray.
            reader (Reader): AQUA Reader instance for time mean and std operations.
            
        Returns:
            xr.DataArray: Correlation coefficient between da1 and da2.
        """
        # Compute anomalies
        da1_anom = da1 - reader.timmean(da1)
        da2_anom = da2 - reader.timmean(da2)
        
        # Mean of product of anomalies
        numerator = reader.timmean(da1_anom * da2_anom)
        
        # Product of standard deviations
        denominator = reader.timstd(da1) * reader.timstd(da2)
        
        # Pearson correlation
        correlation = numerator / denominator
        
        return correlation

    def compute_and_plot(self, save_fig: bool = False, save_netcdf: bool = False):
        """
        Computes correlation for each variable pair and plots the results.

        Args:
            save_fig (bool): Whether to save the figures. Defaults to False.
            save_netcdf (bool): Whether to save the netcdf files. Defaults to False.
        """

        self.logger.info('Starting correlation computation and plotting.')

        # Get the variable pairs
        variable_pairs = self.config.get('variable_pairs', [])
        if not variable_pairs:
            self.logger.warning("No variable pairs to process.")
            return

        # Process each variable pair
        for pair in variable_pairs:
            var1_info = pair['var1']
            var2_info = pair['var2']

            var1_name, var1_level = var1_info['name'], var1_info.get('level')
            var2_name, var2_level = var2_info['name'], var2_info.get('level')

            key1 = f"{var1_name}_{int(var1_level)}" if var1_level is not None else var1_name
            key2 = f"{var2_name}_{int(var2_level)}" if var2_level is not None else var2_name
            
            pair_name = f"{key1}_vs_{key2}"
            self.logger.info(f"Processing correlation for pair: {pair_name}")

            # Ensure both variables for the pair were retrieved successfully
            if not all(k in self.retrieved_data for k in [key1, key2]) or \
               not all(k in self.retrieved_data_ref for k in [key1, key2]):
                self.logger.warning(f"Skipping pair {pair_name} due to missing data.")
                continue

            try:
                # Get data for the pair
                data_ref1 = self.retrieved_data_ref[key1][var1_name]
                data_ref2 = self.retrieved_data_ref[key2][var2_name]
                data1 = self.retrieved_data[key1][var1_name]
                data2 = self.retrieved_data[key2][var2_name]

                # Calculations
                self.logger.debug(f"Calculating correlation for reference: {pair_name}")
                corr_ref = COV._calculate_correlation(data_ref1, data_ref2, self.reader_data_ref)
                corr_ref.name = 'correlation'

                self.logger.debug(f"Calculating correlation for model: {pair_name}")
                corr_model = COV._calculate_correlation(data1, data2, self.reader_data)
                corr_model.name = 'correlation'

                self.logger.debug(f"Calculating difference for: {pair_name}")
                corr_diff = corr_model - corr_ref
                corr_diff.name = 'correlation_diff'

                # Plotting
                # Plot 1: Reference Correlation
                self.logger.debug(f"Plotting reference correlation for {pair_name}")
                fig_ref, _ = plot_single_map(corr_ref, title=f"Reference Correlation: {pair_name}", return_fig=True,
                                             vmin=-1.0, vmax=1.0, sym=True, cmap='RdBu_r')
                if save_fig:
                    self._save_figure(fig_ref, f"{pair_name}_ref", "correlation")

                # Plot 2: Model Correlation
                self.logger.debug(f"Plotting model correlation for {pair_name}")
                fig_model, _ = plot_single_map(corr_model, title=f"Model Correlation: {pair_name}", return_fig=True,
                                               vmin=-1.0, vmax=1.0, sym=True, cmap='RdBu_r')
                if save_fig:
                    self._save_figure(fig_model, f"{pair_name}_model", "correlation")

                # Plot 3: Difference
                self.logger.debug(f"Plotting difference for {pair_name}")
                vmax_diff = np.abs(corr_diff).max()
                fig_diff, _ = plot_single_map(corr_diff, title=f"Difference (Model - Ref): {pair_name}",
                                              return_fig=True, sym=True, cmap='RdBu_r',
                                              vmin=-vmax_diff, vmax=vmax_diff)
                if save_fig:
                    self._save_figure(fig_diff, f"{pair_name}_diff", "correlation")
                
                plt.close('all') # Close all figures to save memory

                if save_netcdf:
                    self._save_netcdf(corr_ref.to_dataset(), f"{pair_name}_ref", "correlation")
                    self._save_netcdf(corr_model.to_dataset(), f"{pair_name}_model", "correlation")
                    self._save_netcdf(corr_diff.to_dataset(), f"{pair_name}_diff", "correlation")

            except Exception as e:
                self.logger.error(f"Failed to process pair {pair_name}: {e}", exc_info=True)
        
        self.logger.info("Finished correlation processing.")

    # TODO: Move this to a shared utility module.
    def _sanitize_filename_part(self, part):
        """Sanitizes a string part for use in a filename."""
        if not isinstance(part, str): part = str(part)
        return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in part)

    # TODO: Move this to a shared utility module.
    def _generate_filename(self, pair_name, output_type, suffix):
        """Generates a unique filename."""
        data_cfg = self.config.get('data', {})
        model = self._sanitize_filename_part(data_cfg.get('model', 'model'))
        exp = self._sanitize_filename_part(data_cfg.get('exp', 'exp'))
        startdate = self._sanitize_filename_part(self.startdate)
        enddate = self._sanitize_filename_part(self.enddate)
        
        base_filename = f"{model}_{exp}_{pair_name}_{suffix}_{startdate}_{enddate}"

        if output_type == 'figure':
            output_dir = self.config.get('outputdir_fig', './figs_cov')
            extension = '.pdf'
        elif output_type == 'netcdf':
            output_dir = self.config.get('outputdir_data', './data_cov')
            extension = '.nc'
        else:
            self.logger.error(f"Unknown output_type: {output_type}")
            return f"./{base_filename}_unknown_type"
            
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base_filename}{extension}")

    # TODO: Move this to a shared utility module.
    def _save_figure(self, fig, pair_name, plot_type):
        """Saves the figure with a descriptive name."""
        filename = self._generate_filename(pair_name, 'figure', plot_type)
        try:
            fig.savefig(filename, bbox_inches='tight')
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

    # TODO: Move this to a shared utility module.
    def _save_netcdf(self, data, pair_name, data_type):
        """Saves the netcdf with a descriptive name."""
        filename = self._generate_filename(pair_name, 'netcdf', data_type)
        try:
            data.to_netcdf(filename)
            self.logger.info(f"NetCDF data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save NetCDF {filename}: {e}", exc_info=True)