import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

try:
    import healpy as hp
except ImportError:
    hp = None

from aqua.core.logger import log_configure
from aqua.core.util import load_yaml
from aqua.core.reader import Reader

# TODO: Move some methods to a shared module.

class PSD:
    """Power Spectral Density (PSD) diagnostic
    
    Supports both regular lat-lon grids (using FFT) and HEALPix grids (using spherical harmonics).
    Automatically detects the grid type and uses the appropriate method.
    """

    def __init__(self, config, loglevel: str = 'WARNING'):
        """
        Initialize the PSD diagnostic class.

        Args:
            config: Configuration for the PSD diagnostic.
            loglevel (str): Logging level. Defaults to 'WARNING'.

        Raises:
            ValueError: If required configuration sections or keys are missing.
        """
        # Configure the logger
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, 'PSD')

        # Load the configuration
        if isinstance(config, str):
            self.logger.debug("Reading configuration file %s", config)
            self.config = load_yaml(config)
        else:
            self.logger.debug("Configuration is a dictionary")
            self.config = config

        # Get the start and end dates
        self.startdate = self.config.get('dates', {}).get('startdate')
        self.enddate = self.config.get('dates', {}).get('enddate')

        # Check if the start and end dates are provided
        if self.startdate is None or self.enddate is None:
            self.logger.error("Both startdate and enddate must be provided")
            raise ValueError("Both startdate and enddate must be provided")

        # Initialize the readers
        self._reader()

    def _reader(self):
        """Initializes the Reader class for reference and main data."""
        # Get the data configuration
        config_data_ref = self.config.get('data_ref', {})
        config_data = self.config.get('data', {})

        # Initialize the readers
        self.reader_data_ref = Reader(**config_data_ref,
            startdate=self.startdate, enddate=self.enddate, loglevel=self.loglevel)
        self.reader_data = Reader(**config_data,
            startdate=self.startdate, enddate=self.enddate, loglevel=self.loglevel)

        self.logger.debug('Reader classes initialized for data_ref and data')

    def retrieve(self):
        """Retrieve data for all variables specified in the configuration."""
        # Initialize the retrieved data
        self.retrieved_data_ref = {}
        self.retrieved_data = {}

        # Get the variables to process
        variables_to_process = self.config.get('variables', [])

        # Start the data retrieval
        self.logger.info("Starting data retrieval...")
        for var_info in variables_to_process:
            var_name = var_info.get('name')
            levels = var_info.get('level')
            levels_to_iterate = [None] if levels is None else np.atleast_1d(levels).astype(int)

            # Iterate over the levels
            for level in levels_to_iterate:
                data_key = f"{var_name}_{level}" if level is not None else var_name
                log_msg_suffix = f" at level {level}" if level is not None else " (surface)"
                retrieve_args = {'var': var_name}
                if level is not None:
                    retrieve_args['level'] = level

                try:
                    # Retrieve reference data
                    self.logger.debug(f"Retrieving reference data for {var_name}{log_msg_suffix}")
                    data_ref = self.reader_data_ref.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)                    
                    if level is not None and 'plev' in data_ref.coords:
                        data_ref = data_ref.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if getattr(self.reader_data_ref, 'tgt_grid_name', None):
                        self.logger.debug(f"Regridding reference data for {var_name}{log_msg_suffix} to {self.reader_data_ref.tgt_grid_name}")
                        data_ref = self.reader_data_ref.regrid(data_ref)
                    self.retrieved_data_ref[data_key] = data_ref

                    # Retrieve main data
                    self.logger.debug(f"Retrieving main data for {var_name}{log_msg_suffix}")
                    data = self.reader_data.retrieve(**retrieve_args)

                    # TODO: Generalize this to handle other pressure levels naming conventions
                    # This works now under the assumption that the pressure level is in the coordinates
                    # (Now I fullfill this assumption by mapping the level to plev in the fixer if necessary)                    
                    if level is not None and 'plev' in data.coords:
                        data = data.isel(plev=0, drop=True)

                    # Apply regridding if configured
                    # If regrid=False AQUA sets tgt_grid_name to None
                    if getattr(self.reader_data, 'tgt_grid_name', None):
                        self.logger.debug(f"Regridding main data for {var_name}{log_msg_suffix} to {self.reader_data.tgt_grid_name}")
                        data = self.reader_data.regrid(data)
                    self.retrieved_data[data_key] = data

                    # Data retrieved successfully
                    self.logger.debug(f"Successfully retrieved data for key: {data_key}")
                except Exception as e:
                    self.logger.error(f"Failed to retrieve data for {data_key}: {e}", exc_info=True)
                    if data_key in self.retrieved_data_ref: del self.retrieved_data_ref[data_key]
                    if data_key in self.retrieved_data: del self.retrieved_data[data_key]

        self.logger.info("Data retrieval finished.")
        self.logger.info(f"Retrieved reference data for keys: {list(self.retrieved_data_ref.keys())}")
        self.logger.info(f"Retrieved main data for keys: {list(self.retrieved_data.keys())}")

    @staticmethod
    def _radial_average(power_2d):
        """Calculate the radial average of a 2D power spectrum."""
        y_center, x_center = np.array(power_2d.shape) // 2
        y, x = np.indices(power_2d.shape)
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        r = r.astype(np.int32)

        # Calculate the radial average
        tbin = np.bincount(r.ravel(), power_2d.ravel())
        nr = np.bincount(r.ravel())
        
        # Avoid division by zero
        radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)

        return radial_profile

    def _get_healpix_dim(self, data_da):
        """Get the HEALPix dimension from a DataArray."""
        # Candidate dimensions
        candidate_dims = ("ncells", "cell")

        # Iterate over the candidate dimensions
        for dim in candidate_dims:
            if dim in data_da.dims:
                # If the dimension is a valid HEALPix dimension, return the dimension
                if hp.isnpixok(data_da.sizes[dim]):
                    return dim

        # If healpy is not installed, return None
        if hp is None:
            return None
        for dim in data_da.dims:
            if hp.isnpixok(data_da.sizes[dim]):
                return dim
        return None

    def _detect_healpix_order(self, data_da, healpix_dim=None, reader=None):
        """Helper to read ordering metadata (ring or nested)."""
        # Helper to parse the order
        def parse_order(value):
            if not value:
                return None
            value = str(value).lower()
            if "nest" in value:
                return "nested"
            if "ring" in value:
                return "ring"
            return None

        sources = [getattr(data_da, "attrs", {}), getattr(data_da, "encoding", {})]
        if healpix_dim and healpix_dim in data_da.coords:
            sources.append(getattr(data_da.coords[healpix_dim], "attrs", {}))

        # Check the attributes for the order
        for attrs in sources:
            for key in ("healpix_order", "healpix_ordering", "ordering", "order", "nest"):
                order = parse_order(attrs.get(key))
                if order:
                    return order

        # Check the grid attribute for the order
        order = parse_order(data_da.attrs.get("grid"))
        if order:
            return order

        if reader is not None:
            for attr in ("src_grid_name", "tgt_grid_name"):
                order = parse_order(getattr(reader, attr, None))
                if order:
                    return order

        return None

    def _is_healpix_data(self, data_da):
        """Check if data is on a HEALPix grid."""
        if self._get_healpix_dim(data_da):
            return True

        grid_attr = str(data_da.attrs.get("grid", "")).lower()
        if "healpix" in grid_attr:
            return True

        grid_attr = str(data_da.attrs.get("AQUA_grid", "")).lower()
        if "healpix" in grid_attr:
            return True

        for reader in (getattr(self, "reader_data", None), getattr(self, "reader_data_ref", None)):
            grid_name = getattr(reader, "src_grid_name", None)
            if grid_name and "hp" in grid_name.lower():
                return True

        return False

    def _prepare_healpix_maps(self, data_da, reader=None, replace_nans=False):
        """Prepare HEALPix maps for spherical harmonic analysis. Healpy requires
        the data to be in the ring ordering."""
        # Check for healpy
        if hp is None:
            raise ImportError("healpy is required to process HEALPix grids.")

        # Check if the HEALPix dimension is found
        healpix_dim = self._get_healpix_dim(data_da)
        if healpix_dim is None:
            raise ValueError("Could not identify the HEALPix dimension.")

        # Reorder the dimensions
        order = [dim for dim in data_da.dims if dim != healpix_dim] + [healpix_dim]
        arr = data_da.transpose(*order).values

        # Properly shape the array
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        if replace_nans:
            arr = np.nan_to_num(arr, nan=0.0)

        healpix_order = self._detect_healpix_order(data_da, healpix_dim=healpix_dim, reader=reader)
        if healpix_order == "nested":
            self.logger.debug("Detected nested HEALPix ordering; converting to ring.")
            arr = np.asarray([hp.reorder(m, n2r=True) for m in arr])
        elif healpix_order is None:
            self.logger.debug("Could not determine HEALPix ordering; assuming ring.")
        elif healpix_order != "ring":
            self.logger.warning("Unknown HEALPix ordering '%s'; assuming ring ordering.", healpix_order)

        # Get the nside based on the number of pixels
        nside = hp.npix2nside(arr.shape[-1])

        return arr, nside

    def _parse_processed_key(self, processed_key):
        """Parse the processed key to get the base variable name and level."""
        parts = processed_key.split('_')
        level = None
        base_var_name = processed_key
        if len(parts) > 1 and parts[-1].isdigit():
            level = int(parts[-1])
            base_var_name = '_'.join(parts[:-1])
        return base_var_name, level

    def _sanitize_filename_part(self, part):
        """Sanitize the filename part."""
        if not isinstance(part, str): part = str(part)
        return "".join(c if c.isalnum() else "_" for c in part)

    def _generate_filename(self, processed_key, output_type, calculation_type):
        """Generate the filename for the output."""
        # TODO: Naming format could be improved
        data_cfg = self.config.get('data', {})
        ref_cfg = self.config.get('data_ref', {})
        dates_cfg = self.config.get('dates', {})
        
        parts = [data_cfg.get('model'), data_cfg.get('exp'), data_cfg.get('source'), 'vs',
                 ref_cfg.get('model'), dates_cfg.get('startdate'), dates_cfg.get('enddate'),
                 processed_key, calculation_type]
        base_filename = '_'.join(self._sanitize_filename_part(p) for p in parts if p)
        
        if output_type == 'figure':
            output_dir = self.config.get('outputdir_fig', './figs_psd')
            ext = '.pdf'
        elif output_type == 'netcdf':
            output_dir = self.config.get('outputdir_data', './data_psd')
            ext = '.nc'
        
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base_filename}{ext}")

    def _save_figure(self, fig, processed_key, calculation_type):
        """Save the figure."""
        filename = self._generate_filename(processed_key, 'figure', calculation_type)
        try:
            fig.savefig(filename)
            self.logger.info(f"Figure saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}", exc_info=True)

    def _save_netcdf(self, data, processed_key, calculation_type):
        """Save the netcdf data."""
        filename = self._generate_filename(processed_key, 'netcdf', calculation_type)
        try:
            data.to_netcdf(filename)
            self.logger.info(f"NetCDF data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save NetCDF {filename}: {e}", exc_info=True)

    # TODO: This method is too long and should be split into smaller methods.
    def calculate_and_plot_psd(self, save_fig: bool = False, save_netcdf: bool = False):
        """
        Calculate and plot the PSD for all common variables between reference and target.

        For each common key, this method computes (if enabled in the config):
        - The time-averaged PSD
        - The PSD of the time-mean field 

        It automatically detects whether the data is on a regular lat–lon grid (FFT-based PSD)
        or a HEALPix grid (spherical harmonics via healpy), handles NaNs, generates comparison
        plots between target and reference, and optionally saves figures and NetCDF files
        with the 1D spectra.
        """
        # Check if the data has been retrieved
        self.logger.info('Calculating and plotting PSD.')
        if not hasattr(self, 'retrieved_data') or not hasattr(self, 'retrieved_data_ref'):
             self.logger.error("Data has not been retrieved. Call retrieve() first.")
             return {}
        
        # Get the common keys
        common_keys = set(self.retrieved_data.keys()) & set(self.retrieved_data_ref.keys())
        if not common_keys:
            self.logger.warning("No common data keys found. Cannot calculate PSD.")
            return {}

        # Get the PSD calculations to perform
        psd_calcs = self.config.get('psd_calculations', {})
        do_time_averaged_psd = psd_calcs.get('time_averaged_psd', True)
        do_psd_of_time_mean = psd_calcs.get('psd_of_time_mean', True)

        if not do_time_averaged_psd and not do_psd_of_time_mean:
            self.logger.warning("No PSD calculations enabled in config. Skipping.")
            return {}

        # Initialize the results
        results = {}
        if do_time_averaged_psd: results['time_averaged_psd'] = {}
        if do_psd_of_time_mean: results['psd_of_time_mean'] = {}
        
        # Get the data configuration
        data_cfg = self.config.get('data', {})
        ref_cfg = self.config.get('data_ref', {})

        # Iterate over the common keys
        for key in common_keys:
            base_var, level = self._parse_processed_key(key)
            self.logger.info(f"Processing PSD for key: {key}")
            data_da = self.retrieved_data[key][base_var]
            data_ref_da = self.retrieved_data_ref[key][base_var]

            # Detect HEALPix grid
            is_healpix = self._is_healpix_data(data_da)
            if is_healpix and not self._is_healpix_data(data_ref_da):
                self.logger.warning("Reference data is not on the same HEALPix grid, falling back to FFT PSD.")
                is_healpix = False
            if is_healpix:
                self.logger.info("Detected HEALPix grid, using spherical harmonics.")

            # Check for NaNs in the data
            has_nans = np.isnan(data_da.values).any()
            if has_nans:
                self.logger.warning(f"Emulator data for {key} contains NaNs. Replacing them with 0.")

            has_nans_ref = np.isnan(data_ref_da.values).any()
            if has_nans_ref:
                self.logger.warning(f"Reference data for {key} contains NaNs. Replacing them with 0.")

            # Time-Averaged PSD
            if do_time_averaged_psd:
                try:
                    self.logger.info(f"Calculating time-averaged PSD for {key}...")
                    
                    if is_healpix:
                        if hp is None:
                            raise ImportError("healpy is required to process HEALPix grids.")

                        # Prepare the HEALPix maps
                        maps_target, nside_target = self._prepare_healpix_maps(
                            data_da, reader=self.reader_data, replace_nans=has_nans)
                        maps_ref, nside_ref = self._prepare_healpix_maps(
                            data_ref_da, reader=self.reader_data_ref, replace_nans=has_nans_ref)

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean from each map before hp.anafast
                        # to focus on spatial variability
                        lmax = min(3 * nside_target - 1, 3 * nside_ref - 1)
                        psd_target_stack = [hp.anafast(m, lmax=lmax) for m in maps_target]
                        psd_ref_stack = [hp.anafast(m, lmax=lmax) for m in maps_ref]

                        # Calculate the average PSD
                        avg_psd_1d = np.mean(psd_target_stack, axis=0)
                        avg_psd_1d_ref = np.mean(psd_ref_stack, axis=0)

                        # Plot the PSD
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ell = np.arange(len(avg_psd_1d))
                        ell_ref = np.arange(len(avg_psd_1d_ref))
                        ax.loglog(ell, avg_psd_1d, label=f"{data_cfg.get('model', 'Target')}")
                        ax.loglog(ell_ref, avg_psd_1d_ref, label=f"{ref_cfg.get('model', 'Reference')}", linestyle='--')
                        ax.set_xlabel("Spherical harmonic degree (l)")
                        ax.set_ylabel("Power Spectral Density")
                        ax.set_title(f"Time-Averaged Power Spectrum: {base_var}" + (f" at level {level}" if level else ""))
                        ax.legend()
                        ax.grid(True, which="both", ls="-", alpha=0.5)
                        fig.tight_layout()

                        # Create the xarray dataset for the results
                        max_len = max(len(avg_psd_1d), len(avg_psd_1d_ref))
                        wavenumbers = np.arange(max_len)
                        psd_ds = xr.Dataset({
                            'psd_target': (('wavenumber'), np.pad(avg_psd_1d, (0, max_len - len(avg_psd_1d)), 'constant', constant_values=np.nan)),
                            'psd_reference': (('wavenumber'), np.pad(avg_psd_1d_ref, (0, max_len - len(avg_psd_1d_ref)), 'constant', constant_values=np.nan)),
                        }, coords={'wavenumber': wavenumbers})
                        results['time_averaged_psd'][key] = (fig, ax, psd_ds)
                    else:
                        # Remove nans
                        data_np = data_da.values
                        data_np = np.nan_to_num(data_np, nan=0.0) if has_nans else data_np

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean before FFT to focus on spatial variability
                        fft_2d_stack = fft.fft2(data_np, axes=(-2, -1))
                        shifted_fft_stack = fft.fftshift(fft_2d_stack, axes=(-2, -1))
                        power_2d_stack = np.abs(shifted_fft_stack)**2
                        psd_1d_list = [self._radial_average(power_2d) for power_2d in power_2d_stack]
                        avg_psd_1d = np.mean(psd_1d_list, axis=0)

                        # Remove nans
                        data_ref_np = data_ref_da.values
                        data_ref_np = np.nan_to_num(data_ref_np, nan=0.0) if has_nans_ref else data_ref_np

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean before FFT to focus on spatial variability
                        fft_2d_stack_ref = fft.fft2(data_ref_np, axes=(-2, -1))
                        shifted_fft_stack_ref = fft.fftshift(fft_2d_stack_ref, axes=(-2, -1))
                        power_2d_stack_ref = np.abs(shifted_fft_stack_ref)**2
                        psd_1d_ref_list = [self._radial_average(power_2d) for power_2d in power_2d_stack_ref]
                        avg_psd_1d_ref = np.mean(psd_1d_ref_list, axis=0)

                        # Plot the PSD
                        fig, ax = plt.subplots(figsize=(10, 6))
                        wavenumber = np.arange(len(avg_psd_1d))
                        wavenumber_ref = np.arange(len(avg_psd_1d_ref))
                        ax.loglog(wavenumber, avg_psd_1d, label=f"{data_cfg.get('model', 'Target')}")
                        ax.loglog(wavenumber_ref, avg_psd_1d_ref, label=f"{ref_cfg.get('model', 'Reference')}", linestyle='--')
                        ax.set_xlabel("Wavenumber")
                        ax.set_ylabel("Power Spectral Density")
                        ax.set_title(f"Time-Averaged Power Spectrum: {base_var}" + (f" at level {level}" if level else ""))
                        ax.legend()
                        ax.grid(True, which="both", ls="-", alpha=0.5)
                        fig.tight_layout()

                        # Create the xarray dataset for the results
                        max_len = max(len(avg_psd_1d), len(avg_psd_1d_ref))
                        wavenumbers = np.arange(max_len)
                        psd_ds = xr.Dataset({
                            'psd_target': (('wavenumber'), np.pad(avg_psd_1d, (0, max_len - len(avg_psd_1d)), 'constant', constant_values=np.nan)),
                            'psd_reference': (('wavenumber'), np.pad(avg_psd_1d_ref, (0, max_len - len(avg_psd_1d_ref)), 'constant', constant_values=np.nan)),
                        }, coords={'wavenumber': wavenumbers})
                        results['time_averaged_psd'][key] = (fig, ax, psd_ds)

                except Exception as e:
                    self.logger.error(f"Failed to process time-averaged PSD for key {key}: {e}", exc_info=True)

            # PSD of Time-Mean (climatology)
            if do_psd_of_time_mean:
                try:
                    self.logger.info(f"Calculating PSD of time-mean for {key}...")

                    if is_healpix:
                        if hp is None:
                            raise ImportError("healpy is required to process HEALPix grids.")

                        # Calculate the climatology
                        data_mean = self.reader_data.timmean(data_da)
                        data_ref_mean = self.reader_data_ref.timmean(data_ref_da)

                        # Prepare the HEALPix maps
                        maps_mean, nside_mean = self._prepare_healpix_maps(
                            data_mean, reader=self.reader_data, replace_nans=has_nans)
                        maps_ref_mean, nside_ref_mean = self._prepare_healpix_maps(
                            data_ref_mean, reader=self.reader_data_ref, replace_nans=has_nans_ref)

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean from each map before hp.anafast
                        # to focus on spatial variability
                        lmax = min(3 * nside_mean - 1, 3 * nside_ref_mean - 1)
                        psd_1d = hp.anafast(maps_mean[0], lmax=lmax)
                        psd_1d_ref = hp.anafast(maps_ref_mean[0], lmax=lmax)

                        # Plot the PSD
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ell = np.arange(len(psd_1d))
                        ell_ref = np.arange(len(psd_1d_ref))
                        ax.loglog(ell, psd_1d, label=f"{data_cfg.get('model', 'Target')}")
                        ax.loglog(ell_ref, psd_1d_ref, label=f"{ref_cfg.get('model', 'Reference')}", linestyle='--')
                        ax.set_xlabel("Spherical harmonic degree (l)")
                        ax.set_ylabel("Power Spectral Density")
                        ax.set_title(f"Power Spectrum of Time-Mean: {base_var}" + (f" at level {level}" if level else ""))
                        ax.legend()
                        ax.grid(True, which="both", ls="-", alpha=0.5)
                        fig.tight_layout()

                        # Create the xarray dataset for the results
                        max_len = max(len(psd_1d), len(psd_1d_ref))
                        wavenumbers = np.arange(max_len)
                        psd_ds = xr.Dataset({
                            'psd_target': (('wavenumber'), np.pad(psd_1d, (0, max_len - len(psd_1d)), 'constant', constant_values=np.nan)),
                            'psd_reference': (('wavenumber'), np.pad(psd_1d_ref, (0, max_len - len(psd_1d_ref)), 'constant', constant_values=np.nan)),
                        }, coords={'wavenumber': wavenumbers})
                        results['psd_of_time_mean'][key] = (fig, ax, psd_ds)
                    else:
                        # Calculate the climatology
                        data_mean = self.reader_data.timmean(data_da).values
                        data_mean = np.nan_to_num(data_mean, nan=0.0) if has_nans else data_mean

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean before FFT to focus on spatial variability
                        psd_1d = self._radial_average(np.abs(fft.fftshift(fft.fft2(data_mean)))**2)

                        # Calculate the climatology
                        data_ref_mean = self.reader_data_ref.timmean(data_ref_da).values
                        data_ref_mean = np.nan_to_num(data_ref_mean, nan=0.0) if has_nans_ref else data_ref_mean

                        # Calculate the PSD
                        # TODO: Consider removing spatial mean before FFT to focus on spatial variability
                        psd_1d_ref = self._radial_average(np.abs(fft.fftshift(fft.fft2(data_ref_mean)))**2)

                        # Plot the PSD
                        fig, ax = plt.subplots(figsize=(10, 6))
                        wavenumber = np.arange(len(psd_1d))
                        wavenumber_ref = np.arange(len(psd_1d_ref))
                        ax.loglog(wavenumber, psd_1d, label=f"{data_cfg.get('model', 'Target')}")
                        ax.loglog(wavenumber_ref, psd_1d_ref, label=f"{ref_cfg.get('model', 'Reference')}", linestyle='--')
                        ax.set_xlabel("Wavenumber")
                        ax.set_ylabel("Power Spectral Density")
                        ax.set_title(f"Power Spectrum of Time-Mean: {base_var}" + (f" at level {level}" if level else ""))
                        ax.legend()
                        ax.grid(True, which="both", ls="-", alpha=0.5)
                        fig.tight_layout()

                        # Create the xarray dataset for the results
                        max_len = max(len(psd_1d), len(psd_1d_ref))
                        wavenumbers = np.arange(max_len)
                        psd_ds = xr.Dataset({
                            'psd_target': (('wavenumber'), np.pad(psd_1d, (0, max_len - len(psd_1d)), 'constant', constant_values=np.nan)),
                            'psd_reference': (('wavenumber'), np.pad(psd_1d_ref, (0, max_len - len(psd_1d_ref)), 'constant', constant_values=np.nan)),
                        }, coords={'wavenumber': wavenumbers})
                        results['psd_of_time_mean'][key] = (fig, ax, psd_ds)

                except Exception as e:
                    self.logger.error(f"Failed to process PSD of time-mean for key {key}: {e}", exc_info=True)

        # Save the figures
        if save_fig:
            self.logger.info("Saving figures...")
            for calc_type, res_dict in results.items():
                if not res_dict: continue
                self.logger.info(f"  Saving {calc_type.replace('_', ' ')} figures...")
                for key, (fig, _, _) in res_dict.items():
                    if fig: self._save_figure(fig, key, calc_type)

        # Save the NetCDF data
        if save_netcdf:
            self.logger.info("Saving NetCDF data...")
            for calc_type, res_dict in results.items():
                if not res_dict: continue
                self.logger.info(f"  Saving {calc_type.replace('_', ' ')} netcdf files...")
                for key, (_, _, ds) in res_dict.items():
                    if ds: self._save_netcdf(ds, key, calc_type)
                
        # Return the results
        return {calc_type: {key: ds for key, (_, _, ds) in res_dict.items() if ds is not None}
                for calc_type, res_dict in results.items()}
