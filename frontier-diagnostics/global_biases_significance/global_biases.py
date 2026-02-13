import pandas as pd
import numpy as np
import xarray as xr
from statistics import NormalDist
from aqua.core.logger import log_configure
from aqua.core.util import select_season, convert_data_units
from aqua.core.fixer import EvaluateFormula
from aqua.core.exceptions import NoDataError
from aqua.diagnostics.base import Diagnostic
try:
    from .util import handle_pressure_level
except ImportError:
    # Allow execution as a standalone script from this directory.
    from util import handle_pressure_level


xr.set_options(keep_attrs=True)


class GlobalBiases(Diagnostic):
    """
    Diagnostic class for computing global and seasonal climatologies of a given variable.

    This class handles data retrieval, pressure level selection, unit conversion, 
    and computation of mean climatologies (total or seasonal).

    Inherits from `Diagnostic`.

    Args:
        catalog (str): The catalog to be used. If None, inferred from Reader.
        model (str): Model to be used.
        exp (str): Experiment name.
        source (str): Source name.
        regrid (str): Target grid for regridding. If None, no regridding.
        startdate (str): Start date for data selection.
        enddate (str): End date for data selection.
        var (str): Variable name to analyze.
        plev (float): Pressure level to select (if applicable).
        diagnostic (str): Name of the diagnostic.
        save_netcdf (bool): If True, saves output climatologies.
        outputdir (str): Output directory for NetCDF files.
        loglevel (str): Log level. Default is 'WARNING'.
    """
    def __init__(self, catalog=None, model=None, exp=None, source=None,
                 regrid=None, startdate=None, enddate=None,
                 var=None, plev=None,
                 diagnostic='globalbiases',
                 save_netcdf=True, outputdir='./', loglevel='WARNING'):

        super().__init__(catalog=catalog, model=model, exp=exp, source=source,
                         regrid=regrid, startdate=startdate, enddate=enddate,
                         loglevel=loglevel)

        self.logger = log_configure(log_level=loglevel, log_name='Global Biases')
        self.var = var
        self.plev = plev
        self.save_netcdf = save_netcdf
        self.outputdir = outputdir
        self.startdate = startdate
        self.enddate = enddate
        self.diagnostic = diagnostic

    def _check_data(self, var: str, units: str):
        """
        Make sure that the data is in the correct units.

        Args:
            var (str): The variable to be checked.
            units (str): The units to be checked.
        """
        self.data[self.var] = super()._check_data(data=self.data[self.var], var=var, units=units)


    def retrieve(self, var: str = None, formula: bool = False,
                 long_name: str = None, short_name: str = None,
                 plev: float = None, units: str = None,
                 reader_kwargs: dict = {}) -> None:
        """
        Retrieve and preprocess dataset, selecting pressure level and/or converting units if needed.

        Args:
            var (str, optional): Variable to retrieve. If None, uses self.var.
            formula (bool): If True, the variable is a formula.
            long_name (str): The long name of the variable, if different from the variable name.
            short_name (str): The short name of the variable, if different from the variable name.
            plev (float, optional): Pressure level to extract.
            units (str): The units of the variable, if different from the original units.
            reader_kwargs (dict, optional): Additional keyword arguments for the Reader.
        Raises:
            NoDataError: If variable not found in dataset.
            KeyError: If the variable is missing from the data.
        """
        if var is not None:
            self.var = var   
        if formula:
            super().retrieve(reader_kwargs=reader_kwargs)
            self.logger.info("Evaluating formula: %s", self.var)
            formula_values = EvaluateFormula(data=self.data, formula=self.var, long_name=long_name,
                                             short_name=short_name, units=units,
                                             loglevel=self.loglevel).evaluate()
            if formula_values is None:
                raise ValueError(f'Error evaluating formula {var}. '
                                 'Check the variable names and the formula syntax.')
            self.data[self.var] = formula_values
        else:
            super().retrieve(var=self.var, reader_kwargs=reader_kwargs)

        if self.data is None:
            self.logger.error("Data could not be retrieved for %s, %s, %s", self.AQUA_model, self.AQUA_exp, self.AQUA_source)
            raise NoDataError("No data retrieved.")

        # Customize metadata and attributes
        if units is not None:
            self._check_data(var=self.var, units=units)

        if short_name is not None:
            self.data = self.data.rename_vars({self.var: short_name})
            self.var = short_name
        else:
            self.data.attrs['short_name'] = self.var

        self.startdate = pd.Timestamp(self.startdate or self.data.time[0].values).strftime("%Y-%m-%d")
        self.enddate = pd.Timestamp(self.enddate or self.data.time[-1].values).strftime("%Y-%m-%d")
        if plev is not None:
            self.plev = plev

        # Final validation and pressure level handling
        if self.var:
            if self.var not in self.data.data_vars:
                raise KeyError(f"Variable '{self.var}' not found in dataset. Available variables: {list(self.data.data_vars)}")

            if self.plev is not None:
                self.logger.info("Selecting pressure level %s for variable '%s'.", self.plev, self.var)
                self.data = handle_pressure_level(self.data, self.var, self.plev, loglevel=self.loglevel)
            elif 'plev' in self.data[self.var].dims:
                self.logger.warning("Variable '%s' has multiple pressure levels, but none was specified.", self.var)
        else:
            self.logger.info("All variables retrieved; no variable-specific operations applied.")

    def _lag1_autocorrelation(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Compute lag-1 temporal autocorrelation at each grid point.

        Args:
            data_array (xr.DataArray): DataArray with a time dimension.

        Returns:
            xr.DataArray: Lag-1 autocorrelation field.
        """
        if 'time' not in data_array.dims:
            raise ValueError("Input data must include a 'time' dimension.")

        n_samples = data_array.sizes['time']
        if n_samples < 2:
            raise ValueError("At least two time samples are required for lag-1 autocorrelation.")

        x_t = data_array.isel(time=slice(0, -1)).assign_coords(time=np.arange(n_samples - 1))
        x_t1 = data_array.isel(time=slice(1, None)).assign_coords(time=np.arange(n_samples - 1))

        x_t_anom = x_t - x_t.mean(dim='time')
        x_t1_anom = x_t1 - x_t1.mean(dim='time')

        covariance = (x_t_anom * x_t1_anom).mean(dim='time')
        std_prod = x_t_anom.std(dim='time') * x_t1_anom.std(dim='time')
        r1 = covariance / std_prod

        # Keep correlation in a numerically stable range for N_eff computation.
        return r1.clip(min=-0.99, max=0.99)

    def _effective_sample_size(self, r1: xr.DataArray, n_samples: int) -> xr.DataArray:
        """
        Compute effective sample size from lag-1 autocorrelation.

        Args:
            r1 (xr.DataArray): Lag-1 autocorrelation.
            n_samples (int): Number of temporal samples.

        Returns:
            xr.DataArray: Effective sample size with numerical safeguards.
        """
        n_eff = n_samples * (1.0 - r1) / (1.0 + r1)
        n_eff = n_eff.where(np.isfinite(n_eff), 2.0)
        return n_eff.clip(min=2.0, max=float(n_samples))

    def _t_critical(self, alpha: float, df: xr.DataArray, use_student_t: bool) -> xr.DataArray:
        """
        Compute two-tailed critical value for significance.

        Args:
            alpha (float): Significance level.
            df (xr.DataArray): Degrees of freedom field.
            use_student_t (bool): If True, use Student's t where possible.

        Returns:
            xr.DataArray: Critical value field.
        """
        z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        if not use_student_t:
            return xr.full_like(df, fill_value=z_critical, dtype=float)

        try:
            from scipy.stats import t as student_t  # Optional dependency in runtime environment.
            t_crit = xr.apply_ufunc(
                lambda x: student_t.ppf(1.0 - alpha / 2.0, x),
                df.clip(min=1.0),
                dask='allowed',
                vectorize=True,
                output_dtypes=[float],
            )
            return t_crit.where(np.isfinite(t_crit), z_critical)
        except ImportError:
            self.logger.warning(
                "scipy is not available. Falling back to normal approximation for critical value."
            )
            return xr.full_like(df, fill_value=z_critical, dtype=float)

    def _compute_bias_significance_from_series(
        self,
        data_array: xr.DataArray,
        data_ref_array: xr.DataArray,
        alpha: float = 0.05,
        use_student_t: bool = True,
    ) -> xr.Dataset:
        """
        Compute bias and statistical significance from full time series.

        Args:
            data_array (xr.DataArray): Dataset variable time series.
            data_ref_array (xr.DataArray): Reference variable time series.
            alpha (float): Significance level for two-tailed test.
            use_student_t (bool): If True, use Student's t critical value.

        Returns:
            xr.Dataset: Bias, standard error, t-statistics and significance mask.
        """
        if 'time' not in data_array.dims or 'time' not in data_ref_array.dims:
            raise ValueError("Both data arrays must include a 'time' dimension.")

        data_array, data_ref_array = xr.align(data_array, data_ref_array, join='inner')
        n_samples = int(data_array.sizes['time'])
        if n_samples < 2:
            raise ValueError("Need at least two aligned samples to compute significance.")

        bias = data_array.mean(dim='time') - data_ref_array.mean(dim='time')
        sigma_data = data_array.std(dim='time', ddof=1)
        sigma_ref = data_ref_array.std(dim='time', ddof=1)

        r1_data = self._lag1_autocorrelation(data_array)
        r1_ref = self._lag1_autocorrelation(data_ref_array)

        n_eff_data = self._effective_sample_size(r1_data, n_samples=n_samples)
        n_eff_ref = self._effective_sample_size(r1_ref, n_samples=n_samples)

        se_data = sigma_data / np.sqrt(n_eff_data)
        se_ref = sigma_ref / np.sqrt(n_eff_ref)
        se_bias = np.sqrt(se_data**2 + se_ref**2).where(lambda x: x > 0)

        t_stat = bias / se_bias
        dof = (n_eff_data + n_eff_ref - 2.0).clip(min=1.0)
        t_critical = self._t_critical(alpha=alpha, df=dof, use_student_t=use_student_t)
        is_significant = xr.where(np.isfinite(t_stat), np.abs(t_stat) > t_critical, False)

        return xr.Dataset({
            'bias': bias,
            'sigma_data': sigma_data,
            'sigma_ref': sigma_ref,
            'r1_data': r1_data,
            'r1_ref': r1_ref,
            'n_eff_data': n_eff_data,
            'n_eff_ref': n_eff_ref,
            'se_bias': se_bias,
            't_stat': t_stat,
            't_critical': t_critical,
            'is_significant': is_significant,
        })

    def compute_bias_significance(
        self,
        data: xr.Dataset = None,
        data_ref: xr.Dataset = None,
        var: str = None,
        seasonal: bool = False,
        alpha: float = 0.05,
        use_student_t: bool = True,
    ) -> xr.Dataset:
        """
        Compute point-wise bias significance for annual and seasonal climatologies.

        Args:
            data (xr.Dataset, optional): Primary dataset. Defaults to self.data.
            data_ref (xr.Dataset): Reference dataset.
            var (str, optional): Variable name. Defaults to self.var.
            seasonal (bool): If True, compute DJF/MAM/JJA/SON significance.
            alpha (float): Significance level for the two-tailed test.
            use_student_t (bool): If True, use Student's t critical value.

        Returns:
            xr.Dataset: Significance dataset (annual or seasonal).
        """
        if data is None:
            data = self.data
        if var is None:
            var = self.var

        if data is None or data_ref is None:
            raise ValueError("Both data and data_ref must be provided.")
        if var not in data or var not in data_ref:
            raise KeyError(f"Variable '{var}' must exist in both data and data_ref datasets.")

        self.logger.info(
            "Computing %s bias significance for variable %s (alpha=%s).",
            "seasonal" if seasonal else "annual",
            var,
            alpha,
        )

        if not seasonal:
            significance = self._compute_bias_significance_from_series(
                data[var], data_ref[var], alpha=alpha, use_student_t=use_student_t
            )
            significance.attrs.update({
                'AQUA_catalog': self.catalog,
                'AQUA_model': self.model,
                'AQUA_exp': self.exp,
                'AQUA_realization': self.realization,
                'startdate': str(self.startdate),
                'enddate': str(self.enddate),
                'alpha': alpha,
            })
            self.bias_significance = significance
            return significance

        season_list = ['DJF', 'MAM', 'JJA', 'SON']
        seasonal_results = []
        for season in season_list:
            data_season = select_season(data[var], season)
            data_ref_season = select_season(data_ref[var], season)
            season_significance = self._compute_bias_significance_from_series(
                data_season,
                data_ref_season,
                alpha=alpha,
                use_student_t=use_student_t,
            )
            seasonal_results.append(season_significance.expand_dims(season=[season]))

        significance = xr.concat(seasonal_results, dim='season')
        significance.attrs.update({
            'AQUA_catalog': self.catalog,
            'AQUA_model': self.model,
            'AQUA_exp': self.exp,
            'AQUA_realization': self.realization,
            'startdate': str(self.startdate),
            'enddate': str(self.enddate),
            'alpha': alpha,
        })
        self.seasonal_bias_significance = significance
        return significance

    def savenetcdf(self, data: xr.Dataset, diagnostic_product: str, 
                    rebuild: bool = True, create_catalog_entry: bool = False, extra_keys = None,
                    dict_catalog_entry: dict = {'jinjalist': ['realization'],
                                                'wildcardlist': ['var']}):
        """
        data (xr.Dataset): Input dataset.
        diagnostic_product (str): The product name to be used in the filename (e.g., 'annual_climatology').
        rebuild (bool): If True, rebuild the data from the original files.
        create_catalog_entry (bool): If True, create a catalog entry for the data. Default is False.
        extra_keys (dict): Extra keys for filename generation.
        dict_catalog_entry (dict): A dictionary with catalog entry information. 
            Default is {'jinjalist': ['freq', 'region', 'realization'], 'wildcardlist': ['var']}.
        """
        super().save_netcdf(data=data,
                diagnostic=self.diagnostic,
                diagnostic_product=diagnostic_product,
                outputdir=self.outputdir,
                create_catalog_entry=create_catalog_entry,
                dict_catalog_entry=dict_catalog_entry,
                extra_keys=extra_keys)

    
    def compute_climatology(self,
                            data: xr.Dataset = None,
                            var: str = None,
                            plev: float = None,
                            save_netcdf: bool = None,
                            seasonal: bool = False,
                            seasons_stat: str = 'mean',
                            create_catalog_entry: bool = False
                            ) -> None:
        """
        Compute total and optionally seasonal climatology for a variable.

        Args:
            data (xarray.Dataset, optional): Input dataset. If None, uses self.data.
            var (str, optional): Variable name. If None, uses self.var.
            plev (float, optional): Pressure level (currently unused).
            save_netcdf (bool, optional): If True, save output to NetCDF.
            seasonal (bool): If True, compute seasonal climatology (DJF, MAM, JJA, SON).
            seasons_stat (str): Aggregation statistic: 'mean', 'std', 'max', 'min'.
            create_catalog_entry (bool): If True, create a catalog entry for the data. Default is False.
        Raises:
            ValueError: If `seasons_stat` is invalid.
        """
        data = data or self.data
        var = var or self.var

        if save_netcdf is None:
            save_netcdf = self.save_netcdf

        if data is None:
            raise ValueError("No data provided or retrieved; cannot compute climatology.")

        self.logger.info(f'Computing climatology for variable {var}.')

        self.climatology = xr.Dataset({var: data[var].mean(dim='time')})
        self.climatology.attrs.update({
            'AQUA_catalog': self.catalog,
            'AQUA_model': self.model,
            'AQUA_exp': self.exp,
            'AQUA_realization': self.realization,
            'startdate': str(self.startdate),
            'enddate': str(self.enddate)
        })

        # Load data in memory for faster plot
        self.logger.debug(f"Loading climatology data in memory")
        self.climatology.load()
        self.logger.debug(f"Loaded climatology data in memory")

        if save_netcdf:
            extra_keys = {
                k: v for k, v in {
                    'var': var,
                    'plev': plev,
                }.items() if v is not None
            }
            self.savenetcdf(
                data=self.climatology,
                diagnostic_product='annual_climatology',
                create_catalog_entry=create_catalog_entry,
                extra_keys=extra_keys
            )

        if seasonal:
            stat_funcs = {'mean': 'mean', 'max': 'max', 'min': 'min', 'std': 'std'}
            if seasons_stat not in stat_funcs:
                raise ValueError("Invalid statistic. Choose one of 'mean', 'std', 'max', 'min'.")

            self.logger.info(f'Computing seasonal climatology for variable {var}.')

            season_list = ['DJF', 'MAM', 'JJA', 'SON']
            seasonal_data = []

            for season in season_list:
                season_data = select_season(data[var], season)
                season_stat = getattr(season_data, stat_funcs[seasons_stat])(dim='time')
                seasonal_data.append(season_stat.expand_dims(season=[season]))

            self.seasonal_climatology = xr.concat(seasonal_data, dim='season', coords='different').to_dataset(name=var)
            self.seasonal_climatology.attrs.update({
                'AQUA_catalog': self.catalog,
                'AQUA_model': self.model,
                'AQUA_exp': self.exp,
                'AQUA_realization': self.realization,
                'startdate': str(self.startdate),
                'enddate': str(self.enddate)
            })

            # Load data in memory for faster plot
            self.logger.debug(f"Loading seasonal climatology data in memory")
            self.seasonal_climatology.load()
            self.logger.debug(f"Loaded seasonal climatology data in memory")

            if save_netcdf:
                extra_keys = {k: v for k, v in [('var', var), ('plev', plev)] if v is not None}
                self.savenetcdf(
                    data=self.seasonal_climatology,
                    diagnostic_product='seasonal_climatology',
                    create_catalog_entry=create_catalog_entry,
                    extra_keys=extra_keys
                )
                self.logger.info(f'Seasonal climatology saved to {self.outputdir}.')
