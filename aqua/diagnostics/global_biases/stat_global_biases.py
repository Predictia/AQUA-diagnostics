import xarray as xr
import numpy as np
from scipy import stats
from aqua.core.logger import log_configure
from aqua.core.fldstat import FldStat
from aqua.core.timstat import TimStat
from .util import handle_pressure_level


xr.set_options(keep_attrs=True)


class StatGlobalBiases:
    """
    Class for computing bias statistics between model and reference data.
    It works directly with xarray datasets and includes statistical significance testing.

    Args:
        loglevel (str): Log level. Default is 'WARNING'.
    """
    def __init__(self, loglevel: str = 'WARNING'):

        self.logger = log_configure(log_level=loglevel, log_name='Bias Statistics')
        self.loglevel = loglevel

    def compute_bias_statistics(self,
                                data: xr.Dataset,
                                data_ref: xr.Dataset,
                                var: str,
                                area: xr.DataArray = None,
                                ) -> xr.Dataset:
        """
        Compute global mean bias and RMSE between model and reference data.

        Args:
            data (xr.Dataset): Model climatology dataset.
            data_ref (xr.Dataset): Reference climatology dataset.
            var (str): Variable name.
            area (xr.DataArray, optional): Grid cell areas for weighted statistics.
                                            If None, unweighted statistics will be computed.
        Returns: 
            xr.Dataset: Dataset containing mean bias and RMSE.

        """
        self.logger.info(f'Computing bias statistics for variable {var}.')

        if data is None or data_ref is None:
            raise ValueError("Data or reference data is None after pressure level handling.")

        # Compute bias
        bias = data[var] - data_ref[var]
        
        fldstat = FldStat(area=area, loglevel=self.loglevel)

        # Compute mean bias
        self.logger.debug('Computing area-weighted mean bias.')
        mean_bias = fldstat.fldstat(
            bias,
            stat='mean',
        )

        # Compute RMSE: sqrt(mean(bias^2))
        self.logger.debug('Computing RMSE.')
        bias_squared = bias ** 2
        mean_squared_error = fldstat.fldstat(
            bias_squared,
            stat='mean'
        )
        rmse = np.sqrt(mean_squared_error)

        stats = xr.Dataset({
            'mean_bias': mean_bias,
            'rmse': rmse
        })
        
        self.logger.info(f'Mean bias: {float(mean_bias.values):.4e} {data[var].attrs.get("units", "")}')
        self.logger.info(f'RMSE: {float(rmse.values):.4e} {data[var].attrs.get("units", "")}')

        return stats
    
    def compute_yearly_temporal_means(self,
                                    data: xr.Dataset,
                                    var: str) -> xr.DataArray:
        """Compute yearly temporal means for a given variable.
        Args:
            data (xr.Dataset): Input dataset with time dimension.
            var (str): Variable name to compute means for.
        Returns:
            xr.DataArray: Yearly temporal means of the variable.
        """
        if 'time' not in data[var].dims:
            raise ValueError(f"Variable {var} does not have a 'time' dimension.")

        timstat = TimStat(loglevel=self.loglevel)
        yearly_means = timstat.timstat(data[[var]], stat='mean', freq='YS')
        self.logger.info(f'Computed {len(yearly_means.time)} yearly means.')
        
        return yearly_means[var]

    def ttest_at_grid_point(self, model_vals, ref_vals, min_samples: int = 3):
        """Perform t-test at a single grid point.
        Args:
            model_vals (np.ndarray): 1D array of model values at a grid point.
            ref_vals (np.ndarray): 1D array of reference values at the same grid point.
            min_samples (int): Minimum number of samples required to perform the t-test. Default is 3.
        Returns:
            float: p-value from the t-test.
        """
        # Remove NaN values
        model_clean = model_vals[np.isfinite(model_vals)]
        ref_clean = ref_vals[np.isfinite(ref_vals)]
        
        if len(model_clean) < min_samples or len(ref_clean) < min_samples:
            return np.nan
        
        # Perform t-test
        _, p_value = stats.ttest_ind(model_clean, ref_clean, equal_var=False)
        return p_value

    @staticmethod
    def _lag1_autocorrelation(data_array: xr.DataArray) -> xr.DataArray:
        """Compute lag-one temporal autocorrelation at every grid point."""
        if 'time' not in data_array.dims:
            raise ValueError("Input data must include a 'time' dimension.")
        n_samples = data_array.sizes['time']
        if n_samples < 2:
            raise ValueError("At least two samples are required.")
        first = data_array.isel(time=slice(0, -1)).assign_coords(
            time=np.arange(n_samples - 1)
        )
        second = data_array.isel(time=slice(1, None)).assign_coords(
            time=np.arange(n_samples - 1)
        )
        first = first - first.mean('time')
        second = second - second.mean('time')
        correlation = (first * second).mean('time') / (
            first.std('time') * second.std('time')
        )
        return correlation.clip(min=-0.99, max=0.99)

    @staticmethod
    def _effective_sample_size(r1: xr.DataArray, n_samples: int) -> xr.DataArray:
        """Estimate effective sample size using the lag-one AR(1) correction."""
        effective = n_samples * (1.0 - r1) / (1.0 + r1)
        effective = effective.where(np.isfinite(effective), 2.0)
        return effective.clip(min=2.0, max=float(n_samples))

    def _compute_autocorr_effective_significance(
            self, data: xr.Dataset, data_ref: xr.Dataset, var: str,
            alpha: float, min_samples: int) -> xr.DataArray:
        """Test the mean bias using autocorrelation-adjusted sample sizes.

        This method uses the full aligned time series, estimates an AR(1)
        effective sample size independently for model and reference, and
        compares the mean difference against a two-sided Student-t threshold.
        """
        model, reference = xr.align(data[var], data_ref[var], join='inner')
        n_samples = int(model.sizes.get('time', 0))
        if n_samples < min_samples:
            self.logger.warning(
                'Insufficient samples for autocorrelation-aware significance '
                '(available=%s, minimum=%s). Returning all False.',
                n_samples, min_samples
            )
            return xr.zeros_like(model.isel(time=0, drop=True), dtype=bool)

        sigma_model = model.std('time', ddof=1)
        sigma_reference = reference.std('time', ddof=1)
        r1_model = self._lag1_autocorrelation(model)
        r1_reference = self._lag1_autocorrelation(reference)
        n_eff_model = self._effective_sample_size(r1_model, n_samples)
        n_eff_reference = self._effective_sample_size(r1_reference, n_samples)
        standard_error = np.sqrt(
            sigma_model ** 2 / n_eff_model
            + sigma_reference ** 2 / n_eff_reference
        )
        t_stat = (model.mean('time') - reference.mean('time')) / standard_error
        degrees_freedom = (n_eff_model + n_eff_reference - 2.0).clip(min=1.0)
        critical = xr.apply_ufunc(
            lambda value: stats.t.ppf(1.0 - alpha / 2.0, value),
            degrees_freedom,
            dask='allowed',
            vectorize=True,
            output_dtypes=[float],
        )
        significant = xr.where(
            np.isfinite(t_stat), np.abs(t_stat) > critical, False
        ).astype(bool)
        significant.attrs.update({
            'long_name': 'Autocorrelation-aware statistical significance of bias',
            'description': (
                'Two-sided t-test using lag-1 AR(1) effective sample sizes'
            ),
            'method': 'autocorr_effective',
            'alpha': alpha,
            'n_samples_model': n_samples,
            'n_samples_reference': n_samples,
            'mean_effective_samples_model': float(n_eff_model.mean().values),
            'mean_effective_samples_reference': float(
                n_eff_reference.mean().values
            ),
        })
        return significant


    def compute_significance_ttest(self,
                                   data: xr.Dataset,
                                   data_ref: xr.Dataset,
                                   var: str,
                                   alpha: float = 0.05,
                                   min_samples: int = 3,
                                   method: str = 'welch_yearly') -> xr.DataArray:
        """
        Compute statistical significance of bias using two-sample t-test.
        
        ``welch_yearly`` performs the established two-sided Welch test on
        yearly means. ``autocorr_effective`` uses the full series with lag-one
        autocorrelation-adjusted effective sample sizes.

        Args:
            data (xr.Dataset): Model dataset with time dimension.
            data_ref (xr.Dataset): Reference dataset with time dimension.
            var (str): Variable name.
            alpha (float): Significance level (default: 0.05 for 95% confidence).
            min_samples (int): Minimum number of samples required to perform test. Default is 3.
            method (str): ``'welch_yearly'`` or ``'autocorr_effective'``.

        Returns:
            xr.DataArray: Boolean array where True indicates statistically significant differences.
                         Same spatial dimensions as input data.
        """
        if method not in {'welch_yearly', 'autocorr_effective'}:
            raise ValueError(
                "method must be 'welch_yearly' or 'autocorr_effective'"
            )
        self.logger.info(
            'Computing statistical significance using %s (alpha=%s).',
            method, alpha
        )
        if method == 'autocorr_effective':
            return self._compute_autocorr_effective_significance(
                data=data, data_ref=data_ref, var=var,
                alpha=alpha, min_samples=min_samples
            )

        # Get temporal means
        data_temporal = self.compute_yearly_temporal_means(data, var)
        data_ref_temporal = self.compute_yearly_temporal_means(data_ref, var)
        
        # Check if we have enough samples
        n_samples = len(data_temporal.time)
        n_samples_ref = len(data_ref_temporal.time) 
        
        self.logger.info(f'Number of samples - Model: {n_samples}, Reference: {n_samples_ref}')

        if n_samples < min_samples or n_samples_ref < min_samples:
            self.logger.warning(
                f'Insufficient samples for t-test. Model: {n_samples}, Reference: {n_samples_ref}. '
                f'Minimum required: {min_samples}. Returning all False.'
            )
            # Return array of False (not significant) with same shape as spatial dimensions
            return xr.DataArray(
                np.zeros(data[var].isel(time=0).shape, dtype=bool),
                coords={k: v for k, v in data[var].isel(time=0).coords.items() if k != 'time'},
                dims=[d for d in data[var].dims if d != 'time']
            )

        # Rechunk along time dimension for dask compatibility
        # This is needed because apply_ufunc with dask='parallelized' requires
        # core dimensions to be in a single chunk
                
        # Get time dimension name
        time_dim = 'time' 
        
        self.logger.debug(f'Rechunking data along {time_dim} dimension.')
        data_temporal = data_temporal.chunk({time_dim: -1})
        data_ref_temporal = data_ref_temporal.chunk({time_dim: -1})

        # Perform t-test at each grid point
        # Use scipy.stats.ttest_ind for independent samples
        self.logger.debug('Performing t-test at each grid point.')
        
        # Rename time dimensions to avoid xarray alignment issues when model and reference
        # have different number of time steps. Using distinct names prevents apply_ufunc
        # from attempting to align the two time axes against each other.       
        time_dim = f'{time_dim}'
        time_dim_ref = f'{time_dim}_ref'

        data_temporal = data_temporal.rename({time_dim: time_dim})
        data_ref_temporal = data_ref_temporal.rename({time_dim: time_dim_ref})

        # Apply t-test across all grid points
        p_values = xr.apply_ufunc(
            self.ttest_at_grid_point,
            data_temporal,
            data_ref_temporal,
            input_core_dims=[[time_dim], [time_dim_ref]], 
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            kwargs={"min_samples": min_samples},
        )
        # Determine significance (True where p-value < alpha)
        is_significant = p_values < alpha
        self.logger.debug("Loading significance map in memory")
        is_significant.load()
        self.logger.debug("Loaded significance map in memory")
        
        # Count significant points
        n_significant = int(is_significant.sum().values)
        n_total = int(np.prod(is_significant.shape))
        pct_significant = 100 * n_significant / n_total if n_total > 0 else 0
        
        self.logger.info(
            f'Statistical significance test completed: '
            f'{n_significant}/{n_total} points ({pct_significant:.1f}%) are significant at alpha={alpha}.'
        )

        # Add metadata
        is_significant.attrs.update({
            'long_name': 'Statistical significance of bias',
            'description': f'Two-sample t-test with alpha={alpha}',
            'method': 'welch_yearly',
            'alpha': alpha,
            'n_samples_model': n_samples,
            'n_samples_reference': n_samples_ref,
            'n_significant_points': n_significant,
            'percent_significant': pct_significant
        })

        return is_significant

        
