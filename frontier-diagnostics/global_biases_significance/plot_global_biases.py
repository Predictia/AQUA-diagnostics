import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from aqua.core.logger import log_configure
from aqua.core.graphics import plot_single_map, plot_single_map_diff, plot_maps, plot_vertical_profile_diff
from aqua.core.util import get_projection, get_realizations
from aqua.diagnostics.base import OutputSaver
try:
    from .util import handle_pressure_level
except ImportError:
    # Allow execution as a standalone script from this directory.
    from util import handle_pressure_level

class PlotGlobalBiases: 
    def __init__(self, 
                 diagnostic='globalbiases',
                 save_pdf=True, save_png=True, 
                 dpi=300, outputdir='./',
                 cmap='RdBu_r',
                 loglevel='WARNING'):
        """
        Initialize the PlotGlobalBiases class.

        Args:
            diagnostic (str): Name of the diagnostic.
            save_pdf (bool): Whether to save the figure as PDF.
            save_png (bool): Whether to save the figure as PNG.
            dpi (int): Resolution of saved figures.
            outputdir (str): Output directory for saved plots.
            cmap (str): Colormap to use for the plots.
            loglevel (str): Logging level.
        """
        self.diagnostic = diagnostic
        self.save_pdf = save_pdf
        self.save_png = save_png
        self.dpi = dpi
        self.outputdir = outputdir
        self.cmap = cmap
        self.loglevel = loglevel

        self.logger = log_configure(log_level=loglevel, log_name='Global Biases')

    def _should_use_contour(self, *arrays: xr.DataArray) -> bool:
        """
        Return False when any input array contains missing values.
        Contourf on projected global grids can fail with NaN-masked regions
        (e.g., SST over land), while pcolormesh is generally robust.
        """
        for arr in arrays:
            if arr is None:
                continue
            try:
                has_missing = bool(arr.isnull().any().item())
            except Exception:
                has_missing = bool(np.isnan(arr.values).any())
            if has_missing:
                self.logger.info(
                    "Detected missing values in map data; using pcolormesh instead of contourf."
                )
                return False
        return True

    def _overlay_non_significant_points(self, ax, significance_mask: xr.DataArray, marker_stride: int = 1):
        """
        Overlay crosses on grid points where bias is not significant.

        Args:
            ax (matplotlib.axes.Axes): Target map axis.
            significance_mask (xr.DataArray): Boolean mask where True means significant.
            marker_stride (int): Subsampling step for markers to reduce clutter.
        """
        if significance_mask is None:
            return

        if not isinstance(significance_mask, xr.DataArray):
            self.logger.warning("Significance mask is not an xarray.DataArray. Skipping overlay.")
            return

        lon_name = 'lon' if 'lon' in significance_mask.coords else 'longitude' if 'longitude' in significance_mask.coords else None
        lat_name = 'lat' if 'lat' in significance_mask.coords else 'latitude' if 'latitude' in significance_mask.coords else None
        if lon_name is None or lat_name is None:
            self.logger.warning("Could not find longitude/latitude coordinates in significance mask. Skipping overlay.")
            return

        non_significant = xr.where(
            np.isfinite(significance_mask),
            ~significance_mask.astype(bool),
            False
        )
        if marker_stride > 1:
            non_significant = non_significant.isel(
                {lat_name: slice(0, None, marker_stride), lon_name: slice(0, None, marker_stride)}
            )

        valid_points = non_significant.values
        if not np.any(valid_points):
            return

        lon_2d, lat_2d = np.meshgrid(non_significant[lon_name].values, non_significant[lat_name].values)
        ax.scatter(
            lon_2d[valid_points],
            lat_2d[valid_points],
            marker='x',
            s=8,
            linewidths=0.35,
            color='k',
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )

    def _save_figure(self, fig, diagnostic_product, 
                     data, description, var, data_ref=None, 
                     plev=None, **kwargs):
        """
        Handles the saving of a figure using OutputSaver.

        Args:
            fig (matplotlib.Figure): The figure to save.
            data (xarray.Dataset): Dataset.
            data_ref (xarray.Dataset, optional): Reference dataset.
            diagnostic_product (str): Name of the diagnostic product.
            description (str): Description of the figure.
            var (str): Variable name.
            plev (float, optional): Pressure level.
        Keyword Args:
            **kwargs: Additional keyword arguments to be passed to the OutputSaver.
        """
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic,
            catalog=data.AQUA_catalog,
            model=data.AQUA_model,
            exp=data.AQUA_exp,
            model_ref=data_ref.AQUA_model if data_ref else None,
            exp_ref=data_ref.AQUA_exp if data_ref else None,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
            **kwargs
        )

        metadata = {"Description": description}
        extra_keys = {}

        if var is not None:
            extra_keys.update({'var': var})
        if plev is not None:
            extra_keys.update({'plev': plev})

        outputsaver.save_figure(fig, diagnostic_product,
                                extra_keys=extra_keys, metadata=metadata,
                                save_pdf=self.save_pdf, save_png=self.save_png,
                                dpi=self.dpi)

    def plot_climatology(self, data, var, plev=None, proj='robinson', proj_params={}, vmin=None, vmax=None, cbar_label=None):
        """
        Plots the climatology map for a given variable and time range.

        Args:
            data (xarray.Dataset): Climatology dataset to plot.
            var (str): Variable name.
            plev (float, optional): Pressure level to plot (if applicable).
            proj (string, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection (e.g., {'central_longitude': 0}).
            vmin (float, optional): Minimum color scale value.
            vmax (float, optional): Maximum color scale value.
            cbar_label (str, optional): Label for the colorbar.

        Returns:
            tuple: Matplotlib figure and axis objects.
        """
        self.logger.info('Plotting climatology.')
        
        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        if data is None:
            return None

        realization = get_realizations(data)
        proj = get_projection(proj, **proj_params)
        
        title = (f"Climatology of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}" 
                + (f" at {int(plev / 100)} hPa" if plev else ""))

        fig, ax = plot_single_map(
            data[var],
            return_fig=True,
            title=title,
            title_size=18,
            vmin=vmin,
            vmax=vmax,
            proj=proj,
            loglevel=self.loglevel,
            cbar_label=cbar_label,
            cmap=self.cmap
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        description = (
            f"Spatial map of the climatology {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" from {data.startdate} to {data.enddate} "
            f"for the {data.AQUA_model} model, experiment {data.AQUA_exp}."
        )

        self._save_figure(fig=fig, diagnostic_product='annual_climatology',
                          data=data, description=description, var=var, plev=plev, realization=realization)


    def plot_bias(
        self,
        data,
        data_ref,
        var,
        plev=None,
        proj='robinson',
        proj_params={},
        vmin=None,
        vmax=None,
        cbar_label=None,
        significance_mask: xr.DataArray = None,
        show_significance: bool = True,
        marker_stride: int = 1,
    ):
        """
        Plots the bias map between two datasets.

        Args:
            data (xarray.Dataset): Primary dataset.
            data_ref (xarray.Dataset): Reference dataset.
            var (str): Variable name.
            plev (float, optional): Pressure level.
            proj (str, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            cbar_label (str, optional): Label for the colorbar.
            significance_mask (xarray.DataArray, optional): Boolean mask where True means significant.
            show_significance (bool): If True, overlay non-significant points as crosses.
            marker_stride (int): Grid subsampling stride for significance overlay markers.
        """
        self.logger.info('Plotting global biases.')

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        data_ref = handle_pressure_level(data_ref, var, plev, loglevel=self.loglevel)

        realization = get_realizations(data)

        sym = vmin is None or vmax is None

        proj = get_projection(proj, **proj_params)

        title = (f"Global bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
                 f"relative to {data_ref.AQUA_model} climatology"
                 + (f" at {int(plev / 100)} hPa" if plev else ""))

        use_contour = self._should_use_contour(data[var], data_ref[var])

        fig, ax = plot_single_map_diff(
            data=data[var], 
            data_ref=data_ref[var],
            return_fig=True,
            contour=use_contour,
            title=title,
            title_size=18,
            sym=sym,
            proj=proj,
            vmin_fill=vmin, 
            vmax_fill=vmax,
            cbar_label=cbar_label,
            cmap=self.cmap,
            loglevel=self.loglevel
        )
        if show_significance and significance_mask is not None:
            self._overlay_non_significant_points(
                ax=ax,
                significance_mask=significance_mask,
                marker_stride=max(1, int(marker_stride)),
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        description = (
            f"Spatial map of global bias of {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" from {data.startdate} to {data.enddate}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp}, with {data_ref.AQUA_model}"
            f" from {data_ref.startdate} to {data_ref.enddate} used as reference data."
            + (" Crosses mark grid points with non-significant bias (two-tailed test)." if show_significance and significance_mask is not None else "")
        )

        self._save_figure(fig=fig, diagnostic_product='bias', data=data, data_ref=data_ref,
                          description=description, var=var, plev=plev, realization=realization)


    def plot_seasonal_bias(
        self,
        data,
        data_ref,
        var,
        plev=None,
        proj='robinson',
        proj_params={},
        vmin=None,
        vmax=None,
        cbar_label=None,
        significance_mask: xr.DataArray = None,
        show_significance: bool = True,
        marker_stride: int = 1,
    ):
        """
        Plots seasonal biases for each season (DJF, MAM, JJA, SON).

        Args:
            data (xarray.Dataset): Primary dataset.
            data_ref (xarray.Dataset): Reference dataset.
            var (str): Variable name.
            plev (float, optional): Pressure level.
            proj (str, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            cbar_label (str, optional): Label for the colorbar.
            significance_mask (xarray.DataArray, optional): Boolean seasonal mask where True means significant.
            show_significance (bool): If True, overlay non-significant points as crosses.
            marker_stride (int): Grid subsampling stride for significance overlay markers.

        Returns:
            matplotlib.figure.Figure: The resulting figure.
        """
        self.logger.info('Plotting seasonal biases.')

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        data_ref = handle_pressure_level(data_ref, var, plev, loglevel=self.loglevel)

        realization = get_realizations(data)

        season_list = ['DJF', 'MAM', 'JJA', 'SON']
        sym = vmin is None or vmax is None

        title = (f"Seasonal bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
                 f"relative to {data_ref.AQUA_model} climatology"
                 + (f" at {int(plev / 100)} hPa" if plev else ""))

        seasonal_maps = [data[var].sel(season=season) - data_ref[var].sel(season=season) for season in season_list]
        use_contour = self._should_use_contour(*seasonal_maps)

        plot_kwargs = {
            'maps': seasonal_maps,
            'proj': get_projection(proj, **proj_params),
            'return_fig': True,
            'title': title,
            'title_size': 18,
            'titles': season_list,
            'titles_size': 16,
            'figsize':(10, 8),
            'contour': use_contour,
            'sym': sym,
            'cbar_label': cbar_label,
            'cmap': self.cmap,
            'loglevel': self.loglevel
        }

        if vmin is not None:
            plot_kwargs['vmin'] = vmin
        if vmax is not None:
            plot_kwargs['vmax'] = vmax

        fig = plot_maps(**plot_kwargs)
        if show_significance and significance_mask is not None:
            map_axes = [ax for ax in fig.axes if hasattr(ax, 'projection')]
            for i, season in enumerate(season_list):
                if i >= len(map_axes):
                    break
                self._overlay_non_significant_points(
                    ax=map_axes[i],
                    significance_mask=significance_mask.sel(season=season),
                    marker_stride=max(1, int(marker_stride)),
                )

        description = (
            f"Seasonal bias map of {data[var].attrs.get('long_name', var)}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp},"
            f" using {data_ref.AQUA_model} as reference data."
            f" The bias is computed for each season over the period from {data.startdate} to {data.enddate} for the model" 
            f" and from {data_ref.startdate} to {data_ref.enddate} for the reference data."
            + (" Crosses mark grid points with non-significant bias (two-tailed test)." if show_significance and significance_mask is not None else "")
        )

        self._save_figure(fig=fig, diagnostic_product='seasonal_bias', data=data, data_ref=data_ref,
                          description=description, var=var, plev=plev, realization=realization)


    def plot_vertical_bias(self, data, data_ref, var, plev_min=None, plev_max=None, 
                           vmin=None, vmax=None, vmin_contour=None, vmax_contour=None, nlevels=18):
        """
        Calculates and plots the vertical bias between two datasets.

        Args:
            data (xarray.Dataset): Dataset to analyze.
            data_ref (xarray.Dataset): Reference dataset for comparison.
            var (str): Variable name to analyze.
            plev_min (float, optional): Minimum pressure level.
            plev_max (float, optional): Maximum pressure level.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            vmin_contour (float, optional): Minimum contour value.
            vmax_contour (float, optional): Maximum contour value.
            nlevels (int, optional): Number of contour levels for the plot.
        """
        self.logger.info('Plotting vertical biases for variable: %s', var)

        realization = get_realizations(data)

        title = (
            f"Vertical bias of {data[var].attrs.get('long_name', var)} for {data.AQUA_model} {data.AQUA_exp}\n"
            f"relative to {data_ref.AQUA_model} climatology\n"
        )

        description = (
            f"Vertical bias plot of {data[var].attrs.get('long_name', var)} across pressure levels from {data.startdate} to {data.enddate}"
            f" for the {data.AQUA_model} model, experiment {data.AQUA_exp}, with {data_ref.AQUA_model} from {data_ref.startdate} to {data_ref.enddate}"
            f" used as reference data."
        )

        fig, ax = plot_vertical_profile_diff(
            data=data[var].mean(dim='lon'),
            data_ref=data_ref[var].mean(dim='lon'),
            var=var,
            lev_min=plev_min,
            lev_max=plev_max,
            vmin=vmin,
            vmax=vmax,
            vmin_contour=vmin_contour,
            vmax_contour=vmax_contour,
            logscale=True,
            add_contour=True, 
            cmap=self.cmap,
            nlevels=nlevels,
            title=title,
            title_size=18,
            return_fig=True,
            loglevel=self.loglevel
        )

        self._save_figure(fig=fig, diagnostic_product='vertical_bias', data=data, data_ref=data_ref,
                          description=description, var=var, realization=realization)

        self.logger.info("Vertical bias plot completed successfully.")
