"""Command-line interface for GlobalBiases diagnostic."""

import argparse
import sys

from aqua.core.util import to_list
from aqua.core.exceptions import NoDataError
from aqua.diagnostics.base import template_parse_arguments
from aqua.diagnostics.base import DiagnosticCLI
try:
    from .global_biases import GlobalBiases
    from .plot_global_biases import PlotGlobalBiases
except ImportError:
    # Allow execution as a standalone script from this directory.
    from global_biases import GlobalBiases
    from plot_global_biases import PlotGlobalBiases

TOOLNAME='GlobalBiases'
TOOLNAME_KEY = TOOLNAME.lower()

def parse_arguments(args):
    """Parse command-line arguments for GlobalBiases diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description=f'{TOOLNAME} CLI')
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)

if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    
    cli = DiagnosticCLI(
        args,
        diagnostic_name=TOOLNAME_KEY,
        default_config='config_global_biases.yaml',
    )
    cli.prepare()
    cli.open_dask_cluster()

    # Retrieve tool-specific configuration
    tool_dict = cli.config_dict['diagnostics'].get(TOOLNAME_KEY, {})
    # Global Biases diagnostic
    if tool_dict and tool_dict.get('run', False):
        cli.logger.info(f"{TOOLNAME} diagnostic is enabled.")

        if len(cli.config_dict['datasets']) > 1:
            cli.logger.warning(
                "Only the first entry in 'datasets' will be used.\n"
                "Multiple datasets are not supported by this diagnostic."
            )
        if len(cli.config_dict['references']) > 1:
            cli.logger.warning(
                "Only the first entry in 'references' will be used.\n"
                "Multiple references are not supported by this diagnostic."
            )
        diagnostic_name = tool_dict.get('diagnostic_name', TOOLNAME_KEY)
        dataset = cli.config_dict['datasets'][0]
        reference = cli.config_dict['references'][0]
        dataset_args = cli.dataset_args(dataset)
        reference_args = cli.reference_args(reference)

        variables = tool_dict.get('variables', [])
        formulae = tool_dict.get('formulae', [])
        plev = tool_dict.get('params', {}).get('default', {}).get('plev')
        seasons = tool_dict.get('params', {}).get('default', {}).get('seasons', False)
        seasons_stat = tool_dict.get('params', {}).get('default', {}).get('seasons_stat', 'mean')
        vertical = tool_dict.get('params', {}).get('default', {}).get('vertical', False)
        alpha = tool_dict.get('params', {}).get('default', {}).get('alpha', 0.05)
        use_student_t = tool_dict.get('params', {}).get('default', {}).get('use_student_t', True)
        show_significance = tool_dict.get('params', {}).get('default', {}).get('show_significance', True)
        marker_stride = tool_dict.get('params', {}).get('default', {}).get('marker_stride', 1)

        cli.logger.debug("Selected levels for vertical plots: %s", plev)

        biases_dataset = GlobalBiases(**dataset_args, diagnostic=diagnostic_name,
                                        outputdir=cli.outputdir, loglevel=cli.loglevel)
        biases_reference = GlobalBiases(**reference_args, diagnostic=diagnostic_name,
                                        outputdir=cli.outputdir, loglevel=cli.loglevel)

        all_vars = [(v, False) for v in variables] + [(f, True) for f in formulae]

        for var, is_formula in all_vars:
            cli.logger.info("Running Global Biases diagnostic for %s: %s",
                        "formula" if is_formula else "variable", var)
            all_plot_params = tool_dict.get('plot_params', {})
            default_params = all_plot_params.get('default', {})
            var_params = all_plot_params.get(var, {})
            plot_params = {**default_params, **var_params}

            vmin, vmax = plot_params.get('vmin'), plot_params.get('vmax')
            param_dict = tool_dict.get('params', {}).get(var, {})
            units = param_dict.get('units', None)
            long_name = param_dict.get('long_name', None)
            short_name = param_dict.get('short_name', None)
            var_name = short_name if short_name is not None else var

            try:
                biases_dataset.retrieve(var=var, units=units, formula=is_formula,
                                        long_name=long_name, short_name=short_name,
                                        reader_kwargs=cli.reader_kwargs)
                biases_reference.retrieve(var=var, units=units, formula=is_formula,
                                        long_name=long_name, short_name=short_name)
            except (NoDataError, KeyError, ValueError) as e:
                cli.logger.warning("Variable '%s' not found in dataset. Skipping. (%s)", var, e)
                continue

            annual_significance = biases_dataset.compute_bias_significance(
                data=biases_dataset.data,
                data_ref=biases_reference.data,
                var=var_name,
                seasonal=False,
                alpha=alpha,
                use_student_t=use_student_t,
            )
            seasonal_significance = None
            if seasons:
                seasonal_significance = biases_dataset.compute_bias_significance(
                    data=biases_dataset.data,
                    data_ref=biases_reference.data,
                    var=var_name,
                    seasonal=True,
                    alpha=alpha,
                    use_student_t=use_student_t,
                )

            biases_dataset.compute_climatology(seasonal=seasons, seasons_stat=seasons_stat, create_catalog_entry=cli.create_catalog_entry)
            biases_reference.compute_climatology(seasonal=seasons, seasons_stat=seasons_stat)

            if short_name is not None:
                var = short_name

            if 'plev' in biases_dataset.data.get(var, {}).dims and plev:
                plev_list = to_list(plev)
            else:
                plev_list = [None]

            for p in plev_list:
                cli.logger.info(f"Processing variable: {var} at pressure level: {p}" if p else f"Processing variable: {var} at surface level")

                proj = plot_params.get('projection', 'robinson')
                proj_params = plot_params.get('projection_params', {})
                cmap= plot_params.get('cmap', 'RdBu_r')

                cli.logger.debug("Using projection: %s for variable: %s", proj, var)
                plot_biases = PlotGlobalBiases(diagnostic=diagnostic_name, save_pdf=cli.save_pdf, save_png=cli.save_png,
                                            dpi=cli.dpi, outputdir=cli.outputdir, cmap=cmap, loglevel=cli.loglevel)

                annual_mask = annual_significance['is_significant']
                if p is not None and 'plev' in annual_mask.coords:
                    annual_mask = annual_mask.sel(plev=p, method='nearest')

                plot_biases.plot_bias(data=biases_dataset.climatology, data_ref=biases_reference.climatology,
                                        var=var, plev=p,
                                        proj=proj, proj_params=proj_params,
                                        vmin=vmin, vmax=vmax,
                                        significance_mask=annual_mask,
                                        show_significance=show_significance,
                                        marker_stride=marker_stride)
                if seasons:
                    seasonal_mask = seasonal_significance['is_significant']
                    if p is not None and 'plev' in seasonal_mask.coords:
                        seasonal_mask = seasonal_mask.sel(plev=p, method='nearest')
                    plot_biases.plot_seasonal_bias(data=biases_dataset.seasonal_climatology,
                                                    data_ref=biases_reference.seasonal_climatology,
                                                    var=var, plev=p,
                                                    proj=proj, proj_params=proj_params,
                                                    vmin=vmin, vmax=vmax,
                                                    significance_mask=seasonal_mask,
                                                    show_significance=show_significance,
                                                    marker_stride=marker_stride)

            if vertical and 'plev' in biases_dataset.data.get(var, {}).dims:
                cli.logger.debug("Plotting vertical bias for variable:  %s", var)
                vmin_v , vmax_v = plot_params.get('vmin_v'), plot_params.get('vmax_v')
                plot_biases.plot_vertical_bias(data=biases_dataset.climatology, data_ref=biases_reference.climatology,
                                                var=var, vmin=vmin_v, vmax=vmax_v)

    cli.close_dask_cluster()

    cli.logger.info("Global Biases diagnostic completed.")
