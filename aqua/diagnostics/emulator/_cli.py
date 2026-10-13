"""Shared command-line runner for emulator diagnostics."""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments

from .acc.plot_acc import PlotACC
from .amo.plot_amo import PlotAMO
from .cov.plot_cov import PlotCOV
from .ipo.plot_ipo import PlotIPO
from .psd.plot_psd import PlotPSD
from .rmse.plot_rmse import PlotRMSE
from .rollout.plot_rollout import PlotRollout
from .seasonal_cycle.plot_seasonal_cycle import PlotSeasonalCycle
from .spatial_agg.plot_spatial_agg import PlotSpatialAgg
from .spatiotemporal_rollout.plot_spatiotemporal_rollout import (
    PlotSpatiotemporalRollout,
)
from .ssim.plot_ssim import PlotSSIM
from .variability.plot_variability import PlotVariability
from .vertical_zonal_mean.plot_vertical_zonal_mean import PlotVerticalZonalMean
from .zonal_mean_profile.plot_zonal_mean_profile import PlotZonalMeanProfile


def parse_arguments(args=None, description="Emulator diagnostic CLI"):
    """Parse AQUA's standard diagnostic command-line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser = template_parse_arguments(parser)
    return parser.parse_args(sys.argv[1:] if args is None else args)


def _reference_kwargs(reference):
    return {
        "ref_catalog": reference.get("catalog"),
        "ref_model": reference.get("model"),
        "ref_exp": reference.get("exp"),
        "ref_source": reference.get("source"),
        "ref_regrid": reference.get("regrid"),
    }


def _diagnostic_options(name, tool):
    options = {}
    if "variables" in tool:
        options["variables"] = tool["variables"]
    params = dict(tool.get("params", {}).get("default", {}))
    if name == "ssim":
        options.update(
            {
                key: params[key]
                for key in ("win_size", "sigma", "data_range")
                if key in params
            }
        )
    elif name == "acc":
        climatology = tool.get("climatology", {})
        for key in (
            "model", "exp", "source", "catalog", "regrid", "startdate", "enddate"
        ):
            if key in climatology:
                options[f"clim_{key}"] = climatology[key]
    elif name == "cov":
        options["variable_pairs"] = tool.get("variable_pairs", [])
    elif name == "psd":
        calculations = tool.get("psd_calculations", {})
        options["time_averaged_psd"] = calculations.get(
            "time_averaged_psd", True
        )
        options["psd_of_time_mean"] = calculations.get(
            "psd_of_time_mean", True
        )
    elif name == "rollout":
        options["annual"] = tool.get("annual", False)
    elif name == "seasonal_cycle":
        options["regions"] = tool.get("regions")
    elif name == "ipo":
        options.update(
            {
                key: tool[key]
                for key in ("field", "regions", "window")
                if key in tool
            }
        )
        filter_cfg = tool.get("filter", {})
        options.update(
            {
                "filter_type": filter_cfg.get("type", "chebyshev"),
                "cutoff_freq": filter_cfg.get("cutoff_freq"),
                "filter_order": filter_cfg.get("order", 4),
                "ripple_db": filter_cfg.get("ripple_db", 1.0),
            }
        )
    elif name == "amo":
        options["field"] = tool.get("field", "tos")
        amo = tool.get("amo", {})
        options.update(
            {
                "region": amo.get("region"),
                "detrend": amo.get("detrend", True),
                "detrend_variable": amo.get("detrend_variable", "tos"),
                "smooth": amo.get("smooth", True),
                "window": amo.get("window", 5),
            }
        )
    elif name in {"vertical_zonal_mean", "zonal_mean_profile"}:
        options["aggregation"] = tool.get("aggregation", "mean")
        if name == "vertical_zonal_mean":
            options["latitude_bins"] = tool.get("latitude_bins")
    return options


def _spatial_temporal(label):
    def draw(plotter, results, _diagnostic, _tool, save):
        for key, data in results["spatial"].items():
            yield plotter.plot_spatial(
                data, title=f"Spatial {label} {key}",
                extra_keys={"var": key}, **save
            )[0]
        for key, data in results["temporal"].items():
            yield plotter.plot_temporal(
                data, title=f"Temporal {label} {key}",
                extra_keys={"var": key}, **save
            )[0]
    return draw


def _draw_three_maps(plotter, results, _diagnostic, _tool, save):
    for key, data in results.items():
        yield plotter.plot(data, extra_keys={"var": key}, **save)[0]


def _draw_cov(plotter, results, _diagnostic, _tool, save):
    for pair, data in results.items():
        yield plotter.plot_three_maps(
            data.correlation_reference,
            data.correlation_prediction,
            data.correlation_difference,
            product="correlation",
            extra_keys={"pair": pair},
            **save,
        )[0]


def _draw_psd(plotter, results, _diagnostic, _tool, save):
    for key, products in results.items():
        for product, data in products.items():
            yield plotter.plot(
                data, title=f"{product} {key}", product=product,
                extra_keys={"var": key}, **save
            )[0]


def _draw_rollout(plotter, results, _diagnostic, _tool, save):
    for key, data in results.items():
        annual = key.endswith("_annual")
        var = key[: -len("_annual")] if annual else key
        yield plotter.plot(
            data, title=key, product="annual" if annual else "timeseries",
            extra_keys={"var": var}, **save
        )[0]


def _draw_seasonal_cycle(plotter, results, _diagnostic, _tool, save):
    for data in results.values():
        var, region = data.attrs["variable"], data.attrs["region"]
        yield plotter.plot(
            data, title=f"Seasonal cycle {var} ({region})",
            extra_keys={"var": var, "region": region}, **save
        )[0]


def _draw_index(title):
    def draw(plotter, results, diagnostic, _tool, save):
        yield plotter.plot_index(
            results, title=title, extra_keys={"var": diagnostic.field}, **save
        )[0]
    return draw


def _draw_zonal(plotter, results, diagnostic, _tool, save):
    for var, data in results.items():
        yield plotter.plot(
            data, title=f"{var} ({diagnostic.aggregation_stat})",
            extra_keys={"var": var, "stat": diagnostic.aggregation_stat},
            **save,
        )[0]


def _draw_animation(plotter, results, _diagnostic, tool, _save):
    video = tool.get("video", {})
    for key, data in results.items():
        plotter.plot(
            data, title=key, fps=video.get("fps", 3), dpi=video.get("dpi", 150),
            extra_keys={"var": key},
        )
    return ()


FIGURES = {
    "rmse": (PlotRMSE, _spatial_temporal("RMSE")),
    "ssim": (PlotSSIM, _spatial_temporal("SSIM")),
    "acc": (PlotACC, _spatial_temporal("ACC")),
    "cov": (PlotCOV, _draw_cov),
    "psd": (PlotPSD, _draw_psd),
    "rollout": (PlotRollout, _draw_rollout),
    "spatiotemporal_rollout": (PlotSpatiotemporalRollout, _draw_animation),
    "variability": (PlotVariability, _draw_three_maps),
    "spatial_agg": (PlotSpatialAgg, _draw_three_maps),
    "seasonal_cycle": (PlotSeasonalCycle, _draw_seasonal_cycle),
    "ipo": (PlotIPO, _draw_index("IPO tripole index")),
    "amo": (PlotAMO, _draw_index("AMO index")),
    "vertical_zonal_mean": (PlotVerticalZonalMean, _draw_zonal),
    "zonal_mean_profile": (PlotZonalMeanProfile, _draw_zonal),
}


def save_figures(name, diagnostic, results, tool, *, save_format, dpi):
    """Draw and save every figure of one diagnostic run through OutputSaver."""
    plot_class, draw = FIGURES[name]
    plotter = plot_class.from_diagnostic(diagnostic)
    save = {"save": True, "format": save_format, "dpi": dpi}
    for fig in draw(plotter, results, diagnostic, tool, save):
        plt.close(fig)


def run_cli(diagnostic_class, name, argv=None):
    """Run one emulator diagnostic from an AQUA collection config."""
    args = parse_arguments(argv, f"{name} emulator diagnostic")
    cli = DiagnosticCLI(
        args,
        diagnostic_name="emulator",
        default_config="config-emulator-cesm2.yaml",
        log_name=f"{name} CLI",
    ).prepare()
    cli.open_dask_cluster()
    try:
        tool = cli.config_dict.get("diagnostics", {}).get(name, {})
        if not tool.get("run", False):
            cli.logger.info("%s is disabled in configuration", name)
            return {}
        if not cli.config_dict.get("datasets") or not cli.config_dict.get(
            "references"
        ):
            raise ValueError("emulator diagnostics require datasets and references")
        dataset = cli.config_dict["datasets"][0]
        reference = cli.config_dict["references"][0]
        kwargs = cli.dataset_args(dataset)
        kwargs.update(_reference_kwargs(reference))
        kwargs.update(
            {
                "diagnostic_name": name,
                "outputdir": cli.outputdir,
                "rebuild": cli.rebuild,
                "loglevel": cli.loglevel,
            }
        )
        kwargs.update(_diagnostic_options(name, tool))
        diagnostic = diagnostic_class(**kwargs)
        results = diagnostic.run(
            reader_kwargs=dataset.get("reader_kwargs"),
            ref_reader_kwargs=reference.get("reader_kwargs"),
            save_netcdf=cli.save_netcdf,
        )
        if cli.save_format:
            save_figures(
                name, diagnostic, results, tool,
                save_format=cli.save_format, dpi=cli.dpi,
            )
        return results
    finally:
        cli.close_dask_cluster()
