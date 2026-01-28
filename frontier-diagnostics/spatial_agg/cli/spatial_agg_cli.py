import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, DIAG_ROOT)


def parse_arguments(argv):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Spatial Aggregation Diagnostic CLI")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the diagnostic YAML configuration file",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        type=str,
        default="WARNING",
        help="Log level (default: WARNING)",
    )
    parser.add_argument( # Figures are saved by default
        "--no-save-fig",
        action="store_true",
        help="Do not save figures to disk",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save climatology fields as NetCDF files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    try:
        from spatial_agg import SpatialAgg
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
        from aqua import __version__ as aqua_version
    except ImportError as exc:
        print(f"Failed to import required modules: {exc}")
        print(f"Diagnostic module path: {DIAG_ROOT}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

    logger = log_configure(log_name="SpatialAgg CLI", log_level=args.loglevel)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)

    if not os.path.exists(config_path):
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        config = load_yaml(config_path)
    except Exception as exc:
        logger.error("Failed to load configuration file %s: %s", config_path, exc)
        sys.exit(1)

    diag = SpatialAgg(config=config, loglevel=args.loglevel)
    diag.retrieve()

    save_fig = not args.no_save_fig
    save_data = args.save_data

    outputs = diag.compute_climatology(save_fig=save_fig, save_data=save_data)
    if outputs:
        for key in outputs:
            logger.info("Processed climatology for %s", key)
    else:
        logger.warning("No climatology outputs were generated.")


if __name__ == "__main__":
    main()


