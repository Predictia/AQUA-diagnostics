import os
import sys
import argparse

# Add the directory containing the diagnostic module to the Python path
# (First position in the sys.path list)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, DIAG_ROOT)


def parse_arguments(argv):
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Rollout Diagnostic CLI")

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
        help="Save rollout time series as NetCDF files",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    try:
        from rollout import Rollout
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
        from aqua import __version__ as aqua_version
    except ImportError as exc:
        print(f"Failed to import required modules: {exc}")
        print(f"Diagnostic module path: {DIAG_ROOT}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

    logger = log_configure(log_name="Rollout CLI", log_level=args.loglevel)
    logger.info("Running aqua version %s", aqua_version)

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

    diag = Rollout(config=config, loglevel=args.loglevel)
    diag.retrieve()

    save_fig = not args.no_save_fig
    save_data = args.save_data

    outputs = diag.compute_rollout(save_fig=save_fig, save_data=save_data)
    
    if outputs:
        for key, path in outputs.items():
            logger.info("Processed rollout for %s", key)
            if save_data and path is not None:
                logger.info("Saved NetCDF for %s to %s", key, path)
    else:
        logger.warning("No rollout outputs were generated.")


if __name__ == "__main__":
    main()


