import argparse
import os
import sys

def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Seasonal Cycle Diagnostic CLI")
    parser.add_argument("-c", "--config", required=True, help="Path to configuration file")
    parser.add_argument("-l", "--loglevel", default="WARNING", help="Log level")
    parser.add_argument("--save-fig", default=True, action="store_true", help="Save generated figures")
    parser.add_argument("--save-netcdf", default=True, action="store_true", help="Save processed data")
    return parser.parse_args(args)

def main():
    args = parse_arguments(sys.argv[1:])

    # Add the diagnostic directory to the Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diagnostic_path = os.path.join(script_dir, "../")
    sys.path.insert(0, diagnostic_path)

    from seasonal_cycle import SeasonalCycle
    from aqua.core.logger import log_configure
    from aqua.core.util import load_yaml
    from aqua import __version__ as aqua_version

    # Set the log 
    loglevel = args.loglevel
    logger = log_configure(log_name="Seasonal Cycle CLI", log_level=loglevel)
    logger.info("Running aqua version %s", aqua_version)

    config_file = args.config
    logger.info("Reading configuration from %s", config_file)

    if not os.path.exists(config_file):
        logger.error("Configuration file not found: %s", config_file)
        sys.exit(1)

    try:
        config = load_yaml(config_file)
    except Exception as exc:
        logger.error("Failed to load configuration %s: %s", config_file, exc)
        sys.exit(1)

    # Run the seasonal cycle diagnostic
    try:
        diagnostic = SeasonalCycle(config=config, loglevel=loglevel)
        diagnostic.retrieve()
        diagnostic.compute(save_fig=args.save_fig, save_data=args.save_netcdf)
        logger.info("Seasonal cycle diagnostic finished successfully")
    except Exception as exc:
        logger.error("Seasonal cycle diagnostic failed: %s", exc, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

