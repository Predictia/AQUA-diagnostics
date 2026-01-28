import argparse
import os
import sys

# Add the directory containing the diagnostic module to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Atlantic Multidecadal Oscillation (AMO) diagnostic"
    )
    parser.add_argument("-c", "--config", required=True, help="YAML configuration file")
    parser.add_argument("-l", "--loglevel", default="WARNING", help="Log level")
    return parser.parse_args(args)


def main(raw_args):
    args = parse_args(raw_args)

    try:
        from amo import AMO
        from aqua import __version__ as aqua_version
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
    except ImportError as e:
        print(f"Failed to import aqua or AMO diagnostic: {e}")
        print("Ensure the AQUA package and the AMO diagnostic are correctly installed/discoverable.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed during initial setup: {e}")
        sys.exit(1)

    loglevel = args.loglevel
    logger = log_configure(log_name="AMO CLI", log_level=loglevel)
    logger.info("Running aqua version %s", aqua_version)

    # Change the current directory to the one of the CLI so that relative paths work
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    if os.getcwd() != dname:
        logger.info("Changing current directory to %s to run!", dname)
        os.chdir(dname)

    config_file = args.config
    logger.info("Reading configuration from %s", config_file)

    if not os.path.exists(config_file):
        logger.error("Configuration file not found: %s", config_file)
        sys.exit(1)

    try:
        config = load_yaml(config_file)
    except Exception as e:
        logger.error("Failed to load or parse configuration file %s: %s", config_file, e)
        sys.exit(1)

    # Run the AMO diagnostic
    try:
        logger.info("Initializing AMO diagnostic...")
        diagnostic = AMO(config=config, loglevel=loglevel)

        logger.info("Retrieving data...")
        diagnostic.retrieve()

        logger.info("Computing AMO index...")
        diagnostic.compute_index()

        logger.info("Diagnostic finished")
    except ValueError as ve:
        logger.error("Configuration error in %s: %s", config_file, ve)
        sys.exit(1)
    except FileNotFoundError as fnf_err:
        logger.error("Data file not found during retrieval: %s", fnf_err)
        logger.critical("Check paths and availability in the data catalog specified in the config.")
        sys.exit(1)
    except Exception as e:
        logger.error("An error occurred during AMO diagnostic execution: %s", e, exc_info=True)
        logger.critical("AMO diagnostic failed.")
        sys.exit(1)


if __name__ == "__main__":
    print("Running AMO diagnostic CLI")
    main(sys.argv[1:])