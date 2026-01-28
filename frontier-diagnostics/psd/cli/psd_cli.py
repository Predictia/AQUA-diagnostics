import os
import sys
import argparse

# Add the directory containing the diagnostic module to the Python path.
script_dir = os.path.dirname(os.path.abspath(__file__))
diagnostic_module_path = os.path.join(script_dir, "../")
sys.path.insert(0, diagnostic_module_path)

def parse_arguments(args):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description='PSD Diagnostic CLI')

    # Require configuration file
    parser.add_argument('-c', '--config', type=str,
                        help='YAML configuration file for the PSD diagnostic',
                        required=True)

    # Keep loglevel argument
    parser.add_argument('-l', '--loglevel', type=str, default='WARNING',
                        help='Log level [default: WARNING]')

    # Add flags for saving outputs
    parser.add_argument('--save-fig', action='store_true',
                        help='Save output figures')
    parser.add_argument('--save-netcdf', action='store_true',
                        help='Save output NetCDF data')

    return parser.parse_args(args)

if __name__ == '__main__':

    print('Running PSD Diagnostic CLI')
    args = parse_arguments(sys.argv[1:])

    try:
        from psd import PSD
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
        from aqua import __version__ as aqua_version

        loglevel = args.loglevel
        logger = log_configure(log_name='PSD CLI', log_level=loglevel)
    except ImportError as e:
        print(f'Failed to import aqua or PSD diagnostic: {e}')
        print("Ensure the AQUA package and the PSD diagnostic are correctly installed/discoverable.")
        sys.exit(1)
    except Exception as e:
        print(f'Failed during initial setup: {e}')
        sys.exit(1)

    logger.info(f'Running aqua version {aqua_version}')

    # Change the current directory to the one of the CLI so that relative paths work
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    if os.getcwd() != dname:
        logger.info(f'Changing current directory to {dname} to run!')
        os.chdir(dname)

    config_file = args.config
    logger.info(f'Reading configuration from {config_file}')

    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    try:
        config = load_yaml(config_file)
    except Exception as e:
        logger.error(f"Failed to load or parse configuration file {config_file}: {e}")
        sys.exit(1)

    # Run the PSD diagnostic
    try:
        logger.info("Initializing PSD diagnostic...")
        psd_diagnostic = PSD(config=config, loglevel=loglevel)

        logger.info("Retrieving data...")
        psd_diagnostic.retrieve()

        logger.info("Calculating and plotting PSD...")
        psd_diagnostic.calculate_and_plot_psd(save_fig=args.save_fig, save_netcdf=args.save_netcdf)

        logger.info('PSD diagnostic finished successfully!')

    except ValueError as ve:
         logger.error(f"Configuration error in {config_file}: {ve}")
         sys.exit(1)
    except FileNotFoundError as fnf_err:
        logger.error(f"Data file not found during retrieval: {fnf_err}")
        logger.critical("Check paths and availability in the data catalog specified in the config.")
        sys.exit(1)
    except Exception as e:
        logger.error('An error occurred during PSD diagnostic execution: {}'.format(e), exc_info=True)
        logger.critical('PSD diagnostic failed.')
        sys.exit(1)
