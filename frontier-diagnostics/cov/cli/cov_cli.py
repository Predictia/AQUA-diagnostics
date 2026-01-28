import os
import sys
import argparse

# Add the directory containing the diagnostic module to the Python path.
script_dir = os.path.dirname(os.path.abspath(__file__))
diagnostic_module_path = os.path.join(script_dir, "..")
sys.path.insert(0, os.path.abspath(diagnostic_module_path))

def parse_arguments(args):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description='Covariance (COV) Diagnostic CLI')

    # Required configuration file
    parser.add_argument('-c', '--config', type=str,
                        help='YAML configuration file for the COV diagnostic',
                        required=True)

    # Log level
    parser.add_argument('-l', '--loglevel', type=str, default='WARNING',
                        help='Log level [default: WARNING]')

    # Flags for saving outputs
    parser.add_argument('--save-fig', action='store_true',
                        help='Save output figures')
    parser.add_argument('--save-netcdf', action='store_true',
                        help='Save output NetCDF data')

    return parser.parse_args(args)

if __name__ == '__main__':

    print('Running COV Diagnostic CLI')
    args = parse_arguments(sys.argv[1:])

    try:
        from cov import COV
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
        from aqua import __version__ as aqua_version

        loglevel = args.loglevel
        logger = log_configure(log_name='COV CLI', log_level=loglevel)
    except ImportError as e:
        print(f'Failed to import aqua or COV diagnostic: {e}')
        print("Ensure the AQUA package and the COV diagnostic are correctly installed/discoverable.")
        sys.exit(1)

    logger.info('Running aqua version {}'.format(aqua_version))

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    if os.getcwd() != dname:
        logger.info(f'Changing current directory to {dname} to run!')
        os.chdir(dname)

    config_file = args.config
    logger.info('Reading configuration from {}'.format(config_file))

    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
        
    try:
        config = load_yaml(config_file)
    except Exception as e:
        logger.error(f"Failed to load or parse configuration file {config_file}: {e}")
        sys.exit(1)

    # COV Diagnostic Execution
    try:
        logger.info("Initializing COV diagnostic...")
        cov_diagnostic = COV(config=config, loglevel=loglevel)

        logger.info("Retrieving data...")
        cov_diagnostic.retrieve()

        logger.info("Calculating and plotting covariance...")
        cov_diagnostic.compute_and_plot(save_fig=args.save_fig, save_netcdf=args.save_netcdf)

        logger.info('COV diagnostic finished successfully!')

    except ValueError as ve:
         logger.error(f"Configuration error in {config_file}: {ve}")
         sys.exit(1)
    except Exception as e:
        logger.error('An error occurred during COV diagnostic execution: {}'.format(e), exc_info=True)
        logger.critical('COV diagnostic failed.')
        sys.exit(1)