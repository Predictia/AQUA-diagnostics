import argparse
import os
import sys

# Add the directory containing the diagnostic module to the Python path
# (First position in the sys.path list)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, DIAG_ROOT)


def parse_arguments(argv):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Spatiotemporal Rollout Diagnostic CLI")

    parser.add_argument("-c",
                        "--config",
                        type=str,
                        required=True,
                        help="Path to the diagnostic YAML configuration file")

    parser.add_argument("-l",
                        "--loglevel",
                        type=str,
                        default="WARNING",
                        help="Log level (default: WARNING)")

    parser.add_argument("--save-videos",
                        action="store_true",
                        help="Generate and save rollout videos")

    return parser.parse_args(argv)


def main(argv=None):
    
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    try:
        from spatiotemporal_rollout import SpatiotemporalRollout
        from aqua.core.logger import log_configure
        from aqua.core.util import load_yaml
    except ImportError as exc:
        print(f"Failed to import required modules: {exc}")
        print(f"Diagnostic module path: {DIAG_ROOT}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

    logger = log_configure(log_name="SpatiotemporalRollout CLI", log_level=args.loglevel)

    config_path = args.config if os.path.isabs(args.config) else os.path.join(SCRIPT_DIR, args.config)
    if not os.path.exists(config_path):
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        config = load_yaml(config_path)
    except Exception as exc:
        logger.error("Failed to load configuration file %s: %s", config_path, exc)
        sys.exit(1)

    diag = SpatiotemporalRollout(config=config, loglevel=args.loglevel)
    diag.retrieve()

    if args.save_videos:
        outputs = diag.generate_videos()
        if outputs:
            for key, path in outputs.items():
                logger.info("Generated video for %s: %s", key, path)
        else:
            logger.warning("No videos were generated.")
    else:
        logger.info("Data retrieval completed. Use --save-videos to generate outputs.")


if __name__ == "__main__":
    main()

