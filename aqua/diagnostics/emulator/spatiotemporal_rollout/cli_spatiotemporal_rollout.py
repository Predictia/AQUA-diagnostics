"""CLI for spatiotemporal rollout."""
from .spatiotemporal_rollout import SpatiotemporalRollout
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(SpatiotemporalRollout, "spatiotemporal_rollout")
