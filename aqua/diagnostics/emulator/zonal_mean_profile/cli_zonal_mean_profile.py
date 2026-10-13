"""CLI for zonal mean profile."""
from .zonal_mean_profile import ZonalMeanProfile
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(ZonalMeanProfile, "zonal_mean_profile")
