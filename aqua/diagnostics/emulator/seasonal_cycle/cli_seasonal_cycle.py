"""CLI for seasonal cycle."""
from .seasonal_cycle import SeasonalCycle
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(SeasonalCycle, "seasonal_cycle")
