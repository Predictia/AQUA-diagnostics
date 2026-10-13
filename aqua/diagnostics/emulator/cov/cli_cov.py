"""CLI for COV."""
from .cov import COV
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(COV, "cov")
