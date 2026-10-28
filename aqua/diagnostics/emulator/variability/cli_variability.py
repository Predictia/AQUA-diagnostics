"""CLI for variability."""
from .variability import Variability
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(Variability, "variability")
