"""CLI for PSD."""
from .psd import PSD
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(PSD, "psd")
