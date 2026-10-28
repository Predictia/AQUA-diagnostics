"""CLI for IPO."""
from .ipo import IPO
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(IPO, "ipo")
