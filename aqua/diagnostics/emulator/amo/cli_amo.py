"""CLI for AMO."""
from .amo import AMO
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(AMO, "amo")
