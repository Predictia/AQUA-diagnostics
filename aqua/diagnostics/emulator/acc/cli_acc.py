"""CLI for ACC."""
from .acc import ACC
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(ACC, "acc")
