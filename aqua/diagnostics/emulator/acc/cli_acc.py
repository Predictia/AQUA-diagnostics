"""CLI for ACC."""
from .acc import ACC
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(ACC, "acc")
