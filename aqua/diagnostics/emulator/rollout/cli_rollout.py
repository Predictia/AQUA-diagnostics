"""CLI for rollout."""
from .rollout import Rollout
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(Rollout, "rollout")
