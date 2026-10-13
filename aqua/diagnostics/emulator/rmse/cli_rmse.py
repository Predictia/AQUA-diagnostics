"""CLI for RMSE."""
from .rmse import RMSE
from .._cli import parse_arguments, run_cli

if __name__ == "__main__":
    run_cli(RMSE, "rmse")
