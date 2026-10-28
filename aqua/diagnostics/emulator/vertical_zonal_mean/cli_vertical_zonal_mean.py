"""CLI for vertical zonal mean."""
from .vertical_zonal_mean import VerticalZonalMean
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(VerticalZonalMean, "vertical_zonal_mean")
