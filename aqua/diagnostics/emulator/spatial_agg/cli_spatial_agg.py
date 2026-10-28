"""CLI for spatial aggregation."""
from .spatial_agg import SpatialAgg
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(SpatialAgg, "spatial_agg")
