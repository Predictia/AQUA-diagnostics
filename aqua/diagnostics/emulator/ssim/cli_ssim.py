"""CLI for SSIM."""
from .ssim import SSIM
from .._cli import run_cli

if __name__ == "__main__":
    run_cli(SSIM, "ssim")
