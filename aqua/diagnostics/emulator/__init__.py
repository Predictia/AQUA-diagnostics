"""Diagnostics for evaluating emulator predictions against reference data."""

from .base import EmulatorDiagnostic, EmulatorPlotMixin
from .rmse import RMSE, PlotRMSE
from .ssim import SSIM, PlotSSIM
from .acc import ACC, PlotACC
from .cov import COV, PlotCOV
from .psd import PSD, PlotPSD
from .rollout import Rollout, PlotRollout
from .spatiotemporal_rollout import (
    SpatiotemporalRollout,
    PlotSpatiotemporalRollout,
)
from .variability import Variability, PlotVariability
from .spatial_agg import SpatialAgg, PlotSpatialAgg
from .seasonal_cycle import SeasonalCycle, PlotSeasonalCycle
from .ipo import IPO, PlotIPO
from .amo import AMO, PlotAMO
from .vertical_zonal_mean import VerticalZonalMean, PlotVerticalZonalMean
from .zonal_mean_profile import ZonalMeanProfile, PlotZonalMeanProfile

__all__ = [
    "EmulatorDiagnostic",
    "EmulatorPlotMixin",
    "RMSE",
    "PlotRMSE",
    "SSIM",
    "PlotSSIM",
    "ACC",
    "PlotACC",
    "COV",
    "PlotCOV",
    "PSD",
    "PlotPSD",
    "Rollout",
    "PlotRollout",
    "SpatiotemporalRollout",
    "PlotSpatiotemporalRollout",
    "Variability",
    "PlotVariability",
    "SpatialAgg",
    "PlotSpatialAgg",
    "SeasonalCycle",
    "PlotSeasonalCycle",
    "IPO",
    "PlotIPO",
    "AMO",
    "PlotAMO",
    "VerticalZonalMean",
    "PlotVerticalZonalMean",
    "ZonalMeanProfile",
    "PlotZonalMeanProfile",
]
