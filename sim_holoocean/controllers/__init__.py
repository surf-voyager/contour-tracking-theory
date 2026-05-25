"""Layer-2 (HoloOcean) contour-tracking controllers.

This package re-implements the four Layer-1 controllers (los_controller,
curvature_estimator, mode_manager, reacquire_planner) without importing from the
Layer-1 sim_python package. The logic (LOS saturated form, the hard-coded 720
variance constant, the T*_N / T*_G closed forms, the χ²-gate FSM, the
Archimedean-spiral re-acquire) is identical to the Layer-1 originals so that the
two engines compare the SAME controller; only the input adapter differs (HoloOcean
ImagingSonar / DynamicsSensor state instead of the Layer-1 2D ray-cast).
"""
from __future__ import annotations

from .los_controller import LOSConfig, compute_r_star
from .curvature_estimator import CurvatureEstimator
from .mode_manager import (
    Mode,
    ModeConfig,
    ModeManager,
    beta_N,
    chi2_threshold,
    compute_T_star_G,
    compute_T_star_N,
)
from .reacquire_planner import (
    Pose2D,
    ReacquireConfig,
    ReacquirePlanner,
    coverage_rate,
)

__all__ = [
    "LOSConfig",
    "compute_r_star",
    "CurvatureEstimator",
    "Mode",
    "ModeConfig",
    "ModeManager",
    "beta_N",
    "chi2_threshold",
    "compute_T_star_G",
    "compute_T_star_N",
    "Pose2D",
    "ReacquireConfig",
    "ReacquirePlanner",
    "coverage_rate",
]
