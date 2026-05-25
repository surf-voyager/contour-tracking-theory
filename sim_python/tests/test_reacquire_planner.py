"""Tests for sim_python.controllers.reacquire_planner.

Covers Lemma 3's re-acquisition strategy: the ML prediction phase with
its R² gate, and the Archimedean spiral search (Koopman coverage).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.controllers.reacquire_planner import (
    Pose2D,
    ReacquireConfig,
    ReacquirePlanner,
    coverage_rate,
)


# --------------------------------------------------------------------------- #
# Test 1 — spiral pitch constraint: 2π·a ≤ w_FOV                              #
# --------------------------------------------------------------------------- #


def test_spiral_pitch_satisfies_2pi_a_le_w_fov() -> None:
    """Per spec: spiral pitch 2π·a must be ≤ w_FOV (coverage overlap)."""
    for w in (np.pi / 6.0, np.pi / 3.0, np.pi / 2.0, np.pi):
        cfg = ReacquireConfig(w_fov=w, u_min=1.0)
        assert 2.0 * np.pi * cfg.spiral_pitch_a <= w + 1e-12, (
            f"pitch violates 2π·a ≤ w_FOV for w={w}"
        )
        # Construction enforces the constraint
        ReacquirePlanner(cfg=cfg)


# --------------------------------------------------------------------------- #
# Test 2 — coverage rate λ = ū · w_FOV / (2π · ρ̄)                              #
# --------------------------------------------------------------------------- #


def test_coverage_rate_closed_form() -> None:
    w = np.pi / 2.0
    u = 1.5
    rho = 2.0
    expected = u * w / (2.0 * np.pi * rho)
    assert abs(coverage_rate(w, u, rho) - expected) < 1e-12


# --------------------------------------------------------------------------- #
# Test 3 — ML predict phase used when R² ≥ R²_min                              #
# --------------------------------------------------------------------------- #


def test_ml_predict_used_when_R2_high() -> None:
    """R² ≥ R²_min ⇒ first waypoint is the ML-predict point, ahead of pose."""
    cfg = ReacquireConfig(w_fov=np.pi / 2.0, u_min=1.0, R2_min=0.7)
    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=10.0, y=0.0, psi=0.0)
    wps_high = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.9, side="L")
    wps_low = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.5, side="L")
    # High-R² has one extra (ML) waypoint
    assert len(wps_high) == len(wps_low) + 1
    # ML waypoint is ahead of last_pose (along +x for κ̂=0, ψ=0)
    ml_wp = wps_high[0]
    assert ml_wp[0] > last.x, f"ML waypoint {ml_wp} should be ahead of {last.x}"
    assert abs(ml_wp[1] - last.y) < 1e-9


# --------------------------------------------------------------------------- #
# Test 4 — spiral waypoints expand outward (r(φ) monotone in φ)                #
# --------------------------------------------------------------------------- #


def test_spiral_radius_monotone_increasing() -> None:
    """Archimedean spiral r = ρ_0 + a·φ → distance from center monotone."""
    cfg = ReacquireConfig(w_fov=np.pi / 2.0, u_min=1.0)
    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=0.0, y=0.0, psi=0.0)
    wps = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.3, side="L")
    # All waypoints in this case are spiral (R²<R²_min) around (0,0)
    radii = np.array([np.hypot(x, y) for (x, y) in wps])
    # Monotone non-decreasing (strict after the first turn)
    diffs = np.diff(radii)
    assert np.min(diffs) >= -1e-12, (
        f"spiral radii not monotone; min diff = {np.min(diffs)}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — config validation                                                  #
# --------------------------------------------------------------------------- #


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="w_fov"):
        ReacquireConfig(w_fov=-1.0, u_min=1.0)
    with pytest.raises(ValueError, match="R2_min"):
        ReacquireConfig(w_fov=1.0, u_min=1.0, R2_min=1.5)
    with pytest.raises(ValueError, match="rho_bar"):
        coverage_rate(w_fov=1.0, u_min=1.0, rho_bar=0.0)


# --------------------------------------------------------------------------- #
# Test 6 — R² in [0,1] bound enforced                                          #
# --------------------------------------------------------------------------- #


def test_R2_argument_bound() -> None:
    cfg = ReacquireConfig(w_fov=np.pi / 2.0, u_min=1.0)
    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=0.0, y=0.0, psi=0.0)
    with pytest.raises(ValueError, match="R2"):
        p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=1.2, side="L")
    with pytest.raises(ValueError, match="side"):
        p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.5, side="Z")


# --------------------------------------------------------------------------- #
# Test 7 — determinism: same inputs ⇒ same waypoints (no RNG)                  #
# --------------------------------------------------------------------------- #


def test_planner_is_deterministic() -> None:
    cfg = ReacquireConfig(w_fov=np.pi / 2.0, u_min=1.0)
    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=1.0, y=2.0, psi=0.3)
    wps_a = p.plan_waypoints(last_pose=last, kappa_hat=0.05, R2=0.8, side="L")
    wps_b = p.plan_waypoints(last_pose=last, kappa_hat=0.05, R2=0.8, side="L")
    assert wps_a == wps_b
