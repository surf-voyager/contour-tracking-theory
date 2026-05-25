"""Tests for sim_python.controllers.los_controller.

Covers the LOS + curvature feed-forward law r^* = κ̂ · û + r^*_LOS,
including the saturated sin form of Caharija 2016 (IEEE TCST).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.controllers.los_controller import LOSConfig, compute_r_star


# --------------------------------------------------------------------------- #
# Test 1 — on-track ⇒ r* = 0                                                   #
# --------------------------------------------------------------------------- #


def test_on_track_zero_command() -> None:
    """When d=d*, θ=0, κ̂=0 → r* must be exactly 0 (no command)."""
    cfg = LOSConfig(Delta=2.0, K_p=0.5, d_star=5.0)
    r = compute_r_star(d=5.0, theta=0.0, kappa_hat=0.0, u=1.5, cfg=cfg)
    assert r == 0.0, f"on-track r* = {r}, expected 0"


# --------------------------------------------------------------------------- #
# Test 2 — sin-style saturation: |r*_LOS| asymptotes to K_p                   #
# --------------------------------------------------------------------------- #


def test_sin_style_saturation_to_K_p() -> None:
    """As |d-d*| → ∞ the sin-style LOS saturates to ∓K_p (Caharija 2016)."""
    cfg = LOSConfig(Delta=2.0, K_p=0.7, d_star=0.0, style="sin")
    # Very large positive error → r* → -K_p
    r_pos = compute_r_star(d=1e6, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r_pos - (-cfg.K_p)) < 1e-6, f"+∞ saturation r* = {r_pos}"
    # Very large negative error → r* → +K_p
    r_neg = compute_r_star(d=-1e6, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r_neg - (+cfg.K_p)) < 1e-6, f"-∞ saturation r* = {r_neg}"


# --------------------------------------------------------------------------- #
# Test 3 — atan2-style saturation to ±K_p·π/2                                 #
# --------------------------------------------------------------------------- #


def test_atan2_style_saturation_to_pi_over_2_K_p() -> None:
    cfg = LOSConfig(Delta=1.0, K_p=0.3, d_star=0.0, style="atan2")
    r_pos = compute_r_star(d=1e9, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    expected = -cfg.K_p * (np.pi / 2.0)
    assert abs(r_pos - expected) < 1e-6, f"atan2 +∞ r* = {r_pos}, expected {expected}"
    r_neg = compute_r_star(d=-1e9, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r_neg - (-expected)) < 1e-6


# --------------------------------------------------------------------------- #
# Test 4 — feed-forward κ̂·û component                                        #
# --------------------------------------------------------------------------- #


def test_feed_forward_term_matches_kappa_u() -> None:
    """On-track + K_ff=1: r* = κ̂ · u (pure feed-forward)."""
    cfg = LOSConfig(Delta=2.0, K_p=0.5, K_ff=1.0, d_star=0.0)
    kappa_hat = 0.1
    u = 2.0
    r = compute_r_star(d=0.0, theta=0.0, kappa_hat=kappa_hat, u=u, cfg=cfg)
    assert abs(r - kappa_hat * u) < 1e-12, f"r* = {r}, expected {kappa_hat * u}"


# --------------------------------------------------------------------------- #
# Test 5 — input validation                                                   #
# --------------------------------------------------------------------------- #


def test_input_validation() -> None:
    cfg = LOSConfig(Delta=2.0, K_p=0.5)
    with pytest.raises(ValueError, match="d "):
        compute_r_star(d=float("nan"), theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    with pytest.raises(ValueError, match="u "):
        compute_r_star(d=0.0, theta=0.0, kappa_hat=0.0, u=float("inf"), cfg=cfg)
    # Config validation
    with pytest.raises(ValueError, match="Delta"):
        LOSConfig(Delta=0.0, K_p=0.5)
    with pytest.raises(ValueError, match="K_p"):
        LOSConfig(Delta=1.0, K_p=-1.0)
    with pytest.raises(ValueError, match="style"):
        LOSConfig(Delta=1.0, K_p=0.5, style="bogus")


# --------------------------------------------------------------------------- #
# Test 6 — sign convention: e > 0 ⇒ r* < 0 (sin-style, K_ff=0, K_θ=0)         #
# --------------------------------------------------------------------------- #


def test_sign_convention_pulls_back_to_dstar() -> None:
    """e = d - d* > 0 ⇒ LOS term must pull d down toward d* (r* < 0)."""
    cfg = LOSConfig(Delta=2.0, K_p=0.5, d_star=5.0)
    r_above = compute_r_star(d=8.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    r_below = compute_r_star(d=2.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert r_above < 0, f"d>d* should give r*<0; got {r_above}"
    assert r_below > 0, f"d<d* should give r*>0; got {r_below}"
    # Symmetric magnitudes for symmetric errors
    assert abs(r_above + r_below) < 1e-12


# --------------------------------------------------------------------------- #
# Test 7 — superposition: r* = K_ff · κ̂ · u + r*_LOS + (-K_θ · θ)             #
# --------------------------------------------------------------------------- #


def test_superposition_with_theta_correction() -> None:
    cfg = LOSConfig(Delta=2.0, K_p=0.5, K_ff=1.0, K_theta=0.4, d_star=0.0)
    r = compute_r_star(d=1.0, theta=0.2, kappa_hat=0.05, u=1.5, cfg=cfg)
    # Manual reconstruction
    e = 1.0
    r_los = -cfg.K_p * e / np.sqrt(e * e + cfg.Delta ** 2)
    r_ff = cfg.K_ff * 0.05 * 1.5
    r_th = -cfg.K_theta * 0.2
    expected = r_ff + r_los + r_th
    assert abs(r - expected) < 1e-12
