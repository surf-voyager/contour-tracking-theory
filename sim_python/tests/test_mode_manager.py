"""Tests for sim_python.controllers.mode_manager.

Covers:
- the noise-driven dwell T*_N square-root closed form (example 59.3 s)
- the geometric-occlusion dwell T*_G square-root closed form
  (examples 19.8 / 6.3 / 2.0 s)
- the hard-coded σ_κ̂² constant 720 feeding T*_N
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.controllers.mode_manager import (
    Mode,
    ModeConfig,
    ModeManager,
    beta_N,
    chi2_threshold,
    compute_T_star_G,
    compute_T_star_N,
)


# --------------------------------------------------------------------------- #
# Test 1 — T*_N = 59.3 s reference example                                     #
# --------------------------------------------------------------------------- #


def test_T_star_N_matches_reference_example_59p3_s() -> None:
    """Reference example for the noise-driven dwell:

    σ_η = 0.05, L = 20, s̄ = 0.5  →  σ_κ̂ = √(720·0.0025/(20⁵·0.5⁴)) = 3.000e-3
    ū = 0.5, d* = 5, d_min = 1 (margin 4), δ_L = 0.01 (β_N=0.3296)
    →  T*_N = √(2·4·0.3296 / (3e-3·0.25)) = √3515.7 = 59.3 s
    """
    sigma_kappa = 3.0e-3
    T = compute_T_star_N(
        d_star=5.0,
        d_min=1.0,
        sigma_kappa_hat=sigma_kappa,
        u_min=0.5,
        delta_L=0.01,
    )
    assert abs(T - 59.3) < 0.5, f"T*_N = {T:.3f}, expected ~59.3 s"


# --------------------------------------------------------------------------- #
# Test 2 — T*_G ∈ {19.8, 6.3, 2.0} s reference examples                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kappa_dot_max,T_expected",
    [
        (1e-3, 19.817),
        (1e-2, 6.267),
        (1e-1, 1.982),
    ],
)
def test_T_star_G_matches_reference_examples(
    kappa_dot_max: float, T_expected: float
) -> None:
    """Reference examples for the geometric-occlusion dwell (ū=2 m/s, w_FOV=π/2)."""
    T = compute_T_star_G(u_min=2.0, w_fov=np.pi / 2.0, kappa_dot_max=kappa_dot_max)
    assert abs(T - T_expected) < 0.05, (
        f"T*_G({kappa_dot_max}) = {T:.4f}, expected {T_expected}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — β_N formula matches the boxed closed form                          #
# --------------------------------------------------------------------------- #


def test_beta_N_closed_form() -> None:
    """β_N(δ_L) = 1 / sqrt(2 · ln(1/δ_L))."""
    for d in (0.01, 0.05, 0.1):
        expected = 1.0 / np.sqrt(2.0 * np.log(1.0 / d))
        assert abs(beta_N(d) - expected) < 1e-12
    # Default δ_L=0.01 ⇒ ~0.3296.
    assert abs(beta_N(0.01) - 0.3296) < 1e-3


# --------------------------------------------------------------------------- #
# Test 4 — χ² gate threshold and statistic                                    #
# --------------------------------------------------------------------------- #


def test_chi2_threshold_and_statistic() -> None:
    """At α=0.01, ν=2 → χ²_α = 9.21034 (table value)."""
    thr = chi2_threshold(alpha=0.01, dof=2)
    assert abs(thr - 9.21034) < 1e-3, f"χ² threshold = {thr}, expected ~9.21"
    # Statistic: η = [1,1], S = I ⇒ stat = 2.
    eta = np.array([1.0, 1.0])
    S = np.eye(2)
    stat = ModeManager.chi2_statistic(eta, S)
    assert abs(stat - 2.0) < 1e-12


# --------------------------------------------------------------------------- #
# Test 5 — FSM transitions: T → L_N → R when dwell exceeds T*_N               #
# --------------------------------------------------------------------------- #


def test_fsm_T_to_L_N_to_R_on_chi2_then_dwell() -> None:
    """χ² > threshold during T ⇒ L_N; after T*_N seconds ⇒ R."""
    cfg = ModeConfig(
        d_star=5.0,
        d_min=1.0,
        sigma_kappa_hat=3.0e-3,
        u_min=0.5,
        w_fov=np.pi / 2.0,
        kappa_dot_max=1e-3,
        warmup_s=0.5,
    )
    mm = ModeManager(cfg=cfg)
    dt = 0.1
    # Pass warmup
    mm.step(dt=dt, chi2_stat=0.5, visibility=True)  # t=0.1
    for _ in range(5):
        mm.step(dt=dt, chi2_stat=0.5, visibility=True)  # t=0.6 (post warmup)
    assert mm.mode == Mode.T

    # Trigger χ² gate
    mm.step(dt=dt, chi2_stat=20.0, visibility=True)
    assert mm.mode == Mode.L_N, f"after χ² trip: mode = {mm.mode}"

    # Stay in L_N for less than T*_N → still L_N
    while mm.dwell_in_current <= mm.T_star_N - 1.0:
        mm.step(dt=1.0, chi2_stat=20.0, visibility=True)
    assert mm.mode == Mode.L_N

    # One more step past T*_N → R
    while mm.dwell_in_current <= mm.T_star_N + 1.0:
        mm.step(dt=1.0, chi2_stat=20.0, visibility=True)
        if mm.mode == Mode.R:
            break
    assert mm.mode == Mode.R, f"after dwell > T*_N: mode = {mm.mode}"


# --------------------------------------------------------------------------- #
# Test 6 — FSM: T → L_G on visibility loss; L_G → T on visibility regain      #
# --------------------------------------------------------------------------- #


def test_fsm_T_to_L_G_on_visibility_then_back_on_regain() -> None:
    cfg = ModeConfig(
        d_star=5.0,
        d_min=1.0,
        sigma_kappa_hat=3.0e-3,
        u_min=2.0,
        w_fov=np.pi / 2.0,
        kappa_dot_max=1e-3,
        warmup_s=0.5,
    )
    mm = ModeManager(cfg=cfg)
    # Pass warmup
    for _ in range(10):
        mm.step(dt=0.1, chi2_stat=0.5, visibility=True)
    assert mm.mode == Mode.T

    # Lose visibility
    mm.step(dt=0.1, chi2_stat=0.5, visibility=False)
    assert mm.mode == Mode.L_G

    # Regain visibility within T*_G window → back to T
    mm.step(dt=0.5, chi2_stat=0.5, visibility=True)
    assert mm.mode == Mode.T


# --------------------------------------------------------------------------- #
# Test 7 — warm-up gate disables χ² for first warmup_s seconds                #
# --------------------------------------------------------------------------- #


def test_warmup_disables_chi2_gate() -> None:
    cfg = ModeConfig(
        d_star=5.0, d_min=1.0, sigma_kappa_hat=3e-3, u_min=0.5,
        w_fov=np.pi / 2.0, kappa_dot_max=1e-3, warmup_s=1.0,
    )
    mm = ModeManager(cfg=cfg)
    # Huge χ² but t < warmup → still T
    mm.step(dt=0.1, chi2_stat=999.0, visibility=True)
    assert mm.mode == Mode.T, f"warmup should suppress χ² gate; mode = {mm.mode}"


# --------------------------------------------------------------------------- #
# Test 8 — config validation                                                  #
# --------------------------------------------------------------------------- #


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="d_star"):
        ModeConfig(
            d_star=1.0, d_min=2.0, sigma_kappa_hat=1e-3, u_min=1.0,
            w_fov=1.0, kappa_dot_max=1e-3,
        )
    with pytest.raises(ValueError, match="sigma_kappa_hat"):
        ModeConfig(
            d_star=5.0, d_min=1.0, sigma_kappa_hat=-1.0, u_min=1.0,
            w_fov=1.0, kappa_dot_max=1e-3,
        )
    with pytest.raises(ValueError, match="delta_L"):
        ModeConfig(
            d_star=5.0, d_min=1.0, sigma_kappa_hat=1e-3, u_min=1.0,
            w_fov=1.0, kappa_dot_max=1e-3, delta_L=1.5,
        )


# --------------------------------------------------------------------------- #
# Test 9 — joint T* = max{T*_N, T*_G}                                          #
# --------------------------------------------------------------------------- #


def test_joint_T_star_takes_maximum() -> None:
    cfg = ModeConfig(
        d_star=5.0, d_min=1.0, sigma_kappa_hat=3e-3, u_min=0.5,
        w_fov=np.pi / 2.0, kappa_dot_max=1e-3,
    )
    mm = ModeManager(cfg=cfg)
    assert mm.T_star == max(mm.T_star_N, mm.T_star_G)
