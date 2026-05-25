"""Tests for sim_python.models.torpedo_kinematic_2d (L0 hull).

Covers the L0 error / actuator dynamics and the rudder gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.models.torpedo_kinematic_2d import (
    CTRL_DIM,
    IDX_PSI,
    IDX_R,
    IDX_U,
    IDX_X,
    IDX_Y,
    STATE_DIM,
    ControlSegment,
    HullConfig,
    make_piecewise_control,
    simulate,
    step_l0,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def cfg() -> HullConfig:
    return HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)


# --------------------------------------------------------------------------- #
# Test 1 — straight run                                                       #
# --------------------------------------------------------------------------- #


def test_straight_run(cfg: HullConfig) -> None:
    """u_cmd=1, r_cmd=0 for 10 s → x≈10, y≈0, ψ≈0 (5% tolerance).

    Note: with τ_u = 0.5 s and u starting at 0, the surge response
    leaves a transient of duration ~3 τ_u = 1.5 s during which the
    integrated x falls slightly short of u_cmd * t.  Theoretical
    final x for a perfect first-order response = 10 - τ_u*(1 - e^-20)
    ≈ 9.5 m, which is the analytical reference.  5% tolerance keeps
    the test agnostic to the exact transient model.
    """
    ctrl = make_piecewise_control(
        [ControlSegment(t_0=0.0, t_1=10.0, u_cmd=1.0, r_cmd=0.0)]
    )
    s0 = np.zeros(STATE_DIM)
    t, traj, _ = simulate(s0, ctrl, cfg, t_end=10.0, dt=0.01)

    final = traj[-1]
    # Analytical x with τ_u=0.5: x(10) = 10 - 0.5*(1 - e^-20) ≈ 9.5
    x_expected = 10.0 - cfg.tau_u * (1.0 - np.exp(-10.0 / cfg.tau_u))
    assert abs(final[IDX_X] - x_expected) / x_expected < 0.05, (
        f"x={final[IDX_X]} not within 5% of analytical {x_expected}"
    )
    assert abs(final[IDX_Y]) < 0.05, f"y={final[IDX_Y]} should be ~0"
    assert abs(final[IDX_PSI]) < 0.01, f"psi={final[IDX_PSI]} should be ~0"
    # Surge should have converged to u_cmd.
    assert abs(final[IDX_U] - 1.0) < 1e-6, f"u={final[IDX_U]} not at u_cmd=1"
    assert abs(final[IDX_R]) < 1e-9, f"r={final[IDX_R]} should be exactly 0"


# --------------------------------------------------------------------------- #
# Test 2 — steady turn                                                        #
# --------------------------------------------------------------------------- #


def test_steady_turn(cfg: HullConfig) -> None:
    """From steady (u=1, r=0) hold u_cmd=1, r_cmd=0.3 and verify:
    - r reaches ~0.3 within ~5/dot_r_max ≈ 5 s
    - after settling, ψ grows linearly with slope ≈ 0.3 rad/s
    """
    # Start already at u=1 to avoid surge transient confounding the test.
    s0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    ctrl = make_piecewise_control(
        [ControlSegment(t_0=0.0, t_1=10.0, u_cmd=1.0, r_cmd=0.3)]
    )
    t, traj, _ = simulate(s0, ctrl, cfg, t_end=10.0, dt=0.01)

    # Settling: at t = 5 / dot_r_max = 5 s, r should be ~0.3 (it
    # actually reaches 0.3 within 0.3 s under bang-bang, with small
    # chatter of order dt*dot_r_max = 0.01 rad/s).
    idx_5s = int(5.0 / 0.01)
    assert abs(traj[idx_5s, IDX_R] - 0.3) < 0.05, (
        f"r at t=5s = {traj[idx_5s, IDX_R]} not within 0.05 of 0.3"
    )

    # ψ linearity: between t=5s and t=10s, slope of ψ ≈ 0.3 rad/s.
    idx_10s = -1
    dpsi = traj[idx_10s, IDX_PSI] - traj[idx_5s, IDX_PSI]
    dt_span = t[idx_10s] - t[idx_5s]
    slope = dpsi / dt_span
    assert abs(slope - 0.3) < 0.01, (
        f"ψ slope = {slope} rad/s not within 0.01 of 0.3"
    )


# --------------------------------------------------------------------------- #
# Test 3 — rudder gate blocks below u_gate                                    #
# --------------------------------------------------------------------------- #


def test_rudder_gate_blocks_below_u_gate(cfg: HullConfig) -> None:
    """u_cmd=0.2 (< u_gate=0.3), r_cmd=0.5 ⇒ r damps to ~0 within 1/ζ.

    When u < u_gate, ṙ = -ζ r regardless of r_cmd.
    With ζ=2 s⁻¹, the damping time constant is 0.5 s; after 5*0.5=2.5 s
    r should be < 1% of its initial value.

    To exercise the damping branch we set u below u_gate at t=0 and
    initialise r=0.4 (non-zero); the test asserts |r(t)| → 0 even though
    a non-zero r_cmd is being requested.
    """
    # u starts at 0.2 < u_gate and stays there (u_cmd=0.2); r starts
    # at 0.4 so we can watch it decay.
    s0 = np.array([0.0, 0.0, 0.0, 0.2, 0.4])
    ctrl = make_piecewise_control(
        [ControlSegment(t_0=0.0, t_1=5.0, u_cmd=0.2, r_cmd=0.5)]
    )
    t, traj, _ = simulate(s0, ctrl, cfg, t_end=5.0, dt=0.01)

    # Surge stays below gate throughout (it will sit at 0.2 = u_cmd_eff).
    assert np.all(traj[:, IDX_U] < cfg.u_gate), (
        f"u exceeded u_gate at some sample; max(u)={traj[:, IDX_U].max()}"
    )
    # r at t=2.5s (5/ζ) should be ~0.4 * e^-5 ≈ 0.0027.
    idx_25s = int(2.5 / 0.01)
    assert abs(traj[idx_25s, IDX_R]) < 0.01, (
        f"r at t=2.5s = {traj[idx_25s, IDX_R]}; expected ~0"
    )
    # By end of run r should be vanishingly small.
    assert abs(traj[-1, IDX_R]) < 1e-3, (
        f"r at end = {traj[-1, IDX_R]}; expected ~0 under gate-disable damping"
    )


# --------------------------------------------------------------------------- #
# Test 4 — saturation r_max                                                   #
# --------------------------------------------------------------------------- #


def test_saturation_r_max(cfg: HullConfig) -> None:
    """r_cmd=10 ≫ r_max ⇒ r settles at exactly r_max (the clip enforces it)."""
    s0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    ctrl = make_piecewise_control(
        [ControlSegment(t_0=0.0, t_1=5.0, u_cmd=1.0, r_cmd=10.0)]
    )
    t, traj, _ = simulate(s0, ctrl, cfg, t_end=5.0, dt=0.01)

    # Final r must be exactly r_max (not just close — the clip is exact).
    assert traj[-1, IDX_R] == cfg.r_max, (
        f"r={traj[-1, IDX_R]}, expected exactly {cfg.r_max}"
    )
    # And no sample should ever exceed r_max in magnitude.
    assert np.max(np.abs(traj[:, IDX_R])) <= cfg.r_max + 1e-12, (
        "saturation clip violated"
    )


# --------------------------------------------------------------------------- #
# Test 5 — saturation dot_r_max (rate limit)                                  #
# --------------------------------------------------------------------------- #


def test_saturation_dot_r_max(cfg: HullConfig) -> None:
    """Ramp r_cmd from 0 to large value; per-step |dr/dt| ≤ dot_r_max·(1+1%)."""
    s0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    # Request a yaw rate large in magnitude immediately ⇒ slew at dot_r_max.
    ctrl = make_piecewise_control(
        [ControlSegment(t_0=0.0, t_1=2.0, u_cmd=1.0, r_cmd=5.0)]
    )
    dt = 0.01
    t, traj, _ = simulate(s0, ctrl, cfg, t_end=2.0, dt=dt)

    # Per-step dr/dt; ignore the trailing samples after r saturated at r_max
    # (then dr=0 by definition, which trivially satisfies the bound).
    drdt = np.diff(traj[:, IDX_R]) / dt
    # 1% tolerance per stage spec.
    assert np.max(np.abs(drdt)) <= cfg.dot_r_max * 1.01, (
        f"|dr/dt|_max = {np.max(np.abs(drdt))} exceeds dot_r_max={cfg.dot_r_max} +1%"
    )


# --------------------------------------------------------------------------- #
# Test 6 — determinism                                                        #
# --------------------------------------------------------------------------- #


def test_determinism(cfg: HullConfig) -> None:
    """Two identical runs produce bit-identical trajectories.

    L0 dynamics are deterministic (no Generator state); this test
    guards against any future regression that introduces hidden
    stochasticity (e.g. accidentally using np.random.* somewhere).
    """
    s0 = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    segments = [
        ControlSegment(t_0=0.0, t_1=5.0, u_cmd=1.5, r_cmd=0.4),
        ControlSegment(t_0=5.0, t_1=10.0, u_cmd=0.1, r_cmd=-0.3),
    ]
    ctrl_a = make_piecewise_control(segments)
    ctrl_b = make_piecewise_control(segments)
    _, traj_a, ctl_a = simulate(s0, ctrl_a, cfg, t_end=10.0, dt=0.01)
    _, traj_b, ctl_b = simulate(s0, ctrl_b, cfg, t_end=10.0, dt=0.01)

    assert np.array_equal(traj_a, traj_b), "trajectories differ — determinism leak"
    assert np.array_equal(ctl_a, ctl_b), "controls differ — determinism leak"


# --------------------------------------------------------------------------- #
# Sanity: every test produced no NaN                                          #
# --------------------------------------------------------------------------- #


def test_no_nan_in_smoke(cfg: HullConfig) -> None:
    """Defensive: full 20-s smoke trajectory has no NaN/inf in any column."""
    ctrl = make_piecewise_control(
        [
            ControlSegment(t_0=0.0, t_1=10.0, u_cmd=1.0, r_cmd=0.0),
            ControlSegment(t_0=10.0, t_1=20.0, u_cmd=1.0, r_cmd=0.3),
        ]
    )
    s0 = np.zeros(STATE_DIM)
    _, traj, _ = simulate(s0, ctrl, cfg, t_end=20.0, dt=0.01)
    assert np.all(np.isfinite(traj)), "trajectory contains NaN/inf"
    assert traj.shape == (2001, STATE_DIM)
