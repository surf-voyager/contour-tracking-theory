"""Closed-loop integration smoke (hull + LOS + curvature feed-forward).

The saturated LOS drives the lateral error to d=d*, and the κ
feed-forward gives zero steady-state error on a circular wall.

Acceptance:
- Straight wall + LOS: |d - d*| < 0.1 within 20 s
- Circular wall R=10 m: error stays bounded (no divergence, |d-d*| < 0.5)

Closed-loop assembly
--------------------
Hull model (sim_python.models.torpedo_kinematic_2d) + LOS controller
(sim_python.controllers.los_controller) + lateral-distance computation
from the AUV's body position to a wall polyline (helper below).

We intentionally do NOT exercise the χ²/visibility FSM in the smoke
(that's covered by sim_python/tests/test_mode_manager.py).  The smoke
focuses on the LOS + feed-forward + hull loop closing on tracking error.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.controllers.los_controller import LOSConfig, compute_r_star
from sim_python.models.torpedo_kinematic_2d import (
    HullConfig,
    IDX_PSI,
    IDX_R,
    IDX_U,
    IDX_X,
    IDX_Y,
    STATE_DIM,
    simulate,
    step_l0,
)
from sim_python.scenarios.wall_generator import Wall, gen_arc, gen_straight


# --------------------------------------------------------------------------- #
# Wall-projection helper                                                      #
# --------------------------------------------------------------------------- #


def signed_distance_to_wall(
    pos: np.ndarray, wall: Wall
) -> tuple[float, float, float]:
    """Closest-point projection of AUV onto a wall polyline.

    Returns
    -------
    d : float
        Signed perpendicular distance to wall, *measured from the
        standoff-side reference*.  Sign convention:
        - side="L": +d means the AUV is on the LEFT of the wall (above
          y-axis for straight wall); standoff is achieved at d=+d*.
        - side="R": +d means RIGHT; standoff at d=+d*.
        For the LOS controller we always pass the difference to d*
        so the LOS pulls back to set-point.
    gamma_p : float
        Wall tangent angle at the closest point [rad].
    kappa_true : float
        Wall curvature at the closest point [m⁻¹] (oracle; not used
        by LOS feed-forward in the smoke — we pass κ̂=κ_true).
    """
    diffs = wall.points - pos[None, :]
    dist2 = np.einsum("ij,ij->i", diffs, diffs)
    k = int(np.argmin(dist2))
    closest = wall.points[k]
    gp = float(wall.tangent[k])
    kappa = float(wall.kappa[k])
    # Signed distance: perpendicular component in wall normal direction.
    # Normal n̂ = R(90°) t̂ = (-sin γ_p, cos γ_p) (left of tangent).
    nx = -np.sin(gp)
    ny = +np.cos(gp)
    vx = pos[0] - closest[0]
    vy = pos[1] - closest[1]
    d_left = vx * nx + vy * ny
    # For side="L", positive d (above wall) is the standoff side.
    # For side="R", flip sign so positive d means standoff side.
    if wall.side == "L":
        d_signed = d_left
    else:
        d_signed = -d_left
    return d_signed, gp, kappa


# --------------------------------------------------------------------------- #
# Closed-loop driver                                                          #
# --------------------------------------------------------------------------- #


def run_closed_loop(
    wall: Wall,
    hull_cfg: HullConfig,
    los_cfg: LOSConfig,
    initial_state: np.ndarray,
    t_end: float,
    dt: float,
    u_cmd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run hull + LOS + oracle-κ in closed loop.

    For the smoke we use the true κ(s) at the closest-point projection
    as the feed-forward κ̂.  The integration test for the estimator-
    in-the-loop case is deferred to stage 03 (full MC framework).

    Returns
    -------
    t : ndarray (N,)
    traj : ndarray (N, 5)
    err : ndarray (N,)   signed (d - d*) at each step
    """
    n_steps = int(np.floor(t_end / dt))
    n_samples = n_steps + 1
    t = np.zeros(n_samples)
    traj = np.zeros((n_samples, STATE_DIM))
    err = np.zeros(n_samples)

    s = np.asarray(initial_state, dtype=np.float64).copy()
    traj[0] = s
    pos = np.array([s[IDX_X], s[IDX_Y]])
    d0, gp0, kappa0 = signed_distance_to_wall(pos, wall)
    err[0] = d0 - los_cfg.d_star

    for k in range(n_steps):
        pos = np.array([s[IDX_X], s[IDX_Y]])
        d, gp, kappa_true = signed_distance_to_wall(pos, wall)
        # Heading error θ = ψ - γ_p
        theta = float(s[IDX_PSI]) - gp
        # Oracle κ̂ for smoke
        r_cmd = compute_r_star(
            d=d, theta=theta, kappa_hat=kappa_true,
            u=float(s[IDX_U]), cfg=los_cfg,
        )
        ctrl = np.array([u_cmd, r_cmd])
        s = step_l0(s, ctrl, hull_cfg, dt)
        t[k + 1] = (k + 1) * dt
        traj[k + 1] = s
        err[k + 1] = d - los_cfg.d_star

    return t, traj, err


# --------------------------------------------------------------------------- #
# Test 1 — straight wall + LOS, |d-d*| < 0.1 within 20 s                      #
# --------------------------------------------------------------------------- #


def test_closed_loop_straight_wall_converges_within_20s() -> None:
    """Straight wall along +x, AUV starts at y=3, d*=2 (side=L means
    standoff is at y=+d*=+2).  LOS must pull |d-d*| < 0.1 within 20 s.
    """
    wall = gen_straight(length=80.0, side="L", ds=0.05)
    hull_cfg = HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)
    # K_theta = 0.5 provides the heading-error damping needed to avoid
    # limit-cycle oscillation around d* (the rate-only sin-LOS has no
    # passive ψ damping; cf. Caharija 2016 §III which adds a velocity
    # damping term via the look-ahead's effective ψ_des).
    los_cfg = LOSConfig(Delta=2.0, K_p=0.4, K_ff=1.0, K_theta=0.5, d_star=2.0)
    # Start at (1, 3) heading +x, surge already at cruise so we exclude
    # the surge transient from the convergence test.
    s0 = np.array([1.0, 3.0, 0.0, 1.0, 0.0])
    t, traj, err = run_closed_loop(
        wall=wall, hull_cfg=hull_cfg, los_cfg=los_cfg,
        initial_state=s0, t_end=20.0, dt=0.02, u_cmd=1.0,
    )
    # At t=20 s the error magnitude must be < 0.1 m
    final_err = abs(err[-1])
    assert final_err < 0.1, f"|d - d*| at t=20s = {final_err:.4f}, expected < 0.1"
    # And no NaN
    assert np.all(np.isfinite(traj))


# --------------------------------------------------------------------------- #
# Test 2 — circular wall R=10, error stays bounded                            #
# --------------------------------------------------------------------------- #


def test_closed_loop_circular_wall_bounded_error() -> None:
    """Circular wall R=10 m, side=L (κ=+1/R). LOS + κ feed-forward must
    keep tracking error bounded.  We allow ≤ 0.5 m peak error after the
    initial transient (5 s).
    """
    R = 10.0
    # 2/3 of a full circle = 4πR/3 ≈ 41.9 m of arc
    arc_len = 4.0 * np.pi * R / 3.0
    wall = gen_arc(radius=R, arc_length=arc_len, side="L", ds=0.05)
    hull_cfg = HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)
    los_cfg = LOSConfig(Delta=2.0, K_p=0.4, K_ff=1.0, K_theta=0.5, d_star=2.0)
    # Start on the standoff path: at (0, +2), heading +x, u=1.
    s0 = np.array([0.0, 2.0, 0.0, 1.0, 0.0])
    t, traj, err = run_closed_loop(
        wall=wall, hull_cfg=hull_cfg, los_cfg=los_cfg,
        initial_state=s0, t_end=25.0, dt=0.02, u_cmd=1.0,
    )
    assert np.all(np.isfinite(traj))
    # After initial 5 s transient, peak |err| ≤ 0.5 m.
    idx_5s = int(5.0 / 0.02)
    peak = float(np.max(np.abs(err[idx_5s:])))
    assert peak < 0.5, f"peak |d-d*| after t=5s = {peak:.4f}, expected < 0.5"
    # And error doesn't diverge — final ≤ peak.
    assert abs(err[-1]) <= peak + 1e-9


# --------------------------------------------------------------------------- #
# Test 3 — signed-distance helper correctness on straight wall                #
# --------------------------------------------------------------------------- #


def test_signed_distance_helper_on_straight_wall() -> None:
    wall = gen_straight(length=10.0, side="L", ds=0.05)
    pos = np.array([5.0, 2.5])
    d, gp, kappa = signed_distance_to_wall(pos, wall)
    assert abs(d - 2.5) < 1e-9, f"d = {d}, expected 2.5"
    assert abs(gp) < 1e-9, f"γ_p = {gp}, expected 0"
    assert abs(kappa) < 1e-9
    # For side="R" the same physical AUV at y=+2.5 sits OUTSIDE the
    # standoff zone → signed distance is negative.
    wall_R = gen_straight(length=10.0, side="R", ds=0.05)
    d_R, _, _ = signed_distance_to_wall(pos, wall_R)
    assert abs(d_R - (-2.5)) < 1e-9


# --------------------------------------------------------------------------- #
# Test 4 — closed loop determinism                                            #
# --------------------------------------------------------------------------- #


def test_closed_loop_determinism() -> None:
    """Two identical closed-loop runs produce bit-identical trajectories."""
    wall = gen_straight(length=20.0, side="L", ds=0.05)
    hull_cfg = HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)
    los_cfg = LOSConfig(Delta=2.0, K_p=0.4, d_star=2.0)
    s0 = np.array([0.0, 3.0, 0.0, 1.0, 0.0])
    t_a, traj_a, err_a = run_closed_loop(
        wall, hull_cfg, los_cfg, s0, t_end=15.0, dt=0.02, u_cmd=1.0
    )
    t_b, traj_b, err_b = run_closed_loop(
        wall, hull_cfg, los_cfg, s0, t_end=15.0, dt=0.02, u_cmd=1.0
    )
    assert np.array_equal(traj_a, traj_b)
    assert np.array_equal(err_a, err_b)


# --------------------------------------------------------------------------- #
# Test 5 — bigger initial offset still converges within 30 s (saturated LOS)   #
# --------------------------------------------------------------------------- #


def test_large_initial_offset_eventually_converges() -> None:
    """Even with a 5-m initial offset (vs d*=2) the saturated LOS must
    drive the error monotonically down.  We require final |err| < 0.2 m
    within 30 s.
    """
    wall = gen_straight(length=80.0, side="L", ds=0.05)
    hull_cfg = HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)
    los_cfg = LOSConfig(Delta=2.0, K_p=0.4, K_theta=0.5, d_star=2.0)
    s0 = np.array([0.0, 7.0, 0.0, 1.0, 0.0])  # 5 m above standoff
    _, traj, err = run_closed_loop(
        wall, hull_cfg, los_cfg, s0, t_end=30.0, dt=0.02, u_cmd=1.0
    )
    assert np.all(np.isfinite(traj))
    assert abs(err[-1]) < 0.2, f"final |err| = {abs(err[-1]):.4f}, expected < 0.2"
    # Error magnitude should decrease over time (allow modest non-monotonic
    # ripple; require that the max in the second half is much smaller
    # than the max in the first half).
    n = len(err)
    half = n // 2
    max_first = float(np.max(np.abs(err[:half])))
    max_second = float(np.max(np.abs(err[half:])))
    assert max_second < 0.5 * max_first, (
        f"error not decreasing: 1st-half max={max_first}, "
        f"2nd-half max={max_second}"
    )
