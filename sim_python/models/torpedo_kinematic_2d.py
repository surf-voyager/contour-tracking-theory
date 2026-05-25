"""L0 kinematic 2D torpedo model with saturation + rudder-gate predicate.

Implements the error / actuator dynamics for the Layer-1 abstraction:

    ẋ      = u cos ψ
    ẏ      = u sin ψ
    ψ̇      = r
    u̇      = (u_cmd - u) / τ_u                       , u clipped to [0, u_max]
    ṙ      = { rate-limited ramp toward r_cmd ,  u ≥ u_gate
             { −ζ r                            ,  u < u_gate   (rudder-gate
                                                                 disable case)
    |r| ≤ r_max     (hard clip applied after each step)
    |ṙ|  ≤ dot_r_max (rate limit on the ramp branch)

State vector convention (length 5):

    s = [x, y, ψ, u, r]

This is the pure-kinematic (no hydrodynamic drag, no Fossen mass matrix)
abstraction used for fast verification of the Theorem 1 conditions
(C1)/(C2) and the six must-see phenomena. Higher-fidelity variants live
in sibling modules and replace the (u̇, ṙ) equations with first-order
surge/yaw dynamics or the full Fossen model respectively.

Convention notes:
    - τ_u (surge time constant) and ζ (gate-disable yaw damping) are
      abstract layer constants — NOT MC axes at L0. Fixed defaults are
      τ_u = 0.5 s and ζ = 2.0 s⁻¹ (nominal example values); they are
      exposed as HullConfig fields for testability / sweeps but no
      phase-diagram MC slice varies them at L0.
    - The integrator is fixed-step RK4. The rate-limited yaw branch is
      piecewise-linear; RK4 of a piecewise-constant slope reduces to
      explicit Euler near the rate limit, which is intentional (matches
      a real bang-bang rate limiter).
    - Heading ψ is NOT wrapped to (−π, π] inside the integrator (long
      straight runs would otherwise have spurious jumps in finite-
      difference diagnostics); analysis code may wrap as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# State / control indices                                                     #
# --------------------------------------------------------------------------- #

IDX_X: int = 0
IDX_Y: int = 1
IDX_PSI: int = 2
IDX_U: int = 3
IDX_R: int = 4
STATE_DIM: int = 5

CTRL_U_CMD: int = 0
CTRL_R_CMD: int = 1
CTRL_DIM: int = 2


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HullConfig:
    """L0 kinematic torpedo hull configuration.

    Attributes
    ----------
    u_max : float
        Maximum forward speed [m/s]; surge state is clipped to [0, u_max].
    r_max : float
        Maximum |yaw rate| [rad/s]; yaw-rate state is clipped to ±r_max.
    dot_r_max : float
        Maximum |ṙ| [rad/s²] on the ramp branch (slew rate of the rudder
        commanded yaw-rate set-point).
    u_gate : float
        Rudder-effectiveness gate [m/s]. When u < u_gate the rudder
        produces no yaw torque; commanded yaw rate is forced to zero
        and a first-order damping -ζ r kicks in.
    tau_u : float, default 0.5
        Surge first-order time constant [s].  Abstract layer constant
        per module docstring; not an MC axis at L0.
    zeta_gate : float, default 2.0
        Yaw-rate damping coefficient [s⁻¹] active when u < u_gate.
        Abstract layer constant.
    """

    u_max: float
    r_max: float
    dot_r_max: float
    u_gate: float
    tau_u: float = 0.5
    zeta_gate: float = 2.0

    def __post_init__(self) -> None:
        # Light validation — bare minimum to catch sign errors.
        for name in ("u_max", "r_max", "dot_r_max", "u_gate", "tau_u", "zeta_gate"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"HullConfig.{name} must be finite and ≥ 0; got {v!r}")
        if self.u_gate >= self.u_max:
            raise ValueError(
                f"HullConfig.u_gate ({self.u_gate}) must be < u_max ({self.u_max})"
            )


# --------------------------------------------------------------------------- #
# Continuous-time RHS                                                         #
# --------------------------------------------------------------------------- #


def _rhs(state: np.ndarray, control: np.ndarray, cfg: HullConfig) -> np.ndarray:
    """Continuous-time derivative ds/dt at (state, control).

    See module docstring for the closed-form ODE.  Saturation on (u, r)
    is treated as a hard state constraint: the *evaluated* state has
    its (u, r) components first clipped to their feasible boxes, and
    then if the resulting derivative would push the state further
    outside the box we zero it out.  Clipping the *evaluation point*
    (in addition to post-step clipping in `step_l0`) is necessary
    because RK4's intermediate stages can probe states slightly past
    the boundary (by O(dt²)); without the evaluation-point clip a
    bang-bang rate limiter would produce chatter at ~r_max - O(dt)
    instead of settling exactly at r_max.
    """
    # Clip evaluation point to the feasible (u, r) box so intermediate
    # RK4 stages don't see infeasible states.
    u = float(np.clip(state[IDX_U], 0.0, cfg.u_max))
    r = float(np.clip(state[IDX_R], -cfg.r_max, cfg.r_max))
    psi = state[IDX_PSI]
    u_cmd = control[CTRL_U_CMD]
    r_cmd = control[CTRL_R_CMD]

    ds = np.zeros(STATE_DIM, dtype=np.float64)
    ds[IDX_X] = u * np.cos(psi)
    ds[IDX_Y] = u * np.sin(psi)
    ds[IDX_PSI] = r

    # ----- surge dynamics: u̇ = (u_cmd - u)/τ_u with surge clamp ----------- #
    u_cmd_eff = float(np.clip(u_cmd, 0.0, cfg.u_max))
    du = (u_cmd_eff - u) / cfg.tau_u
    # Stop if at the boundary and being pushed further outside.
    if (u <= 0.0 and du < 0.0) or (u >= cfg.u_max and du > 0.0):
        du = 0.0
    ds[IDX_U] = du

    # ----- yaw-rate dynamics with rudder-gate switch ----------------------- #
    if u < cfg.u_gate:
        # Rudder ineffective: damp r toward zero.
        dr = -cfg.zeta_gate * r
    else:
        # Rate-limited ramp toward r_cmd, with yaw-rate hard clip.
        r_cmd_eff = float(np.clip(r_cmd, -cfg.r_max, cfg.r_max))
        err = r_cmd_eff - r
        # Bang-bang slew toward r_cmd_eff: ṙ = sign(err) * min(|err|/dt_*, dot_r_max).
        # We do NOT know dt here; use the maximum slope dot_r_max and let
        # the discrete step take care of "we'd overshoot in one dt" by
        # post-clipping in step_l0.  This matches a real rudder rate
        # limiter (bang-bang at ±dot_r_max until in the dead-band).
        dr = float(np.sign(err)) * cfg.dot_r_max
        if err == 0.0:
            dr = 0.0
        # If at the yaw-rate clip and being pushed further out, freeze.
        if (r >= cfg.r_max and dr > 0.0) or (r <= -cfg.r_max and dr < 0.0):
            dr = 0.0
    ds[IDX_R] = dr

    return ds


def _apply_clips(state: np.ndarray, cfg: HullConfig) -> np.ndarray:
    """Enforce hard saturation on (u, r) post-integration."""
    state = state.copy()
    state[IDX_U] = float(np.clip(state[IDX_U], 0.0, cfg.u_max))
    state[IDX_R] = float(np.clip(state[IDX_R], -cfg.r_max, cfg.r_max))
    return state


# --------------------------------------------------------------------------- #
# Stepper                                                                     #
# --------------------------------------------------------------------------- #


def step_l0(
    state: np.ndarray,
    control: np.ndarray,
    cfg: HullConfig,
    dt: float,
) -> np.ndarray:
    """One RK4 step of the L0 kinematic torpedo model.

    Parameters
    ----------
    state : ndarray, shape (5,)
        [x, y, ψ, u, r] at time t.
    control : ndarray, shape (2,)
        [u_cmd, r_cmd] held constant over (t, t+dt].
    cfg : HullConfig
    dt : float
        Step size [s], must be > 0.

    Returns
    -------
    next_state : ndarray, shape (5,)
        State at time t+dt with hard clips applied.

    Notes
    -----
    The yaw-rate branch is piecewise-linear in r when u ≥ u_gate; RK4 of
    a piecewise-constant slope reduces to explicit Euler near the rate
    limit, which is intentional.  Post-step clipping handles the case
    where one RK4 step would otherwise push r past r_max because the
    bang-bang derivative is held constant over [t, t+dt].
    """
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"dt must be positive and finite; got {dt!r}")
    s = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    c = np.asarray(control, dtype=np.float64).reshape(CTRL_DIM)

    k1 = _rhs(s, c, cfg)
    k2 = _rhs(s + 0.5 * dt * k1, c, cfg)
    k3 = _rhs(s + 0.5 * dt * k2, c, cfg)
    k4 = _rhs(s + dt * k3, c, cfg)
    s_next = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _apply_clips(s_next, cfg)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def simulate(
    initial_state: np.ndarray,
    control_fn: Callable[[float, np.ndarray], np.ndarray],
    cfg: HullConfig,
    t_end: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the L0 model from t=0 to t=t_end.

    Parameters
    ----------
    initial_state : ndarray, shape (5,)
        Initial [x, y, ψ, u, r].
    control_fn : callable (t, state) -> ndarray shape (2,)
        Returns [u_cmd, r_cmd] given current time and state.
        Held constant over each (t, t+dt] step (zero-order hold).
    cfg : HullConfig
    t_end : float
        Final time [s]; trajectory has N = floor(t_end/dt) + 1 samples.
    dt : float
        Step size [s].

    Returns
    -------
    t : ndarray, shape (N,)
        Time stamps.
    traj : ndarray, shape (N, 5)
        State at each time stamp; row 0 = initial_state.
    controls : ndarray, shape (N, 2)
        Control applied over the step starting at t[k]; controls[-1]
        is what would have been applied at t_end (held for plotting).
    """
    if t_end <= 0:
        raise ValueError(f"t_end must be > 0; got {t_end!r}")
    n_steps = int(np.floor(t_end / dt))
    n_samples = n_steps + 1
    s0 = np.asarray(initial_state, dtype=np.float64).reshape(STATE_DIM)
    # Hard-clip the initial state too (defensive: a caller might pass
    # u > u_max accidentally).
    s = _apply_clips(s0, cfg)

    t = np.zeros(n_samples, dtype=np.float64)
    traj = np.zeros((n_samples, STATE_DIM), dtype=np.float64)
    controls = np.zeros((n_samples, CTRL_DIM), dtype=np.float64)

    traj[0] = s
    c0 = np.asarray(control_fn(0.0, s), dtype=np.float64).reshape(CTRL_DIM)
    controls[0] = c0
    for k in range(n_steps):
        tk = k * dt
        c = np.asarray(control_fn(tk, s), dtype=np.float64).reshape(CTRL_DIM)
        controls[k] = c
        s = step_l0(s, c, cfg, dt)
        t[k + 1] = (k + 1) * dt
        traj[k + 1] = s
    # Final-sample control: just re-evaluate at t_end (purely cosmetic
    # for plotting; not used by the dynamics).
    controls[-1] = np.asarray(control_fn(t[-1], s), dtype=np.float64).reshape(CTRL_DIM)
    return t, traj, controls


# --------------------------------------------------------------------------- #
# Convenience: piecewise-constant control schedule                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ControlSegment:
    """One segment of a piecewise-constant control schedule."""

    t_0: float
    t_1: float
    u_cmd: float
    r_cmd: float


def make_piecewise_control(
    segments: list[ControlSegment],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build a control_fn from a list of (t_0, t_1, u_cmd, r_cmd) segments.

    Segments are tried in order; the first segment with t_0 ≤ t < t_1
    wins.  If t falls outside every segment, returns the *last* segment's
    control (so end-of-trajectory effects don't crash).
    """
    if not segments:
        raise ValueError("segments list must be non-empty")
    segs = list(segments)

    def _ctrl(t: float, _state: np.ndarray) -> np.ndarray:
        for seg in segs:
            if seg.t_0 <= t < seg.t_1:
                return np.array([seg.u_cmd, seg.r_cmd], dtype=np.float64)
        last = segs[-1]
        return np.array([last.u_cmd, last.r_cmd], dtype=np.float64)

    return _ctrl


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    """10 s straight run + 10 s constant turn.  Prints summary stats."""
    cfg = HullConfig(u_max=2.0, r_max=0.5, dot_r_max=1.0, u_gate=0.3)
    segments = [
        ControlSegment(t_0=0.0, t_1=10.0, u_cmd=1.0, r_cmd=0.0),
        ControlSegment(t_0=10.0, t_1=20.0, u_cmd=1.0, r_cmd=0.3),
    ]
    ctrl = make_piecewise_control(segments)
    s0 = np.zeros(STATE_DIM, dtype=np.float64)
    t, traj, controls = simulate(s0, ctrl, cfg, t_end=20.0, dt=0.01)
    print(f"[smoke] t shape={t.shape}, traj shape={traj.shape}")
    print(f"[smoke] final state (x,y,psi,u,r) = {traj[-1]}")
    print(f"[smoke] mid state @ t=10s          = {traj[1000]}")
    assert not np.any(np.isnan(traj)), "smoke produced NaN"
    assert traj.shape == (2001, 5), f"unexpected shape {traj.shape}"
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
