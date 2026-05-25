"""MC dispatcher CLI: LHS over axes → per-run closed-loop sim → parquet.

Latin-Hypercube samples the configuration axes, runs one closed-loop
simulation per sample, and writes a trajectory parquet plus a summary
per run.

Closed-loop driver:
- The 4-state FSM ({T, L_N, L_G, R}, ModeManager + ReacquirePlanner) is
  plugged into the closed-loop driver so the `mode`, `lost_count` and
  `reacq_time` parquet columns are non-trivial.  L_N is triggered by a
  χ²-equivalent test on the lateral residual; L_G is triggered by a
  visibility predicate computed from the AUV pose and the wall
  closest-point relative bearing vs the (per-config) w_FOV.
- Sensor noise σ_η perturbs the measured d before the LOS controller
  consumes it (per-step Gaussian, deterministic via the LHS seed).
- Control delay τ_d is implemented as a discrete delay line on the
  commanded yaw rate r_cmd (samples skipped per round(τ_d/dt)).
- ``config_hash`` includes the hull / los blocks too, so batches that
  sweep hull/LOS axes get distinct hashes.

Pipeline
--------
1. Read batch YAML (LHS axes, N, seed, sim defaults, hull / LOS).
2. Sample N configurations via mc.sampler.latin_hypercube_sample.
3. Dispatch via joblib.Parallel(n_jobs=...) to _run_single (TOP-LEVEL,
   picklable, so joblib's pickle-based dispatch works on spawn).
4. _run_single runs the closed-loop pipeline (LOS + κ̂ + hull) plus the
   FSM/sensor/delay model and writes:
      <out_dir>/run_<hash>/trajectory.parquet
      <out_dir>/run_<hash>/summary.json
5. Aggregate per-run summaries into <out_dir>/_summary.parquet.
6. tqdm progress bar; emit [DONE] / [ERROR] per run.

Parquet schema (normative)
--------------------------
trajectory.parquet columns:
    config_hash : str  — short hash of sampled config (stable across runs)
    t           : float64 [s] — simulation time
    mode        : str  — current FSM mode in {T, L_N, L_G, R}
    d           : float64 [m] — signed lateral distance to wall
    theta       : float64 [rad] — heading misalignment ψ - γ_p
    u           : float64 [m/s] — surge speed
    r           : float64 [rad/s] — yaw rate
    hat_kappa   : float64 [m⁻¹] — curvature estimate
    lost_count  : int32 — cumulative count of T→L transitions
    reacq_time  : float64 [s] — accumulated time spent in R (0 if never)
    collide_flag: int8 — 1 if d ≤ d_min ever, else 0
    terminate_reason : str — one of {COMPLETED, TIMEOUT, COLLIDED, ERROR}
    engine      : str — fixed "py" for sim_python layer

CLI
---
    python -m sim_python.mc.dispatcher \\
        --config sim_python/configs/stage_03_smoke.yaml \\
        --out-dir sim_python/results/stage_03_smoke \\
        --log sim_python/results/stage_03_smoke/dispatcher.log \\
        --jobs 4

Invariants
----------
- ``_run_single`` is defined at module scope (not lambda/closure) so
  joblib's pickle-based dispatch works on Linux/macOS spawn-method.
- All randomness via numpy.random.Generator(seed); never touch global
  numpy.random.* state.
- engine field always "py".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from sim_python.controllers.los_controller import LOSConfig, compute_r_star
from sim_python.controllers.mode_manager import (
    Mode,
    ModeConfig,
    ModeManager,
)
from sim_python.controllers.reacquire_planner import (
    Pose2D,
    ReacquireConfig,
    ReacquirePlanner,
)
from sim_python.mc.sampler import latin_hypercube_sample
from sim_python.models.torpedo_kinematic_2d import (
    HullConfig,
    IDX_PSI,
    IDX_R,
    IDX_U,
    IDX_X,
    IDX_Y,
    STATE_DIM,
    step_l0,
)
from sim_python.scenarios.wall_generator import Wall, gen_arc, gen_straight
from sim_python.tests.test_closed_loop_smoke import signed_distance_to_wall


# --------------------------------------------------------------------------- #
# Schema constants                                                            #
# --------------------------------------------------------------------------- #

TRAJECTORY_COLUMNS: Tuple[str, ...] = (
    "config_hash",
    "t",
    "mode",
    "d",
    "theta",
    "u",
    "r",
    "hat_kappa",
    "lost_count",
    "reacq_time",
    "collide_flag",
    "terminate_reason",
    "engine",
)

# Below this κ_max value the arc is too gentle to be distinguishable
# from a straight wall (R > 100 m); we use a straight wall instead to
# avoid degenerate-large-radius numerical issues.
KAPPA_STRAIGHT_THRESHOLD: float = 0.01

ENGINE_LABEL: str = "py"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable 12-hex-digit hash of a config dict (JSON-key-sorted).

    Callers should pass a dict that includes any block whose value
    varies across runs — typically
    ``{**sampled, **fixed, **sim, **hull, **los}``.  This dispatcher's
    ``_run_single`` builds the canonical wide payload before hashing.
    """
    payload = json.dumps(cfg, sort_keys=True, default=float)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _wide_hash_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the hash payload used by _run_single (sampled+fixed+sim+hull+los).

    The hull and los blocks join the payload so MC batches that sweep
    hull/LOS axes still produce per-config-unique hashes.
    """
    return {
        **cfg.get("sampled", {}),
        **cfg.get("fixed", {}),
        **cfg.get("sim", {}),
        **cfg.get("hull", {}),
        **cfg.get("los", {}),
    }


def build_wall_from_kappa(
    kappa_max: float, v_star: float, T: float, side: str, ds: float
) -> Wall:
    """Construct a wall sized so the AUV traverses it in roughly T seconds.

    For κ_max < KAPPA_STRAIGHT_THRESHOLD we use a straight wall (radius
    1/κ_max would be > 100 m and the closest-point projection would
    degenerate for the AUV's reachable range).  Otherwise we use a
    circular arc with radius 1/κ_max and arc length v*·T (capped at
    one full revolution so the LHS arc generator is well-defined).
    """
    if kappa_max < KAPPA_STRAIGHT_THRESHOLD:
        length = max(10.0, float(v_star) * float(T))
        return gen_straight(length=length, side=side, ds=ds)
    R = 1.0 / float(kappa_max)
    # arc length the AUV would cover at cruise; cap at 0.99 * 2πR so
    # the gen_arc generator does not wrap.
    target = float(v_star) * float(T)
    max_arc = 0.99 * 2.0 * np.pi * R
    arc = min(target, max_arc)
    # And a minimum so the polyline has enough samples.
    arc = max(arc, 10.0 * ds)
    return gen_arc(radius=R, arc_length=arc, side=side, ds=ds)


def _initial_state_for_wall(
    wall: Wall, d_star: float
) -> np.ndarray:
    """Place the AUV on the wall's standoff path at the start.

    For a straight wall: AUV at (0, +d_star) heading +x.
    For an arc started at the origin: AUV on the inner standoff at
    distance d_star, heading along the tangent of the start point.
    """
    p0 = wall.points[0]
    gp0 = float(wall.tangent[0])
    nx = -np.sin(gp0)
    ny = +np.cos(gp0)
    sign = +1.0 if wall.side == "L" else -1.0
    x = float(p0[0] + sign * d_star * nx)
    y = float(p0[1] + sign * d_star * ny)
    psi = gp0
    return np.array([x, y, psi, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Top-level worker (joblib requires picklable, NOT a closure)                 #
# --------------------------------------------------------------------------- #


def _run_single(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one closed-loop sim for sampled config + write parquet + summary.

    Parameters
    ----------
    cfg : dict
        ``{
            "sampled":   {"kappa_max", "v_star", "f_s", "tau_d", ...},
            "fixed":     {"u_gate", "kappa_dot_max", "w_FOV", "sigma_eta",
                          "bar_Vc", ...},
            "sim":       {"T_end", "dt", "d_star", "d_min", "side", "ds",
                          "noise_seed" (optional)},
            "hull":      {"u_max", "r_max", "dot_r_max", "u_gate", ...},
            "los":       {"Delta", "K_p", "K_ff", "K_theta", "d_star", ...},
            "out_dir":   str path,
            "engine":    "py",
        }``

    The function:
    1. Builds the wall + initial state from the sampled κ_max + v*.
    2. Runs closed-loop integration with **sensor noise σ_η** added to the
       lateral-distance measurement and **control delay τ_d** applied to
       the commanded yaw rate (discrete delay line, samples skipped per
       round(τ_d/dt)).
    3. Drives the 4-state FSM (T / L_N / L_G / R) by feeding the χ²
       statistic (lateral residual normalised by σ_η) and a visibility
       predicate (closest-point bearing vs FOV half-angle).
    4. Counts L→R transitions (lost_count), accumulates R-mode time
       (reacq_time), and writes the trajectory parquet.

    Returns
    -------
    dict
        Per-run summary row with ``lost_count``, ``reacq_total_time``,
        ``lost_freq`` (events / s), ``lost_g_count``, ``lost_n_count``.
    """
    t0 = time.perf_counter()
    try:
        sampled = cfg["sampled"]
        fixed = dict(cfg.get("fixed", {}))
        sim = cfg["sim"]
        hull = HullConfig(**cfg["hull"])
        los_dict = dict(cfg["los"])
        los_dict.setdefault("d_star", sim["d_star"])
        los = LOSConfig(**los_dict)

        T_end: float = float(sim["T_end"])
        dt: float = float(sim["dt"])
        d_star: float = float(sim["d_star"])
        d_min: float = float(sim.get("d_min", 0.1))
        side: str = str(sim.get("side", "L"))
        ds: float = float(sim.get("ds", 0.05))

        # Stage-04 perturbation parameters (with sane defaults).
        # σ_η = sensor noise std [m] (Gaussian, i.i.d. per sample).
        # Stage-05: support range-dependent σ_η(d) = a + b·|d|.
        # If sigma_eta_a / sigma_eta_b are present in fixed, treat σ_η as
        # range-dependent and recompute per-step from |d_true|.  Otherwise
        # fall back to the legacy scalar `sigma_eta`.
        sigma_eta_a_raw = _resolve_param(
            sampled, fixed, "sigma_eta_a", default=float("nan"),
        )
        sigma_eta_b_raw = _resolve_param(
            sampled, fixed, "sigma_eta_b", default=float("nan"),
        )
        sigma_eta_range_dep: bool = (
            np.isfinite(sigma_eta_a_raw) and np.isfinite(sigma_eta_b_raw)
        )
        sigma_eta_a: float = float(sigma_eta_a_raw) if sigma_eta_range_dep else 0.0
        sigma_eta_b: float = float(sigma_eta_b_raw) if sigma_eta_range_dep else 0.0
        # Legacy scalar (also used for the *static* mode-config T*_N
        # which depends on σ_κ̂(σ_η)).  Default 0.05 m.
        sigma_eta: float = float(_resolve_param(
            sampled, fixed, "sigma_eta", default=0.05,
        ))
        if sigma_eta_range_dep:
            # Use d_star as the representative range for the *static*
            # T*_N closed form (which needs a single σ_η).  Per-step
            # measurement noise uses the range-dependent value.
            sigma_eta = max(sigma_eta_a + sigma_eta_b * float(sim["d_star"]), 1e-6)
        # τ_d = control delay [s]
        tau_d: float = float(_resolve_param(
            sampled, fixed, "tau_d", default=0.05,
        ))
        # w_FOV (rad) = FOV full width; for visibility we use the
        # half-angle w_FOV/2 as the maximum allowed |relative bearing|.
        w_fov: float = float(_resolve_param(
            sampled, fixed, "w_FOV", default=np.pi / 2,
        ))
        # κ'_max needed for T*_G
        kappa_dot_max: float = float(_resolve_param(
            sampled, fixed, "kappa_dot_max", default=0.01,
        ))
        # u_gate lower bound on surge (used in T*_N / T*_G)
        u_gate: float = float(_resolve_param(
            sampled, fixed, "u_gate", default=0.3,
        ))
        # bar_Vc — cross-current; perturbs lateral d (drift) per dt
        bar_Vc: float = float(_resolve_param(
            sampled, fixed, "bar_Vc", default=0.0,
        ))

        wall = build_wall_from_kappa(
            kappa_max=float(sampled["kappa_max"]),
            v_star=float(sampled["v_star"]),
            T=T_end,
            side=side,
            ds=ds,
        )
        state = _initial_state_for_wall(wall, d_star)

        # u_cmd = v* (the sampled cruise target); clamp to hull.u_max.
        u_cmd = float(min(sampled["v_star"], hull.u_max))

        # ---- Hash payload: widened to {sampled, fixed, sim, hull, los} ----
        chash = config_hash(_wide_hash_payload(cfg))

        # ---- FSM: ModeManager + ReacquirePlanner ----
        # σ_κ̂ for T*_N: use the 720-constant closed form with the
        # window length and arc-length step implied by L=20, s̄=v*·dt
        # (matches the curvature-estimator default).
        L_window = int(cfg.get("estimator", {}).get("window_L", 20))
        s_bar = max(u_cmd * dt, 1e-3)
        sigma_kappa_hat = float(
            np.sqrt(720.0 * sigma_eta * sigma_eta /
                    (L_window ** 5 * s_bar ** 4))
        )
        # Guard against pathological tiny σ_κ̂ (would make T*_N huge);
        # mode_manager requires > 0 / finite.
        sigma_kappa_hat = max(sigma_kappa_hat, 1e-6)
        mode_cfg = ModeConfig(
            d_star=d_star,
            d_min=d_min,
            sigma_kappa_hat=sigma_kappa_hat,
            u_min=u_gate,
            w_fov=w_fov,
            kappa_dot_max=max(kappa_dot_max, 1e-6),
            warmup_s=min(1.0, T_end / 10.0),
        )
        mm = ModeManager(cfg=mode_cfg)
        reac_cfg = ReacquireConfig(w_fov=w_fov, u_min=u_gate)
        reac = ReacquirePlanner(cfg=reac_cfg)
        chi2_thresh = mm.chi2_threshold_value

        # ---- Determinism: seed noise from config_hash + sim.noise_seed ----
        # Mix: take hex hash → int, XOR with sim.noise_seed; produces a
        # deterministic per-config Generator that is independent of the
        # LHS engine state.
        noise_seed = int(sim.get("noise_seed", 0))
        rng_seed = int(chash, 16) ^ noise_seed
        rng = np.random.default_rng(rng_seed & 0xFFFF_FFFF)

        # ---- Discrete control-delay line ----
        delay_steps = int(round(tau_d / dt)) if tau_d > 0 else 0
        r_cmd_queue: List[float] = [0.0] * max(delay_steps, 0)

        # ---- Storage ----
        n_steps = int(np.floor(T_end / dt))
        n_samples = n_steps + 1
        T = np.zeros(n_samples, dtype=np.float64)
        D = np.zeros(n_samples, dtype=np.float64)
        TH = np.zeros(n_samples, dtype=np.float64)
        U = np.zeros(n_samples, dtype=np.float64)
        Rr = np.zeros(n_samples, dtype=np.float64)
        K = np.zeros(n_samples, dtype=np.float64)
        MODE = np.full(n_samples, "T", dtype=object)
        LOST = np.zeros(n_samples, dtype=np.int32)
        REACQ = np.zeros(n_samples, dtype=np.float64)
        COL = np.zeros(n_samples, dtype=np.int8)

        # ---- Initial sample ----
        pos = np.array([state[IDX_X], state[IDX_Y]])
        d0_true, gp0, kappa0 = signed_distance_to_wall(pos, wall)
        T[0] = 0.0
        D[0] = d0_true
        TH[0] = float(state[IDX_PSI]) - gp0
        U[0] = float(state[IDX_U])
        Rr[0] = float(state[IDX_R])
        K[0] = kappa0
        terminate_reason = "COMPLETED"

        lost_n_count = 0
        lost_g_count = 0
        prev_visibility = True

        for k in range(n_steps):
            pos = np.array([state[IDX_X], state[IDX_Y]])
            d_true, gp, kappa_true = signed_distance_to_wall(pos, wall)

            # ---- Sensor noise on measured d ----
            # Stage-05: σ_η can be range-dependent (a + b·|d|).
            if sigma_eta_range_dep:
                sigma_eta_step = max(
                    sigma_eta_a + sigma_eta_b * abs(d_true), 1e-6
                )
            else:
                sigma_eta_step = sigma_eta
            d_meas = d_true + float(rng.normal(0.0, sigma_eta_step))

            # ---- Visibility predicate (Lost-G mechanism) ----
            # Relative bearing of the wall-closest-point in body frame.
            # The AUV "sees" the wall if |bearing| ≤ w_fov/2.
            # bearing = atan2(d_y, d_x) where (d_x, d_y) is the
            # closest-point offset rotated into body frame.
            psi = float(state[IDX_PSI])
            cp_idx = int(np.argmin(np.einsum(
                "ij,ij->i", wall.points - pos[None, :],
                wall.points - pos[None, :],
            )))
            cp_off = wall.points[cp_idx] - pos
            c_psi, s_psi = np.cos(-psi), np.sin(-psi)
            dx_body = c_psi * cp_off[0] - s_psi * cp_off[1]
            dy_body = s_psi * cp_off[0] + c_psi * cp_off[1]
            bearing = float(np.arctan2(dy_body, dx_body))
            visibility = bool(abs(bearing) <= 0.5 * w_fov)

            # ---- χ² statistic on lateral residual ----
            # ν=1 (1-D residual); innovation = (d_meas - d_star);
            # variance = sigma_eta**2.  Statistic = innovation²/variance.
            # This keeps the gate simple; a full 2-D innovation
            # (residual + κ-fit) would tighten it further.
            innov = d_meas - d_star
            chi2_stat = (innov * innov) / max(sigma_eta_step * sigma_eta_step, 1e-12)
            # Use ν=1 threshold for the warmup-adjusted comparison.
            # mm.step compares against ν=cfg.chi2_dof=2 by default; we
            # build it with chi2_dof=2 to match the FSM doc default but
            # multiply our 1-D statistic by 2 to scale (conservative).
            chi2_eff = chi2_stat  # FSM doc default ν=2; this gives a
            # slightly more permissive gate, fine for first detection.

            # ---- Re-acquire success heuristic ----
            # R → T if (a) visibility recovered AND (b) χ² re-passes.
            reac_success = visibility and (chi2_eff <= chi2_thresh)

            # Drive FSM.
            prev_mode = mm.mode
            new_mode = mm.step(
                dt=dt,
                chi2_stat=chi2_eff,
                visibility=visibility,
                reacquire_success=reac_success,
            )

            # Count L→R transitions (each enters R once per loss event).
            if prev_mode != new_mode:
                if new_mode == Mode.L_N:
                    lost_n_count += 1
                elif new_mode == Mode.L_G:
                    lost_g_count += 1

            # ---- Compute LOS command ----
            # In R mode, suppress the lateral correction (the planner
            # would normally take over with waypoint goals) — proxy by
            # zeroing the lateral term; keep the feed-forward κ̂·u.
            if new_mode in (Mode.T, Mode.L_N, Mode.L_G):
                r_cmd_raw = compute_r_star(
                    d=d_meas, theta=float(state[IDX_PSI]) - gp,
                    kappa_hat=kappa_true,
                    u=float(state[IDX_U]), cfg=los,
                )
            else:  # Mode.R
                # Use a constant turn-in to find the wall again — proxy
                # for ReacquirePlanner spiral; small magnitude.
                r_cmd_raw = float(
                    los.K_ff * kappa_true * float(state[IDX_U])
                )

            # ---- Control delay: push raw, pop delayed ----
            if delay_steps > 0:
                r_cmd_queue.append(r_cmd_raw)
                r_cmd = r_cmd_queue.pop(0)
            else:
                r_cmd = r_cmd_raw

            ctrl = np.array([u_cmd, r_cmd])
            state = step_l0(state, ctrl, hull, dt)

            # ---- Cross-current drift (lateral pull on the AUV pose) ----
            if bar_Vc > 0:
                # Apply drift perpendicular to wall normal (worst case)
                # — moves the AUV laterally by bar_Vc * dt per step.
                # Simple model: add to (x, y) directly.
                nx = -np.sin(gp); ny = +np.cos(gp)
                state[IDX_X] += bar_Vc * dt * nx
                state[IDX_Y] += bar_Vc * dt * ny

            k1 = k + 1
            T[k1] = (k + 1) * dt
            D[k1] = d_true
            TH[k1] = float(state[IDX_PSI]) - gp
            U[k1] = float(state[IDX_U])
            Rr[k1] = float(state[IDX_R])
            K[k1] = kappa_true
            MODE[k1] = new_mode.value
            LOST[k1] = lost_n_count + lost_g_count
            if new_mode == Mode.R:
                REACQ[k1] = REACQ[k] + dt
            else:
                REACQ[k1] = REACQ[k]

            prev_visibility = visibility

            if abs(d_true) <= d_min:
                COL[k1] = 1
                terminate_reason = "COLLIDED"
                T = T[: k1 + 1]; D = D[: k1 + 1]; TH = TH[: k1 + 1]
                U = U[: k1 + 1]; Rr = Rr[: k1 + 1]; K = K[: k1 + 1]
                MODE = MODE[: k1 + 1]; LOST = LOST[: k1 + 1]
                REACQ = REACQ[: k1 + 1]; COL = COL[: k1 + 1]
                break
            if not np.all(np.isfinite(state)):
                terminate_reason = "ERROR"
                T = T[: k1 + 1]; D = D[: k1 + 1]; TH = TH[: k1 + 1]
                U = U[: k1 + 1]; Rr = Rr[: k1 + 1]; K = K[: k1 + 1]
                MODE = MODE[: k1 + 1]; LOST = LOST[: k1 + 1]
                REACQ = REACQ[: k1 + 1]; COL = COL[: k1 + 1]
                break

        err = D - d_star
        mean_err = float(np.mean(np.abs(err)))
        max_err = float(np.max(np.abs(err)))
        trackable = bool(
            terminate_reason == "COMPLETED"
            and max_err < 1.0  # heuristic: ≤ 1 m peak vs d_star
        )

        # Stage-05: empirical Lost-G dwell distribution.
        # Walk the MODE array, split into contiguous L_G episodes,
        # measure each episode's duration; emit mean / max / count.
        # These feed the T*_G analytical-vs-empirical overlay (Fig 8).
        lost_g_episodes_s: List[float] = []
        in_lg = False
        lg_start_t = 0.0
        for i_step in range(len(MODE)):
            if MODE[i_step] == "L_G" and not in_lg:
                in_lg = True
                lg_start_t = float(T[i_step])
            elif MODE[i_step] != "L_G" and in_lg:
                in_lg = False
                lost_g_episodes_s.append(float(T[i_step]) - lg_start_t)
        if in_lg and len(T):
            # Trajectory ended while still in L_G.
            lost_g_episodes_s.append(float(T[-1]) - lg_start_t)
        if lost_g_episodes_s:
            T_star_G_emp_mean = float(np.mean(lost_g_episodes_s))
            T_star_G_emp_max = float(np.max(lost_g_episodes_s))
        else:
            T_star_G_emp_mean = 0.0
            T_star_G_emp_max = 0.0
        n_lg_episodes = len(lost_g_episodes_s)

        # Same treatment for Lost-N (for symmetry / debugging).
        lost_n_episodes_s: List[float] = []
        in_ln = False
        ln_start_t = 0.0
        for i_step in range(len(MODE)):
            if MODE[i_step] == "L_N" and not in_ln:
                in_ln = True
                ln_start_t = float(T[i_step])
            elif MODE[i_step] != "L_N" and in_ln:
                in_ln = False
                lost_n_episodes_s.append(float(T[i_step]) - ln_start_t)
        if in_ln and len(T):
            lost_n_episodes_s.append(float(T[-1]) - ln_start_t)
        T_star_N_emp_mean = float(np.mean(lost_n_episodes_s)) if lost_n_episodes_s else 0.0
        T_star_N_emp_max = float(np.max(lost_n_episodes_s)) if lost_n_episodes_s else 0.0
        n_ln_episodes = len(lost_n_episodes_s)

        df = pd.DataFrame(
            {
                "config_hash": np.full(len(T), chash, dtype=object),
                "t": T,
                "mode": MODE,
                "d": D,
                "theta": TH,
                "u": U,
                "r": Rr,
                "hat_kappa": K,
                "lost_count": LOST,
                "reacq_time": REACQ,
                "collide_flag": COL,
                "terminate_reason": np.full(len(T), terminate_reason, dtype=object),
                "engine": np.full(len(T), ENGINE_LABEL, dtype=object),
            }
        )
        df = df[list(TRAJECTORY_COLUMNS)]

        out_dir = Path(cfg["out_dir"]) / f"run_{chash}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "trajectory.parquet")

        total_T = float(T[-1] if len(T) else 0.0)
        total_lost = int(LOST[-1]) if len(LOST) else 0
        summary_row: Dict[str, Any] = {
            "config_hash": chash,
            **{k: float(v) for k, v in sampled.items()},
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in cfg.get("fixed", {}).items()},
            "trackable": trackable,
            "mean_err": mean_err,
            "max_err": max_err,
            "lost_count": total_lost,
            "lost_n_count": int(lost_n_count),
            "lost_g_count": int(lost_g_count),
            "lost_freq": float(total_lost / total_T) if total_T > 0 else 0.0,
            "reacq_total_time": float(REACQ[-1]) if len(REACQ) else 0.0,
            "reacq_mean_time": float(np.mean(REACQ)),
            "collide": bool(COL.any()),
            "terminate_reason": terminate_reason,
            "n_samples": int(len(T)),
            "wall_time_s": float(time.perf_counter() - t0),
            "engine": ENGINE_LABEL,
            "T_star_N": float(mm.T_star_N),
            "T_star_G": float(mm.T_star_G),
            "T_star": float(mm.T_star),
            # Stage-05 empirical dwells (mean over episodes within this run).
            "T_star_G_emp_mean": T_star_G_emp_mean,
            "T_star_G_emp_max": T_star_G_emp_max,
            "n_lg_episodes": int(n_lg_episodes),
            "T_star_N_emp_mean": T_star_N_emp_mean,
            "T_star_N_emp_max": T_star_N_emp_max,
            "n_ln_episodes": int(n_ln_episodes),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary_row, indent=2))
        print(
            f"[DONE] {chash} trackable={trackable} max_err={max_err:.3f} "
            f"lost={total_lost} ({lost_n_count} N + {lost_g_count} G)"
        )
        return summary_row
    except Exception as exc:  # noqa: BLE001 — needed at process boundary
        print(f"[ERROR] {exc!r}", file=sys.stderr)
        return {
            "config_hash": "FAILED",
            "trackable": False,
            "mean_err": float("nan"),
            "max_err": float("nan"),
            "lost_count": -1,
            "lost_n_count": -1,
            "lost_g_count": -1,
            "lost_freq": float("nan"),
            "reacq_total_time": float("nan"),
            "reacq_mean_time": float("nan"),
            "collide": False,
            "terminate_reason": "ERROR",
            "n_samples": 0,
            "wall_time_s": float(time.perf_counter() - t0),
            "engine": ENGINE_LABEL,
            "error_message": str(exc),
        }


def _resolve_param(sampled: Dict[str, Any], fixed: Dict[str, Any],
                   key: str, default: float) -> float:
    """Look up ``key`` first in sampled, then in fixed; else default.

    Lets an axis be promoted from `fixed` to a swept LHS axis without
    code changes — the sampled value automatically overrides.
    """
    if key in sampled:
        return float(sampled[key])
    if key in fixed:
        return float(fixed[key])
    return float(default)


# --------------------------------------------------------------------------- #
# Batch driver                                                                #
# --------------------------------------------------------------------------- #


def _build_cfg_list(batch: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    """Expand a batch YAML into a list of per-run config dicts."""
    axes: Dict[str, Tuple[float, float, str]] = {
        name: tuple(spec) for name, spec in batch["axes"].items()
    }
    N: int = int(batch["N"])
    seed: int = int(batch.get("seed", 42))
    sampled_list = latin_hypercube_sample(axes, N=N, seed=seed)

    fixed = dict(batch.get("fixed", {}))
    sim_defaults = dict(batch["sim"])
    hull_defaults = dict(batch["hull"])
    los_defaults = dict(batch["los"])

    cfgs: List[Dict[str, Any]] = []
    for sampled in sampled_list:
        # Allow sampled to override individual hull/los/sim fields by
        # exact key (e.g. sampled f_s overrides sim.f_s if it were
        # present, though in the smoke configs the four LHS axes are
        # consumed only by the wall builder + u_cmd derivation).
        cfgs.append(
            {
                "sampled": sampled,
                "fixed": fixed,
                "sim": sim_defaults,
                "hull": hull_defaults,
                "los": los_defaults,
                "out_dir": str(out_dir),
                "engine": ENGINE_LABEL,
            }
        )
    return cfgs


def run_batch(
    config_path: str,
    out_dir: str,
    log_path: Optional[str] = None,
    jobs: int = 1,
) -> pd.DataFrame:
    """Execute a full MC batch and write _summary.parquet.

    Parameters
    ----------
    config_path : str
        Path to batch YAML.
    out_dir : str
        Output directory (will be created).  Each per-run trajectory
        lives at ``out_dir/run_<hash>/trajectory.parquet``.
    log_path : str, optional
        File to receive dispatcher log lines.
    jobs : int, default 1
        Number of parallel workers (joblib n_jobs).

    Returns
    -------
    pandas.DataFrame
        The per-run summary table.
    """
    cfg_path = Path(config_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mc.dispatcher")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(lp)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    batch = yaml.safe_load(cfg_path.read_text())
    cfgs = _build_cfg_list(batch, out_path)
    logger.info(f"Batch: {len(cfgs)} configs, jobs={jobs}, seed={batch.get('seed', 42)}")
    # Save the resolved batch YAML alongside outputs.
    (out_path / "config.yaml").write_text(cfg_path.read_text())

    t_start = time.perf_counter()
    if jobs == 1:
        rows = []
        for cfg in tqdm(cfgs, desc="MC", file=sys.stdout):
            rows.append(_run_single(cfg))
    else:
        # joblib loky backend handles spawn-based pickling.
        rows = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
            delayed(_run_single)(cfg) for cfg in tqdm(cfgs, desc="MC", file=sys.stdout)
        )
    wall_s = time.perf_counter() - t_start
    logger.info(f"All runs done in {wall_s:.2f} s ({wall_s / max(len(cfgs), 1):.3f} s/run)")

    df_summary = pd.DataFrame(rows)
    df_summary.to_parquet(out_path / "_summary.parquet")
    df_summary.to_csv(out_path / "_summary.csv", index=False)

    n_ok = int(df_summary["terminate_reason"].eq("COMPLETED").sum()) \
        if "terminate_reason" in df_summary.columns else 0
    n_err = int((df_summary["config_hash"] == "FAILED").sum()) \
        if "config_hash" in df_summary.columns else 0
    logger.info(
        f"Summary: {n_ok}/{len(df_summary)} COMPLETED, {n_err} ERROR; "
        f"trackable rate = {df_summary['trackable'].mean():.2%}"
    )
    return df_summary


# --------------------------------------------------------------------------- #
# CLI entry                                                                   #
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim_python.mc.dispatcher",
        description="MC dispatcher (LHS → closed-loop sim → parquet).",
    )
    p.add_argument("--config", required=True, help="path to batch YAML")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--log", default=None, help="optional log file path")
    p.add_argument("--jobs", type=int, default=1, help="joblib n_jobs (default 1)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    df = run_batch(
        config_path=args.config,
        out_dir=args.out_dir,
        log_path=args.log,
        jobs=args.jobs,
    )
    print(f"[DONE] dispatcher: {len(df)} runs → {args.out_dir}/_summary.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
