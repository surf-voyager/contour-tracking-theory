"""Closed-loop contour-tracking driver (runs on a machine with the HoloOcean engine).

Demonstrates the contour-tracking controller (sim_holoocean.controllers) running
CLOSED-LOOP in HoloOcean's PierHarbor world, with a forward ImagingSonar imaging a
REAL static harbor wall.

PIPELINE PER TICK
-----------------
  1. read ImagingSonar intensity image (RangeBins x AzimuthBins) from `state`
  2. extract wall points: per azimuth bin, nearest range bin with intensity above
     threshold -> (range, bearing); the nearest such point gives the standoff
     distance d and its bearing theta (heading-misalignment proxy)
  3. feed d to the sliding-window curvature estimator -> hat_kappa (+ fit R2)
  4. FSM step (visibility = wall in FOV this frame; chi2 gate on the standoff
     innovation) -> mode in {T, L_N, L_G, R}
  5. LOS + kappa feed-forward -> commanded yaw rate r*
  6. map r* -> rudder fin deflection (validated sign: bottom +delta / top -delta
     => yaw-left). thrust = fixed RPM (constant surge).
  7. FossenInterface.update -> world-frame accel -> env.act -> env.tick

When the sonar returns no wall this frame (visibility=0), we hold the last
hat_kappa feed-forward and let the FSM run its guaranteed-loss dwell, exactly as the
theory prescribes.

OUTPUTS (schema parity with the Layer-1 output + engine="ho")
-------------------------------------------------------------
  trajectory.parquet: config_hash, t, mode, d, theta, u, r, hat_kappa, lost_count,
                      reacq_time, collide_flag, terminate_reason, engine  (+ Layer-2
                      diagnostics: d_err, valid_frame, laser_gt, x, y, yaw)
  sonar_frames.npz:   periodic (RangeBins x AzimuthBins) intensity snapshots + meta
                      for the manuscript figure and re-rendering.

API NOTES
---------
  * Control loop: FossenInterface([name], scenario); set_u_control([r_fin, top_fin,
    left_fin, bottom_fin, thrust_rpm]); loop accel=fossen.update(name,state) ->
    env.act(name,accel) -> state=env.tick().
  * Fin order = [right(0), top(1), left(2), bottom(3), thrust(4)]; fin angles RAD.
    Pure yaw-left torque: bottom(3)=+delta, top(1)=-delta, horizontal fins 0.
  * convert_NWU_to_NED(dyn) -> eta=[x,y,z,roll,pitch,yaw], nu=[u,v,w,p,q,r] (NED body).
  * ImagingSonar value = float32 (RangeBins, AzimuthBins) intensity image; azimuth
    bin j bearing = -Az/2 + (j+0.5)/AzimuthBins * Az (deg), centred on fwd axis.

This module is layer-isolated: it imports nothing from sim_python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from sim_holoocean.controllers.curvature_estimator import CurvatureEstimator
from sim_holoocean.controllers.los_controller import LOSConfig, compute_r_star
from sim_holoocean.controllers.mode_manager import Mode, ModeConfig, ModeManager
from sim_holoocean.controllers.reacquire_planner import (
    Pose2D,
    ReacquireConfig,
    ReacquirePlanner,
)
from sim_holoocean.models.mss_sonar import MSSScanner
from sim_holoocean.scenarios.wall_tracking_world import build_tracking_scenario


# --------------------------------------------------------------------------- #
# Sonar wall-point extraction                                                 #
# --------------------------------------------------------------------------- #


def extract_wall(
    image,
    range_min: float,
    range_max: float,
    azimuth_deg: float,
    intensity_thresh: float,
    min_az_bins: int = 2,
):
    """Extract (d, theta, valid, n_az_hit) from a sonar intensity image.

    For each azimuth bin, take the NEAREST range bin whose intensity exceeds the
    threshold (closest wall return along that beam). The minimum such range across
    all bins is the standoff distance d; the bearing of that nearest hit is theta
    (radians, + = wall to the LEFT of the forward axis given +y-left NWU).

    Returns
    -------
    d : float          nearest wall distance [m] (nan if no wall this frame)
    theta : float      bearing of the nearest wall point [rad] (nan if no wall)
    valid : bool       True iff at least min_az_bins azimuth bins returned a wall
    n_az_hit : int     number of azimuth bins with a return
    """
    if image is None:
        return float("nan"), float("nan"), False, 0
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return float("nan"), float("nan"), False, 0
    n_range, n_az = img.shape
    nearest_range = np.full(n_az, np.nan)
    for j in range(n_az):
        col = img[:, j]
        hits = np.nonzero(col > intensity_thresh)[0]
        if hits.size:
            b = int(hits[0])  # nearest (smallest range bin) above threshold
            nearest_range[j] = range_min + (b + 0.5) / n_range * (range_max - range_min)
    valid_mask = np.isfinite(nearest_range)
    n_az_hit = int(valid_mask.sum())
    if n_az_hit < min_az_bins:
        return float("nan"), float("nan"), False, n_az_hit
    j_near = int(np.nanargmin(nearest_range))
    d = float(nearest_range[j_near])
    # bearing of azimuth bin j: span [-Az/2, +Az/2], bin centre.
    bearing_deg = -azimuth_deg / 2.0 + (j_near + 0.5) / n_az * azimuth_deg
    theta = float(np.deg2rad(bearing_deg))
    return d, theta, True, n_az_hit


def beam_has_return(beam_1d, intensity_thresh: float) -> bool:
    """True iff a single MSS beam (1-D intensity vector) has a wall return.

    Used for the arm-B 360 deg visibility predicate: the MSS polar map records the
    nearest return on each steered beam; vis_t = (the wall is seen on SOME beam in
    the most-recent full sweep).
    """
    b = np.asarray(beam_1d, dtype=np.float64).ravel()
    return bool(b.size and np.any(b > intensity_thresh))


def mss_sweep_visibility(polar_buffer, intensity_thresh: float):
    """(vis, n_angle_hit) over the accumulated MSS 360 deg polar map.

    polar_buffer is (RangeBins x AngleBins). An angle bin "sees" the wall if any of
    its range bins exceeds the intensity threshold. vis_t (arm B) is True iff >= 1
    angle bin in the swept circle holds a return — i.e. the wall is somewhere in the
    360 deg coverage (geometric rigidity: a full-circle FOV is never geometrically
    occluded by a finite-curvature wall, so the 360-degree scanning sonar keeps the
    wall visible on every sweep).
    """
    buf = np.asarray(polar_buffer, dtype=np.float64)
    if buf.ndim != 2 or buf.size == 0:
        return False, 0
    per_angle = np.max(buf, axis=0)
    hit = per_angle > intensity_thresh
    return bool(hit.any()), int(hit.sum())


# --------------------------------------------------------------------------- #
# r* -> rudder deflection                                                     #
# --------------------------------------------------------------------------- #


def r_star_to_fins(r_star: float, k_rudder: float, delta_max_rad: float):
    """Map commanded yaw rate r* [rad/s] to the 4-fin deflection vector (rad).

    Proportional rudder law delta = clip(k_rudder * r*, +/- delta_max). Sign
    convention for a yaw-LEFT turn (r* > 0 => turn toward +yaw in NED):
        bottom(3) = +delta,  top(1) = -delta,  horizontal fins(0,2) = 0.
    Returns [right, top, left, bottom] fin angles in radians.
    """
    delta = float(np.clip(k_rudder * r_star, -delta_max_rad, delta_max_rad))
    return np.array([0.0, -delta, 0.0, +delta], dtype=float)


# --------------------------------------------------------------------------- #
# Config / hashing                                                            #
# --------------------------------------------------------------------------- #


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #


def run_tracking(cfg_path: Path, out_dir: Path, smoke: bool = False,
                 arm: str = "A") -> int:
    import holoocean
    from holoocean.fossen_dynamics import FossenInterface
    from holoocean.fossen_dynamics.helper_functions import convert_NWU_to_NED

    print(f"HoloOcean version: {holoocean.__version__}", flush=True)
    cfg = _load_yaml(cfg_path)

    # Dual-sonar arm. A = forward narrow FOV (geometric occlusion at the corner ->
    # guaranteed-loss episode). B = MSS 360 deg scanning (vis_t never drops; control
    # is driven by the same forward channel, but vis_t is computed from the 360 deg
    # MSS coverage to show the visibility contrast). cfg["arm"] overrides default.
    arm = str(cfg.get("arm", arm)).upper()
    use_mss = (arm == "B")
    print(f"[arm] {arm} ({'MSS 360deg scanning vis_t' if use_mss else 'forward narrow FOV'})",
          flush=True)

    world = cfg.get("world", "PierHarbor")
    ticks_per_sec = int(cfg.get("ticks_per_sec", 30))
    duration_s = float(cfg.get("smoke_duration_s", 30.0) if smoke
                       else cfg.get("duration_s", 240.0))
    sonar_cfg = cfg.get("sonar", {})
    sonar_name = sonar_cfg.get("sensor_name", "ForwardSonar")
    sonar_hz = int(sonar_cfg.get("hz", 5))
    az_deg = float(sonar_cfg.get("azimuth_deg", 120.0))
    range_min = float(sonar_cfg.get("range_min", 1.0))
    range_max = float(sonar_cfg.get("range_max", 40.0))
    range_bins = int(sonar_cfg.get("range_bins", 512))
    az_bins = int(sonar_cfg.get("azimuth_bins", 256))
    intensity_thresh = float(sonar_cfg.get("intensity_thresh", 0.2))

    ctl = cfg.get("controller", {})
    d_star = float(ctl.get("d_star", 8.0))
    d_min = float(ctl.get("d_min", 2.0))
    v_star_rpm = float(ctl.get("thrust_rpm", 400.0))
    Delta = float(ctl.get("Delta", 4.0))
    K_p = float(ctl.get("K_p", 0.4))
    K_ff = float(ctl.get("K_ff", 1.0))
    K_theta = float(ctl.get("K_theta", 0.3))
    los_style = ctl.get("los_style", "sin")
    k_rudder = float(ctl.get("k_rudder", 6.0))
    delta_max_deg = float(ctl.get("delta_max_deg", 15.0))
    delta_max_rad = float(np.deg2rad(delta_max_deg))
    window_L = int(ctl.get("window_L", 8))
    sigma_eta = float(ctl.get("sigma_eta", 0.10))
    u_min = float(ctl.get("u_min", 0.3))
    kappa_dot_max = float(ctl.get("kappa_dot_max", 1e-2))

    location = cfg.get("spawn_location")
    rotation = cfg.get("spawn_rotation")
    frame_dump_every_s = float(cfg.get("frame_dump_every_s", 10.0))

    # MSS (arm B): single-beam scanning sonar swept over a full circle.
    mss_cfg = cfg.get("mss", {})
    mss_name = mss_cfg.get("sensor_name", "MSS")
    mss_hz = float(mss_cfg.get("hz", sonar_hz))
    mss_step = float(mss_cfg.get("step_deg", 6.0))
    mss_start = float(mss_cfg.get("start_deg", -180.0))
    mss_end = float(mss_cfg.get("end_deg", 180.0))
    mss_scan_axis = mss_cfg.get("scan_axis", "roll")
    mss_range_bins = int(mss_cfg.get("range_bins", 256))

    scenario = build_tracking_scenario(
        world=world,
        location=location,
        rotation=rotation,
        ticks_per_sec=ticks_per_sec,
        sonar_kwargs={
            "sensor_name": sonar_name, "hz": sonar_hz, "azimuth_deg": az_deg,
            "range_min": range_min, "range_max": range_max,
            "range_bins": range_bins, "azimuth_bins": az_bins,
        },
        include_laser=True,
        laser_max_distance=range_max,
        include_mss=use_mss,
        mss_kwargs={
            "sensor_name": mss_name, "hz": mss_hz, "range_min": range_min,
            "range_max": range_max, "range_bins": mss_range_bins,
        } if use_mss else None,
    )
    chash = config_hash(scenario)
    print(f"[scenario] world={world} spawn loc={scenario['agents'][0]['location']} "
          f"rot={scenario['agents'][0]['rotation']} hash={chash}", flush=True)
    print(f"[controller] d*={d_star} d_min={d_min} Delta={Delta} K_p={K_p} "
          f"K_ff={K_ff} K_theta={K_theta} thrust_rpm={v_star_rpm} "
          f"k_rudder={k_rudder} delta_max={delta_max_deg}deg L={window_L}", flush=True)

    # ----- controllers -----
    los_cfg = LOSConfig(Delta=Delta, K_p=K_p, K_ff=K_ff, K_theta=K_theta,
                        style=los_style, d_star=d_star)
    s_bar = max(1e-3, (v_star_rpm / 1000.0) * 0.5 if v_star_rpm > 0 else 0.25)
    # s_bar is the approx arc-length between sonar frames; refined below once we know
    # the measured surge. Use a nominal seed; the estimator's 720 closed form uses it.
    curv = CurvatureEstimator(window_L=window_L, s_bar=float(ctl.get("s_bar", 0.5)))
    sigma_kappa = curv.sigma_kappa_hat(sigma_eta)
    w_fov_rad = float(np.deg2rad(az_deg))
    mode_cfg = ModeConfig(
        d_star=d_star, d_min=d_min, sigma_kappa_hat=max(sigma_kappa, 1e-6),
        u_min=u_min, w_fov=w_fov_rad, kappa_dot_max=kappa_dot_max,
        warmup_s=float(ctl.get("warmup_s", 2.0)),
    )
    fsm = ModeManager(cfg=mode_cfg)
    reacq = ReacquirePlanner(cfg=ReacquireConfig(w_fov=w_fov_rad, u_min=u_min))
    print(f"[fsm] sigma_kappa={sigma_kappa:.3e} T*_N={fsm.T_star_N:.2f}s "
          f"T*_G={fsm.T_star_G:.2f}s chi2_thr={fsm.chi2_threshold_value:.3f}", flush=True)

    # chi2 gate innovation: the gate detects MEASUREMENT anomalies (sonar dropout /
    # sudden wall jumps), NOT steady-state standoff tracking error (that is the LOS
    # controller's job). So the innovation is the frame-to-frame change in the wall
    # distance, d_now - d_pred, where d_pred is the previous sonar distance (the wall
    # is smooth so it should not jump between frames). S sizes the plausible
    # per-frame change: measurement noise + the max geometric change ~ u_max *
    # dt_frame. (Using the residual against d* instead would make the gate fire
    # permanently whenever d != d*.)
    dt_frame = 1.0 / max(sonar_hz, 1)
    chi2_motion_std = float(ctl.get("u_max", 1.0)) * dt_frame  # plausible Δd per frame
    S_d = np.array([[max(sigma_eta, 1e-3) ** 2 + chi2_motion_std ** 2]])

    n_ticks = int(duration_s * ticks_per_sec)
    sonar_every = max(1, round(ticks_per_sec / sonar_hz))
    frame_dump_every = max(1, int(frame_dump_every_s * ticks_per_sec))
    dt = 1.0 / ticks_per_sec

    rows: list[dict] = []
    sonar_frames: list[np.ndarray] = []
    frame_meta: list[dict] = []

    lost_count = 0
    reacq_time = 0.0
    collide_flag = 0
    prev_mode = Mode.T
    last_d = d_star
    last_theta = 0.0
    last_kappa = 0.0
    prev_valid_d = float("nan")   # previous valid sonar distance (chi2 innovation)
    valid_frame_count = 0
    sonar_frame_count = 0
    held_visibility = True   # FSM visibility flag, held between sonar frames
    last_dump_t = -1e9       # last sonar-frame-dump sim time [s]
    terminate_reason = "COMPLETED"

    # arm-B MSS scanner: steers a single beam over the full circle and accumulates a
    # 360 deg polar map. vis_mss = wall seen on >=1 swept beam in the most-recent
    # FULL sweep (so we do not falsely "lose" the wall mid-sweep before the beam has
    # pointed at it). n_steps_per_sweep beams span [start,end].
    scanner = None
    mss_step_count = 0
    n_steps_sweep = max(1, int(round(abs(mss_end - mss_start) / max(mss_step, 1e-6))))
    mss_vis_window = []  # rolling window of last n_steps_sweep beam-hit booleans
    if use_mss:
        scanner = MSSScanner(
            sensor_name=mss_name, scan_axis=mss_scan_axis,
            start=mss_start, end=mss_end, step=mss_step, mode="continuous",
            range_min=range_min, range_max=range_max, range_bins=mss_range_bins,
            angle_resolution=max(mss_step / 2.0, 0.5),
        )
        print(f"[mss] axis={mss_scan_axis} step={mss_step}deg "
              f"sweep=[{mss_start},{mss_end}] beams/sweep={n_steps_sweep} "
              f"range_bins={mss_range_bins}", flush=True)

    thrust = v_star_rpm
    u_control = np.array([0.0, 0.0, 0.0, 0.0, thrust], dtype=float)

    print(f"[run] {'SMOKE ' if smoke else ''}duration={duration_s}s "
          f"n_ticks={n_ticks} sonar_every={sonar_every}tk "
          f"frame_dump_every={frame_dump_every}tk", flush=True)

    with holoocean.make(scenario_cfg=scenario, show_viewport=False) as env:
        fossen = FossenInterface([scenario["main_agent"]], scenario)
        fossen.set_u_control(scenario["main_agent"], u_control)
        # arm-B: steer the MSS beam to its start angle before the first frame
        if use_mss and scanner is not None:
            try:
                env.agents[scenario["main_agent"]].sensors[mss_name].rotate(
                    scanner.next_rotation())
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] initial MSS rotate failed: {exc}", flush=True)
        # settle + build octree at spawn pose
        state = env.tick(20)

        for tick in range(n_ticks):
            t_sim = tick * dt
            dyn = state.get("DynamicsSensor") if isinstance(state, dict) else None
            if dyn is None or not np.all(np.isfinite(dyn)):
                terminate_reason = "ERROR"
                print(f"[ERROR] DynamicsSensor missing/NaN at tick {tick}", flush=True)
                break
            eta, nu = convert_NWU_to_NED(dyn)
            x, y, z = float(eta[0]), float(eta[1]), float(eta[2])
            yaw = float(eta[5])
            u_surge = float(nu[0])
            r_yaw = float(nu[5])

            # ---- sonar measurement ----
            # The ImagingSonar publishes its frame only on its own scheduled ticks
            # (Hz=5 => every 6th tick at 30 tps), and the phase is offset from tick 0
            # by the settle ticks. So we DETECT a fresh frame by the sensor key being
            # PRESENT in the state dict, not by a tick-phase predicate (a tick-phase
            # predicate can fail to align and yield 0 valid frames).
            img = state.get(sonar_name) if isinstance(state, dict) else None
            is_sonar_frame = img is not None
            d = last_d
            theta = last_theta
            valid = False
            d_innov = None  # frame-to-frame standoff change (chi2 innovation)
            if is_sonar_frame:
                d_meas, theta_meas, valid, n_hit = extract_wall(
                    img, range_min, range_max, az_deg, intensity_thresh)
                sonar_frame_count += 1
                if valid:
                    valid_frame_count += 1
                    if prev_valid_d == prev_valid_d:  # have a prior valid d
                        d_innov = d_meas - prev_valid_d
                    prev_valid_d = d_meas
                    d, theta = d_meas, theta_meas
                    last_d, last_theta = d, theta
                    last_kappa = curv.update(d)
                # periodic frame dump for the figure / re-rendering
                if (t_sim - last_dump_t) >= frame_dump_every_s - 1e-6:
                    last_dump_t = t_sim
                    sonar_frames.append(np.asarray(img, dtype=np.float32))
                    frame_meta.append({
                        "t": t_sim, "x": x, "y": y, "yaw": yaw, "d": d,
                        "valid": bool(valid), "n_az_hit": int(n_hit),
                    })

            hat_kappa = last_kappa

            # laser ground-truth cross-check (Stage-02-confirmed reliable channel)
            las = state.get("Laser") if isinstance(state, dict) else None
            laser_gt = float(np.asarray(las).ravel()[0]) if las is not None else float("nan")
            if not (laser_gt > 0):
                laser_gt = float("nan")

            # ---- arm-B: MSS 360 deg sweep + visibility ----
            # Steer the single beam, accumulate the polar map, and recompute the
            # 360 deg visibility predicate over the most-recent FULL sweep. The MSS is
            # NOT the control sensor here (the forward channel drives the loop, so both
            # arms fly the identical trajectory); we log vis_mss to demonstrate that
            # the full-circle FOV never loses the wall (geometric rigidity: the
            # 360-degree scanning sonar keeps the wall visible on every sweep).
            mss_is_frame = False
            mss_beam_hit = None
            vis_t = None
            if use_mss and scanner is not None:
                mss_img = state.get(mss_name) if isinstance(state, dict) else None
                if mss_img is not None:
                    mss_is_frame = True
                    beam = np.asarray(mss_img)
                    beam_1d = beam[:, beam.shape[1] // 2] if beam.ndim == 2 else beam
                    mss_beam_hit = beam_has_return(beam_1d, intensity_thresh)
                    scanner.update(state)  # writes beam into polar map, advances scan
                    mss_step_count += 1
                    # rolling window over one full sweep of beam-hit flags
                    mss_vis_window.append(bool(mss_beam_hit))
                    if len(mss_vis_window) > n_steps_sweep:
                        mss_vis_window.pop(0)
                    # re-steer the beam for the next frame
                    try:
                        env.agents[scenario["main_agent"]].sensors[mss_name].rotate(
                            scanner.next_rotation())
                    except Exception:  # noqa: BLE001
                        pass
            # arm-B vis_t: wall seen on >=1 beam in the last full sweep (or the
            # accumulated polar map has any return). True once the sweep has begun.
            if use_mss:
                if mss_vis_window:
                    vis_mss_sweep = any(mss_vis_window)
                else:
                    vis_mss_sweep = True  # not yet swept; assume covered (spawn faces wall)
                vis_map, n_mss_ang = mss_sweep_visibility(scanner.image, intensity_thresh) \
                    if scanner is not None else (False, 0)
                vis_t = bool(vis_mss_sweep or vis_map)
            else:
                vis_t = None  # arm A: vis_t == forward valid (set below)

            # ---- FSM step ----
            # Visibility + chi2 are only re-evaluated on sonar frames; between frames
            # the last sonar decision is held (the sonar is the only wall sensor).
            # Arm A: visibility = wall within the (narrow) forward FOV this frame.
            # Arm B: visibility = wall anywhere in the MSS 360 deg sweep (vis_t above).
            if is_sonar_frame:
                if use_mss:
                    held_visibility = bool(vis_t)
                else:
                    held_visibility = bool(valid)
                if valid and d_innov is not None:
                    resid = np.array([d_innov])
                    chi2_stat = ModeManager.chi2_statistic(resid, S_d)
                else:
                    chi2_stat = None
            elif use_mss and mss_is_frame:
                # an MSS frame can refresh arm-B visibility between forward frames
                held_visibility = bool(vis_t)
                chi2_stat = None
            else:
                chi2_stat = None  # no fresh measurement -> no chi2-driven transition
            mode = fsm.step(dt=dt, chi2_stat=chi2_stat,
                            visibility=held_visibility,
                            reacquire_success=False)
            # detect T -> Lost transition for lost_count
            if prev_mode == Mode.T and mode in (Mode.L_N, Mode.L_G):
                lost_count += 1
            if mode == Mode.R:
                reacq_time += dt
                # trivial re-acquire: if a valid wall reappears, FSM L_*->T handles it;
                # in R we let the spiral search run (waypoints computed but the demo
                # keeps surge+gentle turn toward last side until reacquisition).
                _ = reacq  # planner available; geometry-fixed demo uses passive search
            prev_mode = mode

            # ---- guidance: LOS + kappa feed-forward ----
            d_for_los = d if np.isfinite(d) else d_star
            theta_for_los = theta if np.isfinite(theta) else 0.0
            r_star = compute_r_star(d=d_for_los, theta=theta_for_los,
                                    kappa_hat=hat_kappa, u=max(u_surge, 0.05),
                                    cfg=los_cfg)
            # in Lost / Reacquire, bias a gentle search turn toward the wall side
            if mode in (Mode.L_G, Mode.R):
                r_star += 0.15  # gentle turn-toward-wall during search

            fins = r_star_to_fins(r_star, k_rudder, delta_max_rad)
            u_control[:4] = fins
            u_control[4] = thrust
            fossen.set_u_control(scenario["main_agent"], u_control)

            # collision check
            d_err = abs(d - d_star) if np.isfinite(d) else float("nan")
            if np.isfinite(d) and d <= d_min:
                collide_flag = 1

            # vis_t timeline: the visibility predicate actually fed to the FSM this
            # tick (held between sonar frames). Arm A: forward-FOV visibility. Arm B:
            # MSS 360 deg coverage. wall_bearing_deg = forward nearest-wall bearing
            # (in arm A this exceeds +/- Az/2 at the guaranteed-loss episode, i.e. the
            # wall tangent has rotated out of the forward fan).
            vis_t_log = bool(held_visibility)
            wall_bearing_deg = float(np.rad2deg(theta)) if np.isfinite(theta) else float("nan")
            rows.append({
                "config_hash": chash,
                "t": float(t_sim),
                "mode": mode.value,
                "d": float(d) if np.isfinite(d) else float("nan"),
                "theta": float(theta) if np.isfinite(theta) else float("nan"),
                "u": float(u_surge),
                "r": float(r_yaw),
                "hat_kappa": float(hat_kappa),
                "lost_count": int(lost_count),
                "reacq_time": float(reacq_time),
                "collide_flag": int(collide_flag),
                "terminate_reason": terminate_reason,
                "engine": "ho",
                # Layer-2 diagnostics (beyond the shared core schema)
                "d_err": float(d_err) if np.isfinite(d_err) else float("nan"),
                "valid_frame": bool(valid),
                "vis_t": vis_t_log,
                "wall_bearing_deg": wall_bearing_deg,
                "arm": arm,
                "mss_beam_hit": bool(mss_beam_hit) if mss_beam_hit is not None else False,
                "r_star_cmd": float(r_star),
                "laser_gt": float(laser_gt),
                "x": x, "y": y, "yaw": yaw,
            })

            # ---- advance sim ----
            accel = fossen.update(scenario["main_agent"], state)
            if not np.all(np.isfinite(accel)):
                terminate_reason = "ERROR"
                print(f"[ERROR] non-finite accel at tick {tick}", flush=True)
                break
            env.act(scenario["main_agent"], accel)
            state = env.tick()

            if tick % (ticks_per_sec * 10) == 0:
                print(f"  t={t_sim:6.1f}s mode={mode.value:3s} d={d:6.2f} "
                      f"d_err={d_err:5.2f} theta={np.rad2deg(theta_for_los):+6.1f}deg "
                      f"kappa={hat_kappa:+.4f} u={u_surge:.2f} r*={r_star:+.3f} "
                      f"pos=({x:+.1f},{y:+.1f}) valid_frac="
                      f"{valid_frame_count/max(sonar_frame_count,1):.2f}", flush=True)

    # finalise terminate_reason in all rows
    for row in rows:
        row["terminate_reason"] = terminate_reason

    valid_frac = valid_frame_count / max(sonar_frame_count, 1)
    # Lost-G episode count: T -> L_G transitions in the mode timeline. Re-acquire
    # success: an L_G (or R) episode that returns to T.
    modes_seq = [r["mode"] for r in rows]
    lost_g_episodes = 0
    reacq_back_to_T = 0
    in_loss = False
    for i in range(1, len(modes_seq)):
        prev, cur = modes_seq[i - 1], modes_seq[i]
        if prev == "T" and cur == "L_G":
            lost_g_episodes += 1
            in_loss = True
        if in_loss and cur == "T" and prev in ("L_G", "R", "L_N"):
            reacq_back_to_T += 1
            in_loss = False
    # vis_t fraction (the FSM visibility predicate actually applied)
    vis_vals = [bool(r.get("vis_t", True)) for r in rows]
    vis_frac = (sum(vis_vals) / len(vis_vals)) if vis_vals else float("nan")
    mss_hits = sum(1 for r in rows if r.get("mss_beam_hit", False))
    print(f"[summary] arm={arm} rows={len(rows)} sonar_frames={sonar_frame_count} "
          f"valid_frames={valid_frame_count} valid_frac={valid_frac:.3f} "
          f"vis_t_frac={vis_frac:.3f} lost_count={lost_count} "
          f"lost_g_episodes={lost_g_episodes} reacq_to_T={reacq_back_to_T} "
          f"mss_step_frames={mss_step_count} mss_beam_hits={mss_hits} "
          f"reacq_time={reacq_time:.1f}s collide_flag={collide_flag} "
          f"terminate={terminate_reason}", flush=True)
    # mode timeline
    modes = [r["mode"] for r in rows]
    if modes:
        from collections import Counter
        print(f"[modes] {dict(Counter(modes))}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"run_arm{arm}_{chash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(run_dir / "trajectory.parquet", rows)
    _write_csv(run_dir / "trajectory.csv", rows)
    if sonar_frames:
        npz_path = run_dir / "sonar_frames.npz"
        np.savez_compressed(
            npz_path,
            frames=np.stack(sonar_frames, axis=0),
            meta=np.array([json.dumps(m) for m in frame_meta], dtype=object),
            range_min=range_min, range_max=range_max, azimuth_deg=az_deg,
            range_bins=range_bins, azimuth_bins=az_bins,
        )
        print(f"Sonar frames saved: {npz_path} ({len(sonar_frames)} frames)", flush=True)

    summary = {
        "config_hash": chash, "arm": arm, "world": world, "duration_s": duration_s,
        "n_rows": len(rows), "sonar_frames": sonar_frame_count,
        "valid_frames": valid_frame_count, "valid_frac": valid_frac,
        "vis_t_frac": vis_frac, "azimuth_deg": az_deg,
        "lost_count": lost_count, "lost_g_episodes": lost_g_episodes,
        "reacq_to_T": reacq_back_to_T, "reacq_time": reacq_time,
        "mss_step_frames": mss_step_count, "mss_beam_hits": mss_hits,
        "collide_flag": collide_flag, "terminate_reason": terminate_reason,
        "T_star_N": fsm.T_star_N, "T_star_G": fsm.T_star_G,
        "d_star": d_star, "d_min": d_min, "engine": "ho", "smoke": smoke,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {run_dir / 'summary.json'}", flush=True)

    if smoke and valid_frame_count == 0:
        print("[ERROR] SMOKE FAIL: 0 valid sonar frames (wall not imaged). "
              "Adjust spawn rotation / intensity_thresh.", flush=True)
        return 2
    print("[DONE]", flush=True)
    return 0


def _write_parquet(path: Path, rows: list[dict]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"[WARN] parquet backend unavailable ({exc}); CSV only.", flush=True)
        return False
    if not rows:
        print("[WARN] no rows to write to parquet", flush=True)
        return False
    table = pa.table({k: [r[k] for r in rows] for k in rows[0]})
    pq.write_table(table, path)
    nrows = pq.read_table(path).num_rows
    print(f"Parquet write OK: {path} ({nrows} rows)", flush=True)
    return True


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-03 closed-loop wall tracking")
    parser.add_argument("--config", default="sim_holoocean/configs/stage_03_pierharbor.yaml")
    parser.add_argument("--out-dir", default="sim_holoocean/results/stage_03_tracking")
    parser.add_argument("--smoke", action="store_true",
                        help="short run (smoke_duration_s) verifying sonar+loop")
    parser.add_argument("--arm", default="A", choices=["A", "B"],
                        help="dual-sonar arm: A=forward narrow FOV "
                             "(geometric occlusion -> Lost-G), B=MSS 360deg scanning "
                             "(vis_t never drops). cfg['arm'] overrides.")
    args = parser.parse_args(argv)
    try:
        return run_tracking(Path(args.config), Path(args.out_dir), smoke=args.smoke,
                            arm=args.arm)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
