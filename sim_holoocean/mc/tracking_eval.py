"""Evaluate Layer-2 wall-tracking performance against LiDAR ground truth.

The forward/side sonar's 2-D image cannot serve as truth, so the controller is
scored against the per-frame LiDAR true lateral standoff (d_true_lidar, logged by
run_tracking.py --lidar). Reads a run directory's trajectory.parquet (+ summary.json
for d_star/d_min) and reports:

  * collision (min true standoff vs d_min),
  * TRUE standoff error |d_true - d_star|: RMS, max, and steady-state (last 50%),
  * valid-frame fraction, depth (z) stability,
  * curvature estimate kappa_hat vs the true wall curvature (~0 for a straight wall),
  * lost-episode durations vs T*_G,
  * sonar-d vs LiDAR-truth gap (how much the 2-D sonar misreads the standoff).

Optionally overlays a top-down figure: trajectory over the true wall points (octree
slice .npz) with the d_star band, plus standoff-vs-time (sonar / truth / set-point).

Usage:
    python sim_holoocean/mc/tracking_eval.py <run_dir> [--t-star-g 48.2]
    # optional: --wall-npz <octree_slice.npz> overlays the true wall on the figure.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lost_episodes(modes, dt):
    """Durations [s] of consecutive Lost/Reacquire spans."""
    lost = np.array([m in ("L_N", "L_G", "R") for m in modes])
    eps, run = [], 0
    for v in lost:
        if v:
            run += 1
        elif run:
            eps.append(run * dt); run = 0
    if run:
        eps.append(run * dt)
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--wall-npz", default=None, help="octree depth-slice npz for the true wall")
    ap.add_argument("--d-star", type=float, default=None, help="override standoff set-point")
    ap.add_argument("--d-min", type=float, default=None)
    ap.add_argument("--t-star-g", type=float, default=None, help="T*_G [s] for loss check")
    ap.add_argument("--rms-tol", type=float, default=1.0, help="steady-state RMS tolerance [m]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = args.run_dir
    df = pd.read_parquet(run / "trajectory.parquet")
    summ = {}
    if (run / "summary.json").exists():
        summ = json.loads((run / "summary.json").read_text())
    d_star = args.d_star if args.d_star is not None else float(summ.get("d_star", 10.0))
    d_min = args.d_min if args.d_min is not None else float(summ.get("d_min", 4.0))

    t = df["t"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    dtrue = df["d_true_lidar"].to_numpy()
    dson = df["d"].to_numpy()
    fin = np.isfinite(dtrue)
    half = t > (t[0] + 0.5 * (t[-1] - t[0]))   # steady-state window (last 50%)

    e = dtrue - d_star
    ss = fin & half
    rms_ss = float(np.sqrt(np.nanmean(e[ss] ** 2))) if ss.any() else float("nan")
    rms_all = float(np.sqrt(np.nanmean(e[fin] ** 2))) if fin.any() else float("nan")
    min_true = float(np.nanmin(dtrue[fin])) if fin.any() else float("nan")
    # valid_frame is only set on sonar frames (1-in-N ticks); the summary's valid_frac
    # is over sonar frames, which is the meaningful visibility metric.
    valid_frac = float(summ.get("valid_frac",
                                df["valid_frame"].mean() if "valid_frame" in df else float("nan")))
    z_std = float(df["z"].std()) if "z" in df else float("nan")
    kappa_abs = float(np.nanmean(np.abs(df["hat_kappa"]))) if "hat_kappa" in df else float("nan")
    gap = float(np.nanmean(np.abs(dson[fin] - dtrue[fin]))) if fin.any() else float("nan")
    eps = lost_episodes(df["mode"].tolist(), dt) if "mode" in df else []
    max_lost = max(eps) if eps else 0.0
    collided_truth = min_true <= d_min
    collide_flag = int(summ.get("collide_flag", df.get("collide_flag", pd.Series([0])).max()))

    # ---- verdict ----
    checks = {
        "no_collision (min_true>d_min)": (not collided_truth, f"min_true={min_true:.2f} > d_min={d_min}"),
        "steady RMS<=tol": (rms_ss <= args.rms_tol, f"RMS_ss={rms_ss:.2f} <= {args.rms_tol}"),
        "valid_frac>=0.9": (valid_frac >= 0.9, f"valid_frac={valid_frac:.2f}"),
        "z stable (std<=0.1)": (z_std <= 0.1, f"z_std={z_std:.3f}"),
    }
    if args.t_star_g is not None:
        checks["losses<=T*_G"] = (max_lost <= args.t_star_g,
                                  f"max_lost={max_lost:.1f}s <= T*_G={args.t_star_g}s")
    passed = all(ok for ok, _ in checks.values())

    print(f"\n===== tracking eval: {run.name} =====")
    print(f"d* = {d_star} m,  d_min = {d_min} m,  dt = {dt*1000:.1f} ms,  T = {t[-1]-t[0]:.0f} s")
    print(f"true standoff (LiDAR): min={min_true:.2f}  RMS_all|e|={rms_all:.2f}  "
          f"RMS_ss|e|={rms_ss:.2f} m")
    print(f"sonar-vs-truth gap (mean |d_sonar-d_true|): {gap:.2f} m")
    print(f"valid_frac={valid_frac:.3f}  z_std={z_std:.4f} m  mean|kappa_hat|={kappa_abs:.4f} /m")
    print(f"lost episodes: n={len(eps)} max={max_lost:.1f}s  collide_flag(sonar)={collide_flag}")
    print("checks:")
    for name, (ok, msg) in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name:28s} {msg}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")

    metrics = dict(run=run.name, d_star=d_star, d_min=d_min, min_true=min_true,
                   rms_all=rms_all, rms_ss=rms_ss, sonar_truth_gap=gap,
                   valid_frac=valid_frac, z_std=z_std, mean_abs_kappa_hat=kappa_abs,
                   n_lost=len(eps), max_lost_s=max_lost, collide_flag=collide_flag,
                   collided_truth=bool(collided_truth), passed=bool(passed))
    (run / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figure ----
    out = args.out or str(run / "tracking_eval.png")
    fig, (axm, axd) = plt.subplots(1, 2, figsize=(15, 6))
    # NED y is flipped vs world; trajectory logged NED -> world for overlay with octree
    txw, tyw = df["x"].to_numpy(), -df["y"].to_numpy()
    if args.wall_npz and Path(args.wall_npz).exists():
        z = np.load(args.wall_npz, allow_pickle=True)
        W, N = z["pts"], z["normals"]
        slope = np.degrees(np.arccos(np.clip(np.abs(N[:, 2]), 0, 1)))
        steep = slope >= 60
        # show only wall points near the trajectory bbox
        m = ((W[:, 0] > txw.min() - 20) & (W[:, 0] < txw.max() + 20) &
             (W[:, 1] > tyw.min() - 20) & (W[:, 1] < tyw.max() + 20))
        axm.scatter(W[m & steep, 0], W[m & steep, 1], s=4, c="k", alpha=0.4, label="true wall (octree)")
    axm.plot(txw, tyw, "-", color="tab:blue", lw=2, label="AUV trajectory")
    axm.plot(txw[0], tyw[0], "go", ms=8, label="start")
    axm.plot(txw[-1], tyw[-1], "rs", ms=8, label="end")
    axm.set_aspect("equal"); axm.set_xlabel("world x [m]"); axm.set_ylabel("world y [m]")
    axm.set_title("(a) trajectory over true wall"); axm.legend(fontsize=8); axm.grid(alpha=0.3)

    axd.plot(t, dson, color="tab:orange", lw=1, alpha=0.8, label="sonar d (control input)")
    axd.plot(t, dtrue, color="tab:green", lw=1.5, label="LiDAR true standoff")
    axd.axhline(d_star, color="k", ls="--", lw=1, label=f"d* = {d_star} m")
    axd.axhline(d_min, color="r", ls=":", lw=1, label=f"d_min = {d_min} m")
    axd.set_xlabel("t [s]"); axd.set_ylabel("standoff [m]")
    axd.set_title(f"(b) standoff vs time  (RMS_ss|e|={rms_ss:.2f} m, "
                  f"{'PASS' if passed else 'FAIL'})")
    axd.legend(fontsize=8); axd.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"[eval] wrote {out}  and {run/'eval_metrics.json'}")


if __name__ == "__main__":
    main()
