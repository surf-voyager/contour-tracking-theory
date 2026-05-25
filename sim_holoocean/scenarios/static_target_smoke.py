"""Static-target sonar range-fidelity check (runs on a machine with the HoloOcean engine).

Validates HoloOcean 2.3.0 sonar *range* fidelity against a ground-truth distance,
for BOTH sonar archetypes:

  * "scanning" -> HoloOcean ProfilingSonar (wide azimuth fan)   [models/scanning_sonar.yaml]
  * "forward"  -> HoloOcean ImagingSonar  (90 deg azimuth)      [models/forward_sonar.yaml]

── EXPERIMENT DESIGN ───────────────────────────────────────────────────────────
Two HoloOcean facts, established empirically, drive the design:

  (D1) HoloOcean sonars ray-trace the STATIC-geometry OCTREE only. Targets created
       with env.spawn_prop("box", ...) are DYNAMIC actors and are INVISIBLE to the
       sonar (the sonar peak range is identical with/without a spawned box). So we
       cannot use a spawned prop as the sonar target.
  (D2) The world's static seafloor returns sparse/no sonar energy (specular surface
       at normal incidence); a co-located RangeFinderSensor (laser raycast) DOES
       reliably range the seafloor and tracks altitude exactly.

Therefore the controlled "distance" is the agent's altitude above the seafloor,
realised by spawning a FRESH env per distance at a chosen altitude looking DOWN
(pitch -90). GROUND TRUTH is the laser RangeFinder range to the seafloor (D2). For
each distance we record, per sonar:
  - the sonar peak-intensity range (if any return at all),
  - the sonar nonzero-frame fraction (detection rate),
  - the laser ground-truth range,
and compute the sonar-vs-laser range error + empirical sonar range std.

If the sonar yields ZERO returns across all distances (D2 worst case), that is the
finding: HoloOcean's bare-OpenWater sonar is not a usable range sensor for this
controlled static-target test — the closed-loop demo must either (a) use a world
with rich static geometry (e.g. PierHarbor) or (b) use the laser RangeFinder as the
range channel. This is recorded, not silently passed.

API NOTES:
  * holoocean.make(scenario_cfg=<dict>, show_viewport=False) -> env  (headless).
    scenario_cfg MUST carry "package_name" (e.g. "Ocean") to locate the binary.
  * env.agents["auv0"].teleport(location, rotation) — but the sonar octree does NOT
    regenerate cleanly after teleport, so we use a fresh env per distance instead.
  * state = env.tick(num_ticks) -> dict keyed by sensor_name; sonar value is a
    float32 (RangeBins, AzimuthBins) intensity image (empty scene -> all zeros);
    RangeFinderSensor value is a 1-D array of laser ranges (negative if no hit).
  * Coordinate frame: right-handed, +x fwd, +y left, +z up, metres. Pitch -90 deg
    points the SonarSocket forward axis downward (toward the seafloor).
  * OpenWater cached octree exists for octree_min=0.02, octree_max=5.0.

This module is layer-isolated: it imports nothing from sim_python.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Nominal range-noise models sigma_eta(d) (metres). These are the paper's sonar
# noise parameters, inlined here so the check is self-contained.
SIGMA_ETA = {
    "scanning": lambda d: 0.02 + 0.005 * d,
    "forward": lambda d: 0.05 + 0.01 * d,
}


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _peak_range(image, range_min: float, range_max: float, range_bins: int) -> float:
    """Scalar range estimate from a (RangeBins, AzimuthBins) intensity image.

    Collapse azimuth -> 1-D range profile, take peak bin, map to metric range via
    linear bin spacing. NaN if empty (no detection).
    """
    if image is None:
        return float("nan")
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        return float("nan")
    profile = img.sum(axis=1) if img.ndim == 2 else img
    if not np.any(profile > 0):
        return float("nan")
    peak_bin = int(np.argmax(profile))
    return range_min + (peak_bin + 0.5) / range_bins * (range_max - range_min)


def _sonar_meta(model: dict) -> dict:
    c = model["configuration"]
    return {
        "sensor_type": model["sensor_type"],
        "range_min": c.get("RangeMin", 0.1),
        "range_max": c.get("RangeMax", 60),
        "range_bins": c.get("RangeBins", 512),
        "azimuth": c.get("Azimuth"),
        "azimuth_bins": c.get("AzimuthBins", 256),
        "init_octree_range": c.get("InitOctreeRange", 50),
        "elevation": c.get("Elevation"),
        "hz": model.get("Hz", 10),
    }


def _build_scenario(cfg: dict, torpedo: dict, sonar_meta: dict, sonar_name: str,
                    loc, rot) -> dict:
    """Single-sonar + laser scenario at a fixed look-down pose (fresh env per distance)."""
    sonar_cfg = {
        "RangeMin": sonar_meta["range_min"],
        "RangeMax": sonar_meta["range_max"],
        "RangeBins": sonar_meta["range_bins"],
        "AzimuthBins": sonar_meta["azimuth_bins"],
        "InitOctreeRange": sonar_meta["init_octree_range"],
        "ShowWarning": False,
    }
    if sonar_meta["azimuth"] is not None:
        sonar_cfg["Azimuth"] = sonar_meta["azimuth"]
    if sonar_meta["elevation"] is not None:
        sonar_cfg["Elevation"] = sonar_meta["elevation"]

    agent = {
        "agent_name": torpedo.get("agent_name", "auv0"),
        "agent_type": torpedo.get("agent_type", "TorpedoAUV"),
        "control_scheme": torpedo.get("control_scheme", 0),
        "location": list(loc),
        "rotation": list(rot),
        "sensors": [
            {"sensor_type": "PoseSensor", "socket": "IMUSocket"},
            {"sensor_type": sonar_meta["sensor_type"], "sensor_name": sonar_name,
             "socket": "SonarSocket", "Hz": sonar_meta["hz"], "configuration": sonar_cfg},
            {"sensor_type": "RangeFinderSensor", "sensor_name": "Laser",
             "socket": "SonarSocket",
             "configuration": {"LaserMaxDistance": float(sonar_meta["range_max"]),
                               "LaserCount": 1, "LaserAngle": 0}},
        ],
    }
    return {
        "name": "stage_02_static_target_smoke",
        "package_name": cfg.get("package_name", "Ocean"),
        "world": cfg.get("world", "OpenWater"),
        "main_agent": agent["agent_name"],
        "ticks_per_sec": int(cfg.get("ticks_per_sec", 30)),
        "frames_per_sec": False,
        "octree_min": float(cfg.get("octree_min", 0.02)),
        "octree_max": float(cfg.get("octree_max", 5.0)),
        "agents": [agent],
    }


def run_smoke(cfg_path: Path, out_dir: Path) -> int:
    import holoocean

    print(f"HoloOcean version: {holoocean.__version__}", flush=True)

    repo_root = Path.cwd()
    cfg = _load_yaml(cfg_path)
    scanning = _load_yaml(repo_root / cfg["scanning_sonar_model"])
    forward = _load_yaml(repo_root / cfg["forward_sonar_model"])
    torpedo = _load_yaml(repo_root / cfg["torpedo_model"])
    print(f"Loaded models: torpedo={torpedo.get('agent_type')} "
          f"scanning={scanning['sensor_type']} forward={forward['sensor_type']}", flush=True)

    sonars = {
        "scanning": {"sensor_name": "ScanningSonar", "meta": _sonar_meta(scanning)},
        "forward": {"sensor_name": "ForwardSonar", "meta": _sonar_meta(forward)},
    }
    print(f"Scanning sonar (ProfilingSonar) Azimuth={sonars['scanning']['meta']['azimuth']} deg "
          f"(paper w_FOV=360 -> capped by the fixed-fan engine; documented deviation)", flush=True)
    print(f"Forward sonar  (ImagingSonar)  Azimuth={sonars['forward']['meta']['azimuth']} deg "
          f"(paper w_FOV=90 -> direct match)", flush=True)

    spawn = cfg["agent_spawn"]
    base = np.array(spawn["location"], dtype=float)
    look_down = [0.0, -90.0, 0.0]   # pitch -90: SonarSocket forward axis -> downward
    n_samples = int(cfg["n_samples"])
    settle = int(cfg["settle_ticks"])
    stride = int(cfg["sample_stride_ticks"])
    ticks_per_sec = int(cfg.get("ticks_per_sec", 30))

    distances = list(cfg["distances_m"])
    probe_d = cfg.get("probe_distance_m")
    test_points = [(d, False) for d in distances]
    if probe_d is not None:
        test_points.append((float(probe_d), True))

    # Map a requested ground-truth distance d to an agent altitude: the seafloor sits
    # ~D0 metres below the spawn (measured by the laser at dz=0). We RAISE the agent so
    # the laser ground-truth ~= d. The laser reading is the authoritative ground truth
    # regardless; the altitude is just to span a range of distances.
    # We sweep altitude so laser_gt brackets the requested distances.
    rows: list[dict] = []
    for d, is_probe in test_points:
        tag = "PROBE " if is_probe else ""
        # raise agent by (d - small_base) so it looks down at ~d to the seafloor
        dz = max(0.0, d - 2.0)
        loc = [base[0], base[1], base[2] + dz]
        print(f"--- {tag}target distance d={d} m (agent z={loc[2]:.1f}, look-down) ---",
              flush=True)
        for which, info in sonars.items():
            name = info["sensor_name"]
            meta = info["meta"]
            scenario = _build_scenario(cfg, torpedo, meta, name, loc, look_down)
            per_sample = max(stride, max(1, round(ticks_per_sec / meta["hz"])))
            ranges, lasers = [], []
            with holoocean.make(scenario_cfg=scenario, show_viewport=False) as env:
                env.tick(settle)  # build octree at this pose
                for _ in range(n_samples):
                    state = env.tick(per_sample)
                    img = state.get(name) if isinstance(state, dict) else None
                    ranges.append(_peak_range(img, meta["range_min"], meta["range_max"],
                                              meta["range_bins"]))
                    las = state.get("Laser") if isinstance(state, dict) else None
                    lv = float(np.asarray(las).ravel()[0]) if las is not None else float("nan")
                    lasers.append(lv if lv > 0 else float("nan"))
            arr = np.array(ranges, dtype=float)
            las_arr = np.array(lasers, dtype=float)
            valid = arr[np.isfinite(arr)]
            las_valid = las_arr[np.isfinite(las_arr)]
            n_valid = int(valid.size)
            mean_r = float(np.mean(valid)) if n_valid else float("nan")
            std_r = float(np.std(valid)) if n_valid else float("nan")
            gt = float(np.mean(las_valid)) if las_valid.size else float(d)
            gt_src = "laser" if las_valid.size else "commanded"
            nominal_sigma = SIGMA_ETA[which](gt)
            print(f"  {which:8s} ({name}): sonar n_valid={n_valid}/{n_samples} "
                  f"mean_range={mean_r:.3f} std={std_r:.4f} | laser_gt={gt:.3f} ({gt_src}) "
                  f"| sigma_eta_nom={nominal_sigma:.4f}", flush=True)
            for i in range(n_samples):
                r = float(ranges[i])
                rows.append({
                    "engine": "ho",
                    "stage": "stage_02_smoke",
                    "sonar": which,
                    "sensor_name": name,
                    "sensor_type": meta["sensor_type"],
                    "distance_req_m": float(d),
                    "agent_z": float(loc[2]),
                    "is_probe": bool(is_probe),
                    "sample_idx": i,
                    "range_est_m": r,
                    "valid": bool(np.isfinite(r)),
                    "laser_gt_m": float(lasers[i]),
                    "gt_range_m": gt,
                    "gt_source": gt_src,
                    "sigma_eta_nominal_m": float(nominal_sigma),
                    "range_min_m": float(meta["range_min"]),
                    "range_max_m": float(meta["range_max"]),
                    "range_bins": int(meta["range_bins"]),
                    "azimuth_deg": float(meta["azimuth"]) if meta["azimuth"] is not None else float("nan"),
                    "ho_version": holoocean.__version__,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_ok = _write_parquet(out_dir / "static_smoke.parquet", rows)
    _write_csv(out_dir / "static_smoke.csv", rows)
    print(f"CSV write OK: {out_dir / 'static_smoke.csv'} ({len(rows)} rows)", flush=True)
    print(f"Parquet backend available: {parquet_ok}", flush=True)

    _inline_verdict(rows, cfg)
    print("[DONE]", flush=True)
    return 0


def _inline_verdict(rows: list[dict], cfg: dict) -> None:
    """Inline range-fidelity verdict at the gate distances."""
    acc = cfg.get("acceptance", {})
    err_max = acc.get("mean_range_err_frac_max", 0.50)
    sig_factor = acc.get("sigma_factor_max", 2.0)
    print("=== inline range-fidelity verdict (gate distances) ===", flush=True)
    by = {}
    for r in rows:
        if r["is_probe"]:
            continue
        by.setdefault((r["sonar"], r["distance_req_m"]), []).append(r)
    sonar_usable = True
    for (sonar, d), grp in sorted(by.items()):
        ranges = np.array([g["range_est_m"] for g in grp], dtype=float)
        valid = ranges[np.isfinite(ranges)]
        gt = grp[0]["gt_range_m"]
        if valid.size == 0:
            print(f"  [NO-RETURN] {sonar:8s} d={d:5.1f}: 0 sonar returns "
                  f"(laser_gt={gt:.2f}m present)", flush=True)
            sonar_usable = False
            continue
        mean_r, std_r = float(np.mean(valid)), float(np.std(valid))
        err_frac = abs(mean_r - gt) / gt if gt > 0 else float("inf")
        nominal = grp[0]["sigma_eta_nominal_m"]
        sig_ratio = (std_r / nominal) if nominal > 0 else float("inf")
        mean_ok = err_frac < err_max
        sig_ok = (1.0 / sig_factor) <= sig_ratio <= sig_factor if std_r > 0 else False
        print(f"  {sonar:8s} d={d:5.1f}: sonar_mean={mean_r:.2f} vs laser_gt={gt:.2f} "
              f"err={err_frac*100:5.1f}% [{'OK' if mean_ok else 'FAIL'}] std={std_r:.4f} "
              f"sig_ratio={sig_ratio:.2f}x [{'within' if sig_ok else 'OUTSIDE'} {sig_factor}x]",
              flush=True)
    if not sonar_usable:
        print("=== finding: HoloOcean sonar gave NO returns at some/all distances. "
              "Laser RangeFinder is the viable range channel. ===",
              flush=True)
    else:
        print("=== inline: sonar returned at all gate distances (see ratios above) ===",
              flush=True)


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
    nrows = pq.read_table(path).num_rows  # round-trip integrity check
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
    parser = argparse.ArgumentParser(description="HoloOcean Stage-02 static-target sonar smoke")
    parser.add_argument("--config", default="sim_holoocean/configs/stage_02_smoke.yaml")
    parser.add_argument("--out-dir", default="sim_holoocean/results/stage_02_smoke")
    args = parser.parse_args(argv)
    try:
        return run_smoke(Path(args.config), Path(args.out_dir))
    except Exception as exc:  # noqa: BLE001 — surface one clear [ERROR] line
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
