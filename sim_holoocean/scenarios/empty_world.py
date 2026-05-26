"""Minimal HoloOcean 2.3.0 hello-world smoke test.

Verifies the HoloOcean install end-to-end:
  1. import holoocean + print version
  2. detect + print GPU (nvidia-smi)
  3. make a built-in Ocean-package scenario HEADLESS (show_viewport=False)
  4. tick ~100 frames, reading the PoseSensor each tick
  5. write a tiny parquet (tick index + a sensor reading) via pyarrow
  6. print status sentinels:
       "HoloOcean version: X"
       "GPU: <name>"
       "Stepped 100 ticks; FPS=<x>"
       "[DONE]"

Run from the repository root on a machine with the HoloOcean engine installed:
    PYTHONPATH=. python -m sim_holoocean.scenarios.empty_world

API NOTE:
    holoocean.make(scenario_name=..., show_viewport=False) -> HoloOceanEnvironment
    env.tick(num_ticks=1) -> dict keyed by sensor name (e.g. "PoseSensor")
    Used as a context manager.

This module is layer-isolated: it imports nothing from sim_python.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# --- Built-in Ocean-package scenario for the smoke (ships with the Ocean package) ---
# SimpleUnderwater-Hovering is the simplest fully-defined Ocean scenario:
# one HoveringAUV (auv0) with PoseSensor/DepthSensor/etc, no sonar plugin needed.
DEFAULT_SCENARIO = "SimpleUnderwater-Hovering"
DEFAULT_TICKS = 100


def detect_gpu() -> str:
    """Return GPU model string via nvidia-smi; fall back to torch / 'unknown'."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        name = out.stdout.strip().splitlines()
        if out.returncode == 0 and name:
            return name[0].strip()
    except Exception as exc:  # noqa: BLE001 — diagnostic fallback only
        print(f"  (nvidia-smi probe failed: {exc})", flush=True)
    # Fallback: torch may not be installed in the holoocean env — that's fine.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def run_smoke(scenario: str, ticks: int, out_dir: Path) -> int:
    import holoocean  # imported here so version line prints even on later failure

    # --- Sentinel 1: HoloOcean version ---
    print(f"HoloOcean version: {holoocean.__version__}", flush=True)
    print(f"Installed packages: {holoocean.installed_packages()}", flush=True)

    # --- Sentinel 2: GPU ---
    gpu = detect_gpu()
    print(f"GPU: {gpu}", flush=True)

    print(f"Making scenario (headless): {scenario}", flush=True)

    # --- Make HEADLESS: show_viewport=False is the headless flag in HO 2.x ---
    pose_samples: list[float] = []
    t0 = time.time()
    with holoocean.make(scenario_name=scenario, show_viewport=False) as env:
        for i in range(ticks):
            state = env.tick(num_ticks=1)
            # state is a dict keyed by sensor name. Grab a scalar to prove the
            # sensor pipeline is live; PoseSensor is a 4x4 transform matrix.
            z = float("nan")
            if isinstance(state, dict):
                pose = state.get("PoseSensor")
                if pose is not None:
                    try:
                        # 4x4 homogeneous transform: row 2, col 3 is the z (depth) translation
                        z = float(pose[2][3])
                    except Exception:  # noqa: BLE001
                        z = float("nan")
            pose_samples.append(z)
    elapsed = time.time() - t0
    fps = ticks / elapsed if elapsed > 0 else float("inf")

    # --- Sentinel 3: stepped N ticks + FPS ---
    print(f"Stepped {ticks} ticks; FPS={fps:.2f}", flush=True)

    # --- Parquet write proof (pyarrow round-trip) ---
    # If pyarrow is missing, we DO NOT fail the smoke: a CSV fallback is always
    # written so the sensor pipeline still produces a usable artifact, and a clear
    # [WARN] flags the missing dependency (pip install pyarrow). HoloOcean
    # viability is independent of the parquet backend.
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_ok = _write_parquet(
        out_dir / "smoke.parquet", scenario, gpu, holoocean.__version__, pose_samples
    )
    # CSV fallback is always written (no third-party backend needed).
    _write_csv(out_dir / "smoke.csv", scenario, gpu, holoocean.__version__, pose_samples)
    print(f"CSV write OK: {out_dir / 'smoke.csv'} ({len(pose_samples)} rows)", flush=True)
    print(f"Parquet backend available: {parquet_ok}", flush=True)

    # --- Final sentinel ---
    print("[DONE]", flush=True)
    return 0


def _rows(scenario: str, gpu: str, version: str, pose_samples: list[float]):
    """Yield uniform dict rows: one per tick (tick index + pose-z reading)."""
    for i, z in enumerate(pose_samples):
        yield {
            "tick": i,
            "pose_z": z,
            "engine": "ho",
            "scenario": scenario,
            "gpu": gpu,
            "ho_version": version,
        }


def _write_parquet(path: Path, scenario: str, gpu: str, version: str,
                   pose_samples: list[float]) -> bool:
    """Write a tiny parquet via pyarrow; return True on success.

    On ImportError (pyarrow not installed) print a [WARN]
    and return False rather than aborting the whole smoke.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(
            f"[WARN] parquet backend unavailable ({exc}); skipping parquet, "
            "writing CSV only. Run `pip install pyarrow` in the 'holoocean' env "
            "to enable parquet output.",
            flush=True,
        )
        return False

    n = len(pose_samples)
    table = pa.table(
        {
            "tick": pa.array(list(range(n)), type=pa.int32()),
            "pose_z": pa.array(pose_samples, type=pa.float64()),
            "engine": pa.array(["ho"] * n, type=pa.string()),
            "scenario": pa.array([scenario] * n, type=pa.string()),
            "gpu": pa.array([gpu] * n, type=pa.string()),
            "ho_version": pa.array([version] * n, type=pa.string()),
        }
    )
    pq.write_table(table, path)
    nrows = pq.read_table(path).num_rows  # round-trip integrity check
    print(f"Parquet write OK: {path} ({nrows} rows)", flush=True)
    return True


def _write_csv(path: Path, scenario: str, gpu: str, version: str,
              pose_samples: list[float]) -> None:
    """Always-available CSV fallback (stdlib csv; no third-party backend)."""
    import csv

    fieldnames = ["tick", "pose_z", "engine", "scenario", "gpu", "ho_version"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in _rows(scenario, gpu, version, pose_samples):
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HoloOcean hello-world smoke test")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                       help=f"Ocean-package scenario name (default: {DEFAULT_SCENARIO})")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                       help=f"number of ticks to step (default: {DEFAULT_TICKS})")
    parser.add_argument(
        "--out-dir",
        default="sim_holoocean/results/stage_01_smoke",
        help="output dir for the tiny smoke parquet",
    )
    args = parser.parse_args(argv)

    try:
        return run_smoke(args.scenario, args.ticks, Path(args.out_dir))
    except Exception as exc:  # noqa: BLE001 — surface a single clear [ERROR] line
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
