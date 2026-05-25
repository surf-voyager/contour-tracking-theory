#!/usr/bin/env python
"""Dual-sonar batch runner — execute one batch with noise-realisation looping.

Given a batch YAML (scanning or forward), this driver:

1. Builds the Latin-Hypercube sample once (from the YAML's N and seed).
2. Loops the noise-realisation index in {0, ..., M_realisations - 1},
   dispatching each (config, seed) pair via ``mc.dispatcher._run_single``.
3. Concatenates all M*N summary rows into ``_summary.parquet`` under the
   requested ``--out-dir``.

It reuses the single-config worker from ``mc.dispatcher`` so that each
per-run parquet remains the canonical record for that configuration.

Usage
-----
    python sim_python/scripts/run_stage_05_dual_sonar.py \\
        --config sim_python/configs/stage_05b_scanning_ctrl_n2000.yaml \\
        --out-dir results/scanning_ctrl \\
        --jobs 12
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from sim_python.mc import dispatcher as disp
from sim_python.mc.sampler import latin_hypercube_sample


def _build_seed_cfgs(
    batch: Dict[str, Any], out_dir: Path, noise_seed: int,
) -> List[Dict[str, Any]]:
    """Expand the YAML into per-run cfgs for a single noise_seed."""
    axes = {name: tuple(spec) for name, spec in batch["axes"].items()}
    N = int(batch["N"])
    seed = int(batch.get("seed", 42))
    sampled_list = latin_hypercube_sample(axes, N=N, seed=seed)

    fixed = dict(batch.get("fixed", {}))
    sim_defaults = dict(batch["sim"])
    sim_defaults["noise_seed"] = int(noise_seed)
    hull_defaults = dict(batch["hull"])
    los_defaults = dict(batch["los"])

    cfgs: List[Dict[str, Any]] = []
    for sampled in sampled_list:
        cfgs.append({
            "sampled": sampled,
            "fixed": fixed,
            "sim": sim_defaults,
            "hull": hull_defaults,
            "los": los_defaults,
            "out_dir": str(out_dir),
            "engine": "py",
        })
    return cfgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Stage-05 YAML (scanning OR forward)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for parquets")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--log", default=None, help="Optional log file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    batch = yaml.safe_load(cfg_path.read_text())
    M = int(batch.get("M_realisations", 1))
    N = int(batch["N"])

    # Echo YAML to output dir for archive provenance.
    (out_path / "config.yaml").write_text(cfg_path.read_text())

    logger = logging.getLogger("stage_05.runner")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    if args.log is not None:
        lp = Path(args.log)
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(lp)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    logger.info(
        f"Stage-05 batch: {cfg_path.name} | N={N} × M={M} = {N*M} runs | "
        f"jobs={args.jobs}"
    )

    rows_all: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for m in range(M):
        cfgs_m = _build_seed_cfgs(batch, out_path, noise_seed=m)
        logger.info(f"M-realisation {m+1}/{M}: noise_seed={m}, {len(cfgs_m)} cfgs")
        if args.jobs == 1:
            rows_m = [disp._run_single(c) for c in tqdm(cfgs_m, desc=f"M={m}")]
        else:
            rows_m = Parallel(n_jobs=args.jobs, backend="loky", verbose=0)(
                delayed(disp._run_single)(c)
                for c in tqdm(cfgs_m, desc=f"M={m}", file=sys.stdout)
            )
        for r in rows_m:
            # Tag the noise-seed so we can disambiguate post-hoc.
            r["noise_seed"] = int(m)
        rows_all.extend(rows_m)
    wall = time.perf_counter() - t_start

    df = pd.DataFrame(rows_all)
    df.to_parquet(out_path / "_summary.parquet")
    df.to_csv(out_path / "_summary.csv", index=False)

    n_ok = int((df["terminate_reason"] == "COMPLETED").sum()) \
        if "terminate_reason" in df else 0
    n_err = int((df["config_hash"] == "FAILED").sum()) \
        if "config_hash" in df else 0
    logger.info(
        f"All M-realisations done: {len(df)} rows in {wall:.1f} s "
        f"({wall / max(len(df), 1):.3f} s/run); {n_ok} COMPLETED, {n_err} ERROR; "
        f"trackable rate = {df['trackable'].mean():.2%}"
    )
    if "lost_g_count" in df:
        lg_rate = float((df["lost_g_count"] > 0).mean())
        logger.info(f"Lost-G frequency (any-G per run) = {lg_rate:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
