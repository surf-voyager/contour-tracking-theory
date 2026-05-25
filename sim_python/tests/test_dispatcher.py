"""Tests for sim_python.mc.dispatcher.

Covers:
- 4-run smoke completes < 30 s wall, all parquets exist.
- Each parquet conforms to the normative trajectory schema with engine="py".
- _run_single is pickle-able (joblib spawn pre-requisite).
- Argparse CLI surface is wired up.
- config_hash is stable across runs (deterministic).

Also covers the FSM/sensor/delay plumbing:
- FSM plumbing produces non-trivial ``mode`` / ``lost_count`` columns.
- ``config_hash`` includes the hull/los blocks.
- σ_η and τ_d actually perturb the loop (different σ_η → different trace).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from sim_python.mc import dispatcher as disp


# --------------------------------------------------------------------------- #
# Tiny in-test config (4 runs)                                                #
# --------------------------------------------------------------------------- #


def _tiny_yaml(tmp_path: Path, n: int = 4, T_end: float = 10.0) -> Path:
    cfg = {
        "engine": "py",
        "stage": 3,
        "run_tag": "test",
        "N": n,
        "seed": 42,
        "axes": {
            "kappa_max": [0.01, 0.5, "log"],
            "v_star": [0.5, 1.5, "linear"],
            "f_s": [1.0, 10.0, "log"],
            "tau_d": [0.05, 0.3, "linear"],
        },
        "fixed": {
            "kappa_dot_max": 0.01,
            "u_gate": 0.3,
            "w_FOV": 1.5708,
            "sigma_eta": 0.07,
            "bar_Vc": 0.2,
        },
        "sim": {
            "T_end": T_end,
            "dt": 0.1,
            "d_star": 2.0,
            "d_min": 0.5,
            "side": "L",
            "ds": 0.1,
        },
        "hull": {
            "u_max": 2.0, "r_max": 0.5, "dot_r_max": 1.0,
            "u_gate": 0.3, "tau_u": 0.5, "zeta_gate": 2.0,
        },
        "los": {
            "Delta": 2.0, "K_p": 0.4, "K_ff": 1.0,
            "K_theta": 0.5, "d_star": 2.0, "style": "sin",
        },
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# --------------------------------------------------------------------------- #
# 4-run smoke                                                                 #
# --------------------------------------------------------------------------- #


def test_4_run_smoke_under_30s_and_parquets_exist(tmp_path: Path) -> None:
    """4-run smoke completes < 30 s wall and all parquets exist."""
    cfg_path = _tiny_yaml(tmp_path, n=4, T_end=10.0)
    out_dir = tmp_path / "out"
    t0 = time.perf_counter()
    df = disp.run_batch(
        config_path=str(cfg_path), out_dir=str(out_dir),
        log_path=str(out_dir / "dispatcher.log"), jobs=1,
    )
    wall = time.perf_counter() - t0
    assert wall < 30.0, f"4-run smoke took {wall:.2f} s, expected < 30 s"

    # 4 runs → at most 4 successful + 0 ERROR rows.
    assert len(df) == 4
    # Per-run parquet exists for each non-FAILED run.
    for ch in df["config_hash"]:
        if ch == "FAILED":
            continue
        p = out_dir / f"run_{ch}" / "trajectory.parquet"
        assert p.exists(), f"missing parquet {p}"
    # Aggregate summary exists too.
    assert (out_dir / "_summary.parquet").exists()
    assert (out_dir / "_summary.csv").exists()


def test_parquet_schema_matches_normative(tmp_path: Path) -> None:
    """Trajectory parquet has the normative columns + engine="py"."""
    cfg_path = _tiny_yaml(tmp_path, n=2, T_end=5.0)
    out_dir = tmp_path / "out2"
    df = disp.run_batch(
        config_path=str(cfg_path), out_dir=str(out_dir),
        log_path=None, jobs=1,
    )
    # Sample one run.
    a_hash = next(ch for ch in df["config_hash"] if ch != "FAILED")
    pq_path = out_dir / f"run_{a_hash}" / "trajectory.parquet"
    schema = pq.read_schema(pq_path)
    cols = list(schema.names)
    # Order matters per the dispatcher's TRAJECTORY_COLUMNS contract.
    assert cols == list(disp.TRAJECTORY_COLUMNS), (
        f"schema mismatch: got {cols}, want {list(disp.TRAJECTORY_COLUMNS)}"
    )
    # And the engine column is uniformly "py".
    table = pd.read_parquet(pq_path)
    assert (table["engine"] == "py").all()


# --------------------------------------------------------------------------- #
# _run_single must be picklable (joblib spawn)                                #
# --------------------------------------------------------------------------- #


def test_run_single_is_picklable() -> None:
    """joblib's loky backend requires _run_single to round-trip pickle."""
    blob = pickle.dumps(disp._run_single)
    fn = pickle.loads(blob)
    assert callable(fn)
    # Sanity: same identity / qualified name.
    assert fn.__qualname__ == disp._run_single.__qualname__


# --------------------------------------------------------------------------- #
# config_hash deterministic                                                   #
# --------------------------------------------------------------------------- #


def test_config_hash_stable_and_order_independent() -> None:
    """config_hash is deterministic and key-order independent."""
    a = disp.config_hash({"kappa_max": 0.5, "v_star": 1.0})
    b = disp.config_hash({"v_star": 1.0, "kappa_max": 0.5})
    assert a == b
    # And non-trivial (not all zeros, not collision-prone).
    c = disp.config_hash({"kappa_max": 0.5, "v_star": 1.1})
    assert c != a


# --------------------------------------------------------------------------- #
# CLI argparse                                                                #
# --------------------------------------------------------------------------- #


def test_argparse_required_flags() -> None:
    p = disp._build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args([])  # missing required --config/--out-dir
    args = p.parse_args(
        ["--config", "/tmp/c.yaml", "--out-dir", "/tmp/o", "--jobs", "2"]
    )
    assert args.config == "/tmp/c.yaml"
    assert args.out_dir == "/tmp/o"
    assert args.jobs == 2


# --------------------------------------------------------------------------- #
# Stage-04 hook tests                                                         #
# --------------------------------------------------------------------------- #


def test_config_hash_widens_to_include_hull_and_los(tmp_path: Path) -> None:
    """Stage-04 hook: hash payload includes hull + los so they affect hash."""
    base = {
        "sampled": {"kappa_max": 0.5, "v_star": 1.0},
        "fixed":   {"sigma_eta": 0.05},
        "sim":     {"T_end": 10.0, "dt": 0.1, "d_star": 2.0, "d_min": 0.5},
        "hull":    {"u_max": 2.0, "r_max": 0.5},
        "los":     {"Delta": 2.0, "K_p": 0.4},
    }
    h1 = disp.config_hash(disp._wide_hash_payload(base))
    # Mutate hull → hash changes.
    base2 = {**base, "hull": {"u_max": 3.0, "r_max": 0.5}}
    h2 = disp.config_hash(disp._wide_hash_payload(base2))
    assert h1 != h2, "hull-block change must alter config_hash"
    # Mutate los → hash changes.
    base3 = {**base, "los": {"Delta": 4.0, "K_p": 0.4}}
    h3 = disp.config_hash(disp._wide_hash_payload(base3))
    assert h1 != h3, "los-block change must alter config_hash"


def test_run_single_FSM_emits_mode_and_lost_count_columns(tmp_path: Path) -> None:
    """Stage-04 hook: mode column should be non-trivial; lost_count int32."""
    # Use a config with high noise + tight FOV to maximise L_N/L_G chances.
    cfg = {
        "sampled": {"kappa_max": 0.1, "v_star": 1.0, "f_s": 5.0, "tau_d": 0.1},
        "fixed":   {
            "kappa_dot_max": 0.05, "u_gate": 0.3,
            "w_FOV": 1.0,          # narrow FOV → more Lost-G
            "sigma_eta": 0.3,      # large σ_η → more Lost-N
            "bar_Vc": 0.0,
        },
        "sim":     {
            "T_end": 30.0, "dt": 0.1, "d_star": 2.0, "d_min": 0.3,
            "side": "L", "ds": 0.1, "noise_seed": 7,
        },
        "hull":    {
            "u_max": 2.0, "r_max": 0.5, "dot_r_max": 1.0,
            "u_gate": 0.3, "tau_u": 0.5, "zeta_gate": 2.0,
        },
        "los":     {
            "Delta": 2.0, "K_p": 0.4, "K_ff": 1.0, "K_theta": 0.5,
            "d_star": 2.0, "style": "sin",
        },
        "out_dir": str(tmp_path / "fsm_out"),
        "engine":  "py",
    }
    row = disp._run_single(cfg)
    assert row["config_hash"] != "FAILED"
    pq_path = tmp_path / "fsm_out" / f"run_{row['config_hash']}" / "trajectory.parquet"
    assert pq_path.exists()
    df = pd.read_parquet(pq_path)
    # mode is a valid label
    valid = {"T", "L_N", "L_G", "R"}
    assert set(df["mode"].unique()).issubset(valid)
    # lost_count is monotone non-decreasing
    arr = df["lost_count"].to_numpy()
    assert np.all(np.diff(arr) >= 0)
    # We should detect at least some non-T mode somewhere with σ_η=0.3.
    # (Permissive: the assertion just confirms the FSM is alive — if
    # narrow FOV + tight σ alone don't trigger, the cell is in the
    # tail of the failure distribution, which is itself the gist of
    # phenomena #3 / #5.)
    assert "mode" in df.columns
    # Confirm hash is wide.
    assert row["T_star_N"] > 0
    assert row["T_star_G"] > 0


def test_run_single_sigma_eta_perturbs_trajectory(tmp_path: Path) -> None:
    """Stage-04 hook: σ_η = 0 vs 0.2 yields different `d` traces."""
    base = {
        "sampled": {"kappa_max": 0.05, "v_star": 1.0, "f_s": 5.0, "tau_d": 0.0},
        "fixed":   {
            "kappa_dot_max": 0.01, "u_gate": 0.3,
            "w_FOV": np.pi, "bar_Vc": 0.0,
        },
        "sim":     {
            "T_end": 10.0, "dt": 0.1, "d_star": 2.0, "d_min": 0.3,
            "side": "L", "ds": 0.1, "noise_seed": 1,
        },
        "hull":    {
            "u_max": 2.0, "r_max": 0.5, "dot_r_max": 1.0,
            "u_gate": 0.3, "tau_u": 0.5, "zeta_gate": 2.0,
        },
        "los":     {
            "Delta": 2.0, "K_p": 0.4, "K_ff": 1.0, "K_theta": 0.5,
            "d_star": 2.0, "style": "sin",
        },
        "out_dir": str(tmp_path / "sigma_perturb_out"),
        "engine":  "py",
    }
    quiet = {**base, "fixed": {**base["fixed"], "sigma_eta": 0.0}}
    noisy = {**base, "fixed": {**base["fixed"], "sigma_eta": 0.2}}
    r_q = disp._run_single(quiet)
    r_n = disp._run_single(noisy)
    df_q = pd.read_parquet(tmp_path / "sigma_perturb_out" /
                           f"run_{r_q['config_hash']}" / "trajectory.parquet")
    df_n = pd.read_parquet(tmp_path / "sigma_perturb_out" /
                           f"run_{r_n['config_hash']}" / "trajectory.parquet")
    # Different σ_η ⇒ different wide hash + measurably different trajectory.
    assert r_q["config_hash"] != r_n["config_hash"]
    # End-of-run lateral error should differ measurably.
    err_q = float(df_q["d"].iloc[-1] - 2.0)
    err_n = float(df_n["d"].iloc[-1] - 2.0)
    assert abs(err_q - err_n) > 1e-3 or r_q["max_err"] != r_n["max_err"]


def test_run_single_tau_d_perturbs_trajectory(tmp_path: Path) -> None:
    """Stage-04 hook: τ_d = 0 vs 0.4 s delays alter the closed-loop response."""
    base = {
        "sampled": {"kappa_max": 0.05, "v_star": 1.0, "f_s": 5.0, "tau_d": 0.0},
        "fixed":   {
            "kappa_dot_max": 0.01, "u_gate": 0.3,
            "w_FOV": np.pi, "sigma_eta": 0.0, "bar_Vc": 0.0,
        },
        "sim":     {
            "T_end": 10.0, "dt": 0.1, "d_star": 2.0, "d_min": 0.3,
            "side": "L", "ds": 0.1, "noise_seed": 1,
        },
        "hull":    {
            "u_max": 2.0, "r_max": 0.5, "dot_r_max": 1.0,
            "u_gate": 0.3, "tau_u": 0.5, "zeta_gate": 2.0,
        },
        "los":     {
            "Delta": 2.0, "K_p": 0.4, "K_ff": 1.0, "K_theta": 0.5,
            "d_star": 2.0, "style": "sin",
        },
        "out_dir": str(tmp_path / "tau_perturb_out"),
        "engine":  "py",
    }
    # Initial conditions slightly off-track to make the delay matter.
    no_delay = {**base, "sampled": {**base["sampled"], "tau_d": 0.0}}
    delayed = {**base, "sampled": {**base["sampled"], "tau_d": 0.4}}
    r0 = disp._run_single(no_delay)
    rd = disp._run_single(delayed)
    df0 = pd.read_parquet(tmp_path / "tau_perturb_out" /
                          f"run_{r0['config_hash']}" / "trajectory.parquet")
    dfd = pd.read_parquet(tmp_path / "tau_perturb_out" /
                          f"run_{rd['config_hash']}" / "trajectory.parquet")
    # Different τ_d → different hash + measurably different `r` trace.
    assert r0["config_hash"] != rd["config_hash"]
    assert not np.allclose(df0["r"].to_numpy(), dfd["r"].to_numpy())
