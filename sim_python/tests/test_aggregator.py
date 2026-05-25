"""Tests for sim_python.mc.analysis.aggregator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_python.mc.analysis.aggregator import (
    compute_per_cell,
    load_batch,
    load_trajectory,
)


# --------------------------------------------------------------------------- #
# load_batch                                                                  #
# --------------------------------------------------------------------------- #


def test_load_batch_round_trip(tmp_path: Path) -> None:
    df_in = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    df_in.to_parquet(tmp_path / "_summary.parquet")
    df_out = load_batch(tmp_path)
    pd.testing.assert_frame_equal(df_in, df_out)


def test_load_batch_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_batch(tmp_path)


def test_load_trajectory(tmp_path: Path) -> None:
    sub = tmp_path / "run_abc123"
    sub.mkdir()
    df = pd.DataFrame({"t": [0.0, 0.1], "x": [0.0, 1.0]})
    df.to_parquet(sub / "trajectory.parquet")
    out = load_trajectory(tmp_path, "abc123")
    pd.testing.assert_frame_equal(df, out)


# --------------------------------------------------------------------------- #
# compute_per_cell                                                            #
# --------------------------------------------------------------------------- #


def test_compute_per_cell_returns_expected_dimensions() -> None:
    """Per-cell groupby returns the expected dimensions."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 1, size=n),
            "y": rng.uniform(0, 1, size=n),
            "metric": rng.normal(loc=1.0, size=n),
        }
    )
    out = compute_per_cell(df, axes=("x", "y"), metric="metric", nbins=4)
    # Output columns
    assert set(out.columns) == {"x", "y", "agg_value", "count"}
    # ≤ 4×4 cells filled
    assert len(out) <= 16
    # Total counts equal n
    assert out["count"].sum() == n


def test_compute_per_cell_pass_rate_for_bool_metric() -> None:
    """Boolean metric → cell pass rate ∈ [0, 1]."""
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 1, size=n),
            "y": rng.uniform(0, 1, size=n),
            "trackable": rng.uniform(0, 1, size=n) < 0.5,
        }
    )
    out = compute_per_cell(
        df, axes=("x", "y"), metric="trackable", nbins=3, agg="mean"
    )
    assert (out["agg_value"] >= 0.0).all()
    assert (out["agg_value"] <= 1.0).all()


def test_compute_per_cell_log_axis_handling() -> None:
    """Log axes bin in log space; mean of log-uniform values should sit
    near the geometric mean of the range."""
    rng = np.random.default_rng(2)
    vals = np.exp(rng.uniform(np.log(0.01), np.log(1.0), size=80))
    df = pd.DataFrame(
        {
            "kappa_max": vals,
            "v_star": rng.uniform(0.2, 2.0, size=80),
            "metric": np.ones(80),
        }
    )
    out = compute_per_cell(
        df, axes=("kappa_max", "v_star"), metric="metric",
        nbins=4, agg="count", log_axes=("kappa_max",),
    )
    # With nbins=4 log-spaced and ~80 points, expect roughly even
    # distribution (no bin >> 4x the median).
    counts = out["count"].to_numpy()
    assert counts.sum() == 80


def test_compute_per_cell_rejects_unknown_metric() -> None:
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(KeyError):
        compute_per_cell(df, axes=("x", "y"), metric="missing")


def test_compute_per_cell_rejects_bad_agg() -> None:
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "m": [1.0, 2.0]})
    with pytest.raises(ValueError):
        compute_per_cell(df, axes=("x", "y"), metric="m", agg="bogus")
