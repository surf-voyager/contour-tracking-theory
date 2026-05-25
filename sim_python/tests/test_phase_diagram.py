"""Tests for sim_python.mc.analysis.phase_diagram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_python.mc.analysis.phase_diagram import plot_2d_slice


def _toy_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "kappa_max": np.exp(rng.uniform(np.log(0.01), np.log(1.0), size=n)),
            "v_star": rng.uniform(0.2, 2.0, size=n),
            "trackable": rng.uniform(0, 1, size=n) < 0.7,
            "mean_err": rng.exponential(scale=0.5, size=n),
        }
    )


# --------------------------------------------------------------------------- #
# PNG size check                                                              #
# --------------------------------------------------------------------------- #


def test_phase_diagram_saves_valid_png_50kb(tmp_path: Path) -> None:
    """Hexbin phase-diagram PNG must be ≥ 50 KB (i.e. actually rendered)."""
    df = _toy_df(n=200, seed=1)
    out = tmp_path / "fig.png"
    ax = plot_2d_slice(
        df,
        x_axis="kappa_max",
        y_axis="v_star",
        metric="trackable",
        mode="hexbin",
        gridsize=15,
        xscale="log",
        title="trackable rate (κ_max, v*)",
        save_path=str(out),
        dpi=150,
    )
    assert ax is not None
    assert out.exists()
    assert out.stat().st_size >= 50_000, (
        f"PNG only {out.stat().st_size} bytes, < 50 KB"
    )


def test_scatter_mode_runs_and_saves(tmp_path: Path) -> None:
    df = _toy_df(n=50, seed=2)
    out = tmp_path / "scatter.png"
    plot_2d_slice(
        df, x_axis="kappa_max", y_axis="v_star", metric="mean_err",
        mode="scatter", xscale="log", save_path=str(out),
    )
    assert out.exists()


def test_contour_mode_with_enough_points(tmp_path: Path) -> None:
    df = _toy_df(n=80, seed=3)
    out = tmp_path / "contour.png"
    plot_2d_slice(
        df, x_axis="kappa_max", y_axis="v_star", metric="mean_err",
        mode="contour", xscale="log", save_path=str(out),
    )
    assert out.exists()


def test_rejects_unknown_mode() -> None:
    df = _toy_df(n=20, seed=4)
    with pytest.raises(ValueError):
        plot_2d_slice(df, x_axis="kappa_max", y_axis="v_star",
                      metric="mean_err", mode="bogus")


def test_rejects_missing_column() -> None:
    df = _toy_df(n=20, seed=5)
    with pytest.raises(KeyError):
        plot_2d_slice(df, x_axis="kappa_max", y_axis="v_star",
                      metric="nonexistent")
