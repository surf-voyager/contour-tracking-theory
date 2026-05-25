"""Tests for sim_python.mc.analysis.dual_sonar_compare.

Verifies the comparison helpers against synthetic data:

- ``compare_metric`` returns a DataFrame with the documented columns;
- bin centres span the joint range;
- scanning batch with all-zero Lost-G correctly differs from a
  forward batch with non-trivial Lost-G;
- error handling for missing axis / metric columns is sane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim_python.mc.analysis.dual_sonar_compare import (
    compare_metric,
    plot_pair_overlay,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two synthetic batches that mimic Stage-05 dual-sonar contrast.

    Scanning batch: 200 rows, κ'_max log-uniform in [1e-3, 1e-1],
    T_star_G_emp_mean ≡ 0 (geometric rigidity).
    Forward batch: 200 rows, same κ'_max range, T_star_G_emp_mean
    ≈ ū⁻¹√(w_FOV/κ'_max) (the T*_G closed form) with ū=1 m/s, w_FOV=π/2.
    """
    rng = np.random.default_rng(0)
    kp = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), size=200))
    w_fov = np.pi / 2
    u_bar = 1.0

    scanning = pd.DataFrame({
        "kappa_dot_max": kp,
        "T_star_G_emp_mean": np.zeros_like(kp),
        "lost_g_count": np.zeros_like(kp, dtype=int),
    })

    # Forward analytical curve + 10 % noise.
    t_analytical = (1.0 / u_bar) * np.sqrt(w_fov / kp)
    forward = pd.DataFrame({
        "kappa_dot_max": kp,
        "T_star_G_emp_mean": t_analytical * (1.0 + rng.normal(0.0, 0.1, size=200)),
        "lost_g_count": rng.integers(0, 5, size=200),
    })
    return scanning, forward


# --------------------------------------------------------------------------- #
# compare_metric                                                              #
# --------------------------------------------------------------------------- #


def test_compare_metric_returns_expected_columns(synthetic_pair) -> None:
    s, f = synthetic_pair
    out = compare_metric(s, f, metric="T_star_G_emp_mean",
                         axis="kappa_dot_max", n_bins=8, axis_log=True)
    expected = {
        "axis_bin_center", "scanning_mean", "scanning_std", "scanning_n",
        "forward_mean", "forward_std", "forward_n", "abs_diff", "rel_diff_pct",
    }
    assert expected.issubset(out.columns), \
        f"missing columns {expected - set(out.columns)}"
    assert len(out) == 8


def test_compare_metric_log_axis_bin_centers(synthetic_pair) -> None:
    s, f = synthetic_pair
    out = compare_metric(s, f, metric="T_star_G_emp_mean",
                         axis="kappa_dot_max", n_bins=6, axis_log=True)
    # In log space, ratios between successive bin centres are constant.
    centres = out["axis_bin_center"].to_numpy()
    ratios = centres[1:] / centres[:-1]
    assert np.allclose(ratios, ratios.mean(), rtol=1e-6), \
        "log-spaced bins should have constant successive ratio"


def test_compare_metric_scanning_zero_forward_nonzero(synthetic_pair) -> None:
    """Scanning batch: T*_G ≡ 0; forward batch: T*_G > 0 everywhere."""
    s, f = synthetic_pair
    out = compare_metric(s, f, metric="T_star_G_emp_mean",
                         axis="kappa_dot_max", n_bins=8, axis_log=True)
    # Drop any bin with zero count on either side.
    keep = (out["scanning_n"] > 0) & (out["forward_n"] > 0)
    sub = out[keep]
    assert (sub["scanning_mean"].abs() < 1e-9).all(), \
        "scanning mean must be 0 everywhere"
    assert (sub["forward_mean"] > 0).all(), \
        "forward mean must be > 0 everywhere"
    # abs_diff is scanning − forward, so all entries are negative.
    assert (sub["abs_diff"] < 0).all()


def test_compare_metric_analytical_overlay_within_30pct(synthetic_pair) -> None:
    """Forward batch empirical T*_G must track analytical within 30 % per bin.

    Checks that the empirical T*_G is within 30 % of the analytical
    ū⁻¹√(w_FOV/κ'_max) across ≥ 5 κ'_max values.
    """
    s, f = synthetic_pair
    out = compare_metric(s, f, metric="T_star_G_emp_mean",
                         axis="kappa_dot_max", n_bins=8, axis_log=True)
    w_fov = np.pi / 2
    u_bar = 1.0
    out = out.assign(
        analytical=(1.0 / u_bar) * np.sqrt(w_fov / out["axis_bin_center"]),
    )
    out = out.dropna(subset=["forward_mean"])
    rel = (out["forward_mean"] - out["analytical"]).abs() / out["analytical"]
    n_within = int((rel < 0.30).sum())
    assert n_within >= 5, \
        f"only {n_within} bins within 30 % of analytical (need ≥ 5)"


def test_compare_metric_missing_column_raises() -> None:
    s = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    f = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(KeyError):
        compare_metric(s, f, metric="nope", axis="a")
    with pytest.raises(KeyError):
        compare_metric(s, f, metric="b", axis="nope")


# --------------------------------------------------------------------------- #
# plot_pair_overlay                                                           #
# --------------------------------------------------------------------------- #


def test_plot_pair_overlay_returns_table_and_handles(synthetic_pair) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, f = synthetic_pair
    fig, ax = plt.subplots()
    res = plot_pair_overlay(
        s, f, axis="kappa_dot_max", metric="T_star_G_emp_mean",
        ax=ax, axis_log=True, n_bins=6,
    )
    assert {"ax", "scanning_line", "forward_line", "table"}.issubset(res.keys())
    assert isinstance(res["table"], pd.DataFrame)
    assert len(res["table"]) == 6
    plt.close(fig)
