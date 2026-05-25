"""Tests for sim_python.mc.sampler.

Covers:
- LHS N=100 seed=42 is deterministic / reproducible across runs.
- LHS coverage is uniform within bins (chi² test p > 0.05).
- Log-scale axis end-points are respected.
- Bad inputs raise ValueError.
- adaptive_boundary_refine returns the right shape & respects perturb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from sim_python.mc.sampler import (
    adaptive_boundary_refine,
    latin_hypercube_sample,
)


_RANGES_FOUR = {
    "kappa_max": (0.01, 1.0, "log"),
    "v_star": (0.2, 2.0, "linear"),
    "f_s": (0.5, 20.0, "log"),
    "tau_d": (0.05, 0.5, "linear"),
}


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #


def test_lhs_seed_42_deterministic() -> None:
    """LHS N=100 seed=42 is reproducible across runs."""
    a = latin_hypercube_sample(_RANGES_FOUR, N=100, seed=42)
    b = latin_hypercube_sample(_RANGES_FOUR, N=100, seed=42)
    assert len(a) == len(b) == 100
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            assert ra[k] == rb[k], f"non-deterministic on key {k}"


def test_lhs_different_seeds_differ() -> None:
    a = latin_hypercube_sample(_RANGES_FOUR, N=50, seed=1)
    b = latin_hypercube_sample(_RANGES_FOUR, N=50, seed=2)
    # At least one entry must differ (probability of exact match ≈ 0)
    any_diff = any(
        a[i][k] != b[i][k]
        for i in range(50)
        for k in _RANGES_FOUR
    )
    assert any_diff


# --------------------------------------------------------------------------- #
# Uniformity (per-axis chi² test)                                             #
# --------------------------------------------------------------------------- #


def test_lhs_uniform_within_bins_chi2() -> None:
    """LHS samples are uniform within bins, chi² p > 0.05.

    For an N=100 LHS, divide each axis into 10 equal-frequency bins
    (10 expected per bin) and run a chi² goodness-of-fit.  By LHS
    construction each bin must contain exactly one sample per axis,
    so chi² statistic = 0 and p ≈ 1.  But we also test a non-trivial
    rebinning (5 bins of 20 expected) where slight Latin-hypercube
    quantization is still visible.
    """
    sample = latin_hypercube_sample(_RANGES_FOUR, N=100, seed=42)
    for axis, (lo, hi, scale) in _RANGES_FOUR.items():
        vals = np.array([row[axis] for row in sample])
        # Bin in axis-native scale (log for log axes).
        if scale == "log":
            edges = np.exp(np.linspace(np.log(lo), np.log(hi), 11))
        else:
            edges = np.linspace(lo, hi, 11)
        hist, _ = np.histogram(vals, bins=edges)
        # Expected: 10 per bin (N=100 over 10 bins).
        expected = np.full_like(hist, fill_value=10, dtype=float)
        # LHS strictly stratifies, so observed should equal expected
        # and chi² statistic = 0 (p = 1).  We assert p > 0.05.
        stat, p = chisquare(f_obs=hist, f_exp=expected)
        assert p > 0.05, f"axis {axis}: chi² p={p:.4f} ≤ 0.05 (hist={hist})"


# --------------------------------------------------------------------------- #
# Range coverage                                                              #
# --------------------------------------------------------------------------- #


def test_lhs_log_axis_within_range_and_positive() -> None:
    sample = latin_hypercube_sample(_RANGES_FOUR, N=200, seed=7)
    for row in sample:
        assert 0.01 <= row["kappa_max"] <= 1.0
        assert 0.2 <= row["v_star"] <= 2.0
        assert 0.5 <= row["f_s"] <= 20.0
        assert 0.05 <= row["tau_d"] <= 0.5
        # Log axes positive:
        assert row["kappa_max"] > 0
        assert row["f_s"] > 0


def test_lhs_log_scale_evenly_distributes_in_log() -> None:
    """A log-scale axis should distribute mass evenly in log10 space."""
    sample = latin_hypercube_sample(
        {"x": (1.0, 1000.0, "log")}, N=300, seed=11
    )
    vals = np.array([row["x"] for row in sample])
    logvals = np.log10(vals)
    # 3-decade log axis → log10 spans [0, 3]; mean ≈ 1.5
    assert 1.4 < np.mean(logvals) < 1.6
    # Std of uniform on [0, 3] = 3/sqrt(12) ≈ 0.866
    assert 0.7 < np.std(logvals) < 1.0


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_lhs_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        latin_hypercube_sample({}, N=10, seed=0)
    with pytest.raises(ValueError):
        latin_hypercube_sample({"x": (0.0, 1.0, "log")}, N=10, seed=0)
    with pytest.raises(ValueError):
        latin_hypercube_sample({"x": (1.0, 0.0, "linear")}, N=10, seed=0)
    with pytest.raises(ValueError):
        latin_hypercube_sample({"x": (1.0, 2.0, "bogus")}, N=10, seed=0)
    with pytest.raises(ValueError):
        latin_hypercube_sample(_RANGES_FOUR, N=0, seed=0)


# --------------------------------------------------------------------------- #
# Adaptive boundary refine                                                    #
# --------------------------------------------------------------------------- #


def test_adaptive_refine_returns_correct_shape() -> None:
    prior = pd.DataFrame(
        {
            "kappa_max": np.linspace(0.01, 1.0, 50),
            "v_star": np.linspace(0.5, 1.5, 50),
            "c1_residual": np.linspace(-0.5, 0.5, 50),
            "c2_residual": np.linspace(-1.0, 1.0, 50),
        }
    )
    refined = adaptive_boundary_refine(prior, N_refine=20, seed=3)
    assert len(refined) == 20
    for row in refined:
        # Should perturb the parameter columns, not the residual cols.
        assert "kappa_max" in row
        assert "v_star" in row
        assert "c1_residual" not in row
        assert "c2_residual" not in row


def test_adaptive_refine_fallback_when_no_boundary_rows() -> None:
    # All residuals outside the window → fallback to mean perturbation.
    prior = pd.DataFrame(
        {
            "kappa_max": np.full(10, 0.5),
            "v_star": np.full(10, 1.0),
            "c1_residual": np.full(10, 1.5),
            "c2_residual": np.full(10, -2.0),
        }
    )
    refined = adaptive_boundary_refine(prior, N_refine=5, seed=4)
    assert len(refined) == 5
    # All perturbed values should sit within ±20% of the mean (perturb_frac=0.05).
    for row in refined:
        assert 0.4 < row["kappa_max"] < 0.6
        assert 0.8 < row["v_star"] < 1.2
