"""Tests for sim_python.controllers.curvature_estimator.

Covers the discrete-sensing closed form σ_κ̂² = 720 σ_η² / (L⁵ s̄⁴),
where the constant 720 is exact, and the sliding-window LS arc fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.controllers.curvature_estimator import (
    CurvatureEstimator,
    _VARIANCE_CONSTANT_720,
)


# --------------------------------------------------------------------------- #
# Test 1 — σ_κ̂² numerical match to the analytical 720 closed form            #
# --------------------------------------------------------------------------- #


def test_sigma_kappa_hat_matches_analytical_720_within_1_percent() -> None:
    """For L=20, s̄=0.5, σ_η=0.05 the computed σ_κ̂ must match the
    closed-form sqrt(720 · 0.05² / (20⁵ · 0.5⁴)) within 1% relative.
    """
    L = 20
    s_bar = 0.5
    sigma_eta = 0.05
    est = CurvatureEstimator(window_L=L, s_bar=s_bar)
    sk_computed = est.sigma_kappa_hat(sigma_eta=sigma_eta)
    sk_analytical = np.sqrt(720.0 * sigma_eta ** 2 / (L ** 5 * s_bar ** 4))
    # 1% relative tolerance; actual error is float-equal
    assert abs(sk_computed - sk_analytical) / sk_analytical < 0.01, (
        f"σ_κ̂ computed = {sk_computed:.6e} vs analytical = {sk_analytical:.6e}"
    )
    # Sanity check: the analytical number equals 3.0e-3.
    assert abs(sk_analytical - 3.0e-3) < 1e-9


# --------------------------------------------------------------------------- #
# Test 2 — hard-coded 720 constant is exactly 720                             #
# --------------------------------------------------------------------------- #


def test_variance_constant_is_exactly_720() -> None:
    """Guard against an accidental change to the σ_κ̂² constant.

    Replacing 720 with any other value must cause this test (and the
    T*_N reference test) to fail.
    """
    assert _VARIANCE_CONSTANT_720 == 720.0, (
        f"σ_κ̂² constant must be exactly 720; "
        f"got {_VARIANCE_CONSTANT_720}"
    )
    est = CurvatureEstimator(window_L=10, s_bar=1.0)
    assert est.variance_constant == 720.0


# --------------------------------------------------------------------------- #
# Test 3 — noiseless arc recovers κ_true exactly                              #
# --------------------------------------------------------------------------- #


def test_noiseless_arc_recovers_true_kappa() -> None:
    """y(s) = (κ/2) s² → LS fit returns κ̂ = κ exactly (small-angle)."""
    kappa_true = 0.05
    L = 20
    s_bar = 0.5
    est = CurvatureEstimator(window_L=L, s_bar=s_bar)
    s = np.arange(L, dtype=np.float64) * s_bar
    y = 0.5 * kappa_true * s * s
    last = 0.0
    for yi in y:
        last = est.update(yi)
    assert abs(last - kappa_true) < 1e-9, f"κ̂ = {last}, expected {kappa_true}"


# --------------------------------------------------------------------------- #
# Test 4 — Monte Carlo: empirical std matches σ_κ̂ closed form within 10%     #
# --------------------------------------------------------------------------- #


def test_monte_carlo_empirical_sigma_matches_closed_form() -> None:
    """Sanity: estimator's empirical std should track the 720-formula
    within ~10% for N=500 trials of noisy arcs.

    This is the strongest end-to-end check that the LS implementation
    is consistent with the closed form.  Uses a seeded
    numpy.random.Generator for reproducibility.
    """
    L = 20
    s_bar = 0.5
    sigma_eta = 0.05
    kappa_true = 0.02
    rng = np.random.default_rng(seed=20260523)
    n_trials = 500
    s_grid = np.arange(L, dtype=np.float64) * s_bar
    y_clean = 0.5 * kappa_true * s_grid * s_grid
    estimates = np.empty(n_trials)
    for t in range(n_trials):
        est = CurvatureEstimator(window_L=L, s_bar=s_bar)
        noise = rng.normal(0.0, sigma_eta, size=L)
        y_noisy = y_clean + noise
        last = 0.0
        for yi in y_noisy:
            last = est.update(yi)
        estimates[t] = last
    emp_std = float(np.std(estimates, ddof=1))
    closed = np.sqrt(720.0 * sigma_eta ** 2 / (L ** 5 * s_bar ** 4))
    # 10% tolerance for n_trials=500 (std-of-std is O(1/sqrt(2n)) ≈ 3%)
    assert abs(emp_std - closed) / closed < 0.10, (
        f"MC σ_κ̂ = {emp_std:.6e}, closed-form = {closed:.6e}, "
        f"rel err = {abs(emp_std - closed) / closed:.2%}"
    )
    # Empirical mean should also recover κ_true (LS unbiasedness)
    emp_mean = float(np.mean(estimates))
    assert abs(emp_mean - kappa_true) < 3 * closed / np.sqrt(n_trials)


# --------------------------------------------------------------------------- #
# Test 5 — warm-up + reset behaviour                                          #
# --------------------------------------------------------------------------- #


def test_warmup_and_reset() -> None:
    L = 8
    est = CurvatureEstimator(window_L=L, s_bar=0.5)
    # Before warmup_L samples: returns 0 and reports not warmed up.
    for k in range(L - 1):
        out = est.update(0.01 * k)
        assert out == 0.0, f"warmup should return 0; got {out} at k={k}"
        assert not est.is_warmed_up
    # Last sample warms up.
    est.update(0.01 * (L - 1))
    assert est.is_warmed_up
    assert est.last_fit_coefficients is not None
    # Reset returns to warmup state.
    est.reset()
    assert not est.is_warmed_up
    assert est.last_kappa == 0.0
    assert est.last_fit_coefficients is None


# --------------------------------------------------------------------------- #
# Test 6 — L < 5 raises ValueError (engineering lower bound)                  #
# --------------------------------------------------------------------------- #


def test_min_window_L_enforced() -> None:
    """L ≥ 5 (design-matrix condition-number requirement)."""
    with pytest.raises(ValueError, match="window_L"):
        CurvatureEstimator(window_L=4, s_bar=0.5)
    # L = 5 is the boundary; should be accepted.
    CurvatureEstimator(window_L=5, s_bar=0.5)


# --------------------------------------------------------------------------- #
# Test 7 — input validation                                                    #
# --------------------------------------------------------------------------- #


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="s_bar"):
        CurvatureEstimator(window_L=10, s_bar=0.0)
    with pytest.raises(ValueError, match="s_bar"):
        CurvatureEstimator(window_L=10, s_bar=-1.0)
    est = CurvatureEstimator(window_L=10, s_bar=0.5)
    with pytest.raises(ValueError, match="sigma_eta"):
        est.sigma_kappa_hat(sigma_eta=-1.0)
    with pytest.raises(ValueError, match="p_meas"):
        est.update(float("nan"))
