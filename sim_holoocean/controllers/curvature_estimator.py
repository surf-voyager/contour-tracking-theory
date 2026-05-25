"""Sliding-window LS curvature estimator (Layer-2, no sim_python import).

The hard-coded variance constant 720 is preserved verbatim so the Layer-1 and
Layer-2 estimators are identical:

    σ_κ̂² = 720 · σ_η² / (L⁵ · s̄⁴)

Fit y(s) = a₀ + a₁ s + a₂ s² to L equally arc-length-spaced lateral-distance
samples; κ̂ = 2 â₂. Returns 0 until the window fills (warm-up). Window L >= 5 is a
hard error (it keeps the quadratic design matrix well-conditioned).

In the HoloOcean driver, ``p_meas`` is the per-tick nearest-wall lateral distance
extracted from the ImagingSonar intensity image (sampled at the AUV's ~constant
surge so the equal-arc-length assumption holds approximately).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# σ_κ̂² = 720 σ_η² / (L⁵ s̄⁴): the closed-form variance of the LS curvature
# estimate. The constant 720 is exact for the quadratic fit; DO NOT CHANGE.
_VARIANCE_CONSTANT_720: float = 720.0
_MIN_WINDOW_L: int = 5


@dataclass
class CurvatureEstimator:
    """Sliding-window LS arc fit returning second-order curvature κ̂."""

    window_L: int
    s_bar: float
    _buffer: list = field(default_factory=list, repr=False)
    _Phi_pinv: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_kappa: float = field(default=0.0, init=False, repr=False)
    _last_fit_coeffs: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_R2: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.window_L, (int, np.integer)):
            raise TypeError(f"window_L must be int; got {type(self.window_L)}")
        if self.window_L < _MIN_WINDOW_L:
            raise ValueError(
                f"window_L must be >= {_MIN_WINDOW_L} (quadratic-fit conditioning); "
                f"got {self.window_L}"
            )
        if not (np.isfinite(self.s_bar) and self.s_bar > 0):
            raise ValueError(f"s_bar must be > 0 and finite; got {self.s_bar!r}")
        self.window_L = int(self.window_L)
        self.s_bar = float(self.s_bar)
        self._build_pseudo_inverse()
        self._buffer = []
        self._last_kappa = 0.0
        self._last_fit_coeffs = None
        self._last_R2 = 0.0

    def sigma_kappa_hat(self, sigma_eta: float) -> float:
        """σ_κ̂ [m⁻¹] = sqrt(720 σ_η² / (L⁵ s̄⁴)); hard-coded constant 720."""
        if not (np.isfinite(sigma_eta) and sigma_eta >= 0):
            raise ValueError(f"sigma_eta must be >= 0 and finite; got {sigma_eta!r}")
        var = (
            _VARIANCE_CONSTANT_720
            * sigma_eta * sigma_eta
            / (self.window_L ** 5 * self.s_bar ** 4)
        )
        return float(np.sqrt(var))

    @property
    def variance_constant(self) -> float:
        return _VARIANCE_CONSTANT_720

    def _build_pseudo_inverse(self) -> None:
        L = self.window_L
        s = np.arange(L, dtype=np.float64) * self.s_bar
        Phi = np.stack([np.ones_like(s), s, s * s], axis=1)
        self._Phi = Phi
        self._Phi_pinv = np.linalg.pinv(Phi)

    def update(self, p_meas: float) -> float:
        """Push one new measurement; return latest κ̂ (0 during warm-up)."""
        if not np.isfinite(p_meas):
            raise ValueError(f"p_meas must be finite; got {p_meas!r}")
        self._buffer.append(float(p_meas))
        if len(self._buffer) > self.window_L:
            self._buffer = self._buffer[-self.window_L:]
        if len(self._buffer) < self.window_L:
            self._last_kappa = 0.0
            self._last_fit_coeffs = None
            self._last_R2 = 0.0
            return 0.0
        y = np.asarray(self._buffer, dtype=np.float64)
        assert self._Phi_pinv is not None
        a = self._Phi_pinv @ y
        kappa = 2.0 * float(a[2])
        # R² of the fit (used by the re-acquire planner's ML-prediction gate).
        y_hat = self._Phi @ a
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        self._last_R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        self._last_kappa = kappa
        self._last_fit_coeffs = a
        return kappa

    @property
    def last_kappa(self) -> float:
        return self._last_kappa

    @property
    def last_R2(self) -> float:
        return self._last_R2

    @property
    def last_fit_coefficients(self) -> Optional[np.ndarray]:
        return self._last_fit_coeffs

    @property
    def is_warmed_up(self) -> bool:
        return len(self._buffer) >= self.window_L

    def reset(self) -> None:
        self._buffer = []
        self._last_kappa = 0.0
        self._last_fit_coeffs = None
        self._last_R2 = 0.0


def _smoke() -> None:
    est = CurvatureEstimator(window_L=20, s_bar=0.5)
    sk = est.sigma_kappa_hat(sigma_eta=0.05)
    sk_analytic = np.sqrt(720.0 * 0.05 ** 2 / (20 ** 5 * 0.5 ** 4))
    assert abs(sk - sk_analytic) / sk_analytic < 1e-9, (sk, sk_analytic)
    kappa_true = 0.05
    s = np.arange(20) * 0.5
    y = 0.5 * kappa_true * s * s
    kh = 0.0
    for yi in y:
        kh = est.update(yi)
    assert abs(kh - kappa_true) < 1e-9, kh
    assert est.last_R2 > 0.999, est.last_R2
    print("[curvature_estimator smoke] OK")


if __name__ == "__main__":
    _smoke()
