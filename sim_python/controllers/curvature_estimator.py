"""Sliding-window LS curvature estimator with the 720 variance closure.

Implements the discrete-sensing curvature-estimate variance σ_κ̂²
closed form used throughout the contour-tracking analysis.

Derivation recap
----------------
Fit a second-degree polynomial y(s) = a₀ + a₁ s + a₂ s² to L equally
arc-length-spaced lateral-distance samples ``y_k`` with i.i.d. noise
``η_k ~ N(0, σ_η²)``.  The Vandermonde design matrix Φ in arc-length s
gives via LS:

    Cov(â) = (ΦᵀΦ)⁻¹ σ_η²

and the [2,2] element after the standard normalisation isolates the
quadratic-coefficient variance:

    Var(â₂) = 180 / (L⁵ s̄⁴) · σ_η²

The geometric translation κ̂ = 2â₂ then gives the closed form:

    σ_κ̂² = (4 · 180) / (L⁵ s̄⁴) · σ_η²  =  720 σ_η² / (L⁵ s̄⁴)

The constant 720 is exact and is hard-coded in this module.  Any other
value (e.g. the dimensionally-inconsistent placeholder σ_η²/(L·s̄²),
which is not a valid closed form) propagates into the noise-driven
dwell time T*_N and is therefore guarded by the unit tests.

Engineering lower bound on L
----------------------------
The design-matrix condition-number analysis requires L ≥ 5.  Below
that the (ΦᵀΦ)⁻¹ becomes ill-conditioned and the 720 constant ceases
to be the leading-order asymptote.  We enforce L ≥ 5 in
``CurvatureEstimator.__init__`` as a hard error.

API
---
- ``CurvatureEstimator(window_L, s_bar)``: construct with the sliding-
  window length L and the (fixed, equal) arc-length spacing s̄.
- ``update(p_meas) -> kappa_hat``: push one new (arc-length-fixed)
  lateral-distance measurement; returns latest κ̂ estimate.
- ``sigma_kappa_hat(sigma_eta) -> float``: closed-form σ_κ̂ given the
  measurement noise σ_η, using the hard-coded constant 720.
- ``reset() -> None``: clear the sliding window.

Note: the estimator returns 0 until L samples have arrived (no
extrapolation; the FSM should disable any κ̂-dependent feed-forward
during that warm-up period).

Convention for ``p_meas``
-------------------------
``p_meas`` is the **scalar** lateral distance from a fixed reference
arc-length offset (e.g. the closest-point projection of the AUV onto
the wall).  In the closed-loop driver this is computed by projecting
the AUV's body position onto the wall polyline and measuring the
signed perpendicular distance; the equal-arc-length assumption is
enforced by sampling at the AUV's own (approximately constant) speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Hard-coded variance constant — DO NOT CHANGE                                #
# --------------------------------------------------------------------------- #

# σ_κ̂² = 720 σ_η² / (L⁵ s̄⁴).  The 720 = 4 · 180 = 4 · c₂ where c₂ = 180
# is the L→∞ asymptote of [(Ψᵀ Ψ)⁻¹]_22 for the Legendre-normalised
# design matrix; the factor 4 is the (2·a₂)² geometric translation
# κ̂ = 2 â₂ → 2² = 4.
_VARIANCE_CONSTANT_720: float = 720.0

# Engineering lower bound on the window length L (design-matrix
# condition-number requirement).
_MIN_WINDOW_L: int = 5


# --------------------------------------------------------------------------- #
# Estimator class                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class CurvatureEstimator:
    """Sliding-window LS arc fit returning second-order curvature κ̂.

    Attributes
    ----------
    window_L : int
        Number of (most-recent) samples in the sliding window.  Must be
        ≥ 5 for a well-conditioned design matrix.
    s_bar : float
        Equal arc-length spacing between adjacent samples [m].  Must
        be > 0.

    Internal state
    --------------
    _buffer : deque-like list of latest measurements (length ≤ window_L)
    """

    window_L: int
    s_bar: float
    _buffer: list = field(default_factory=list, repr=False)
    _Phi_pinv: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_kappa: float = field(default=0.0, init=False, repr=False)
    _last_fit_coeffs: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.window_L, (int, np.integer)):
            raise TypeError(f"window_L must be int; got {type(self.window_L)}")
        if self.window_L < _MIN_WINDOW_L:
            raise ValueError(
                f"window_L must be ≥ {_MIN_WINDOW_L} (design-matrix "
                f"condition); got {self.window_L}"
            )
        if not (np.isfinite(self.s_bar) and self.s_bar > 0):
            raise ValueError(f"s_bar must be > 0 and finite; got {self.s_bar!r}")
        self.window_L = int(self.window_L)
        self.s_bar = float(self.s_bar)
        self._build_pseudo_inverse()
        self._buffer = []
        self._last_kappa = 0.0
        self._last_fit_coeffs = None

    # ------------------------------------------------------------------- #
    # Closed-form variance (HARD-CITES 720)                               #
    # ------------------------------------------------------------------- #

    def sigma_kappa_hat(self, sigma_eta: float) -> float:
        """Return σ_κ̂ in m⁻¹ from the closed-form variance.

        σ_κ̂² = 720 · σ_η² / (L⁵ · s̄⁴).

        Parameters
        ----------
        sigma_eta : float
            Measurement-noise standard deviation [m]; must be ≥ 0.

        Returns
        -------
        sigma_kappa : float
            Standard deviation of the LS curvature estimate [m⁻¹].
        """
        if not (np.isfinite(sigma_eta) and sigma_eta >= 0):
            raise ValueError(f"sigma_eta must be ≥ 0 and finite; got {sigma_eta!r}")
        var = (
            _VARIANCE_CONSTANT_720
            * sigma_eta * sigma_eta
            / (self.window_L ** 5 * self.s_bar ** 4)
        )
        return float(np.sqrt(var))

    @property
    def variance_constant(self) -> float:
        """The hard-coded variance constant 720."""
        return _VARIANCE_CONSTANT_720

    # ------------------------------------------------------------------- #
    # Sliding-window update                                               #
    # ------------------------------------------------------------------- #

    def _build_pseudo_inverse(self) -> None:
        """Pre-compute (ΦᵀΦ)⁻¹ Φᵀ for the LS fit (Φ in arc length).

        Φ has rows [1, s_k, s_k²] for s_k = k · s̄, k = 0..L-1.
        Caching this makes update() O(L) instead of O(L²) per call.
        """
        L = self.window_L
        s = np.arange(L, dtype=np.float64) * self.s_bar
        Phi = np.stack([np.ones_like(s), s, s * s], axis=1)
        self._Phi_pinv = np.linalg.pinv(Phi)

    def update(self, p_meas: float) -> float:
        """Push one new measurement; return latest κ̂ estimate.

        Returns 0 until ``window_L`` samples have arrived (warm-up).

        Parameters
        ----------
        p_meas : float
            Scalar lateral measurement at the next arc-length grid
            point (equal s̄ spacing assumed; the FSM / driver is
            responsible for sampling at constant spacing).

        Returns
        -------
        kappa_hat : float
            κ̂ = 2 â₂ where â = (ΦᵀΦ)⁻¹ Φᵀ y is the LS fit.  Returns
            ``0.0`` if fewer than window_L samples are buffered.
        """
        if not np.isfinite(p_meas):
            raise ValueError(f"p_meas must be finite; got {p_meas!r}")
        self._buffer.append(float(p_meas))
        # Keep only the last window_L samples.
        if len(self._buffer) > self.window_L:
            self._buffer = self._buffer[-self.window_L :]
        if len(self._buffer) < self.window_L:
            self._last_kappa = 0.0
            self._last_fit_coeffs = None
            return 0.0
        y = np.asarray(self._buffer, dtype=np.float64)
        assert self._Phi_pinv is not None  # built in __post_init__
        a = self._Phi_pinv @ y
        # κ ≈ 2 a_2 (small-slope curvature of y(s) = a0 + a1 s + a2 s²)
        kappa = 2.0 * float(a[2])
        self._last_kappa = kappa
        self._last_fit_coeffs = a
        return kappa

    @property
    def last_kappa(self) -> float:
        """Most recent κ̂ (or 0 during warm-up)."""
        return self._last_kappa

    @property
    def last_fit_coefficients(self) -> Optional[np.ndarray]:
        """Most recent (a0, a1, a2) LS fit, or None during warm-up."""
        return self._last_fit_coeffs

    @property
    def is_warmed_up(self) -> bool:
        """True iff at least window_L samples have arrived."""
        return len(self._buffer) >= self.window_L

    def reset(self) -> None:
        """Clear the sliding window."""
        self._buffer = []
        self._last_kappa = 0.0
        self._last_fit_coeffs = None


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    # Closed-form variance check: L=20, s̄=0.5, σ_η=0.05 → σ_κ̂ via 720.
    est = CurvatureEstimator(window_L=20, s_bar=0.5)
    sk = est.sigma_kappa_hat(sigma_eta=0.05)
    sk_analytic = np.sqrt(720.0 * 0.05 ** 2 / (20 ** 5 * 0.5 ** 4))
    print(f"[smoke] σ_κ̂ = {sk:.6e} vs analytic {sk_analytic:.6e}")
    assert abs(sk - sk_analytic) / sk_analytic < 1e-9

    # Fit a known constant-κ arc: y(s) = (s²)/2 · κ for small κ.
    kappa_true = 0.05
    rng = np.random.default_rng(seed=42)
    L = 20
    s_bar = 0.5
    est2 = CurvatureEstimator(window_L=L, s_bar=s_bar)
    # Generate L noiseless samples on the arc y = (κ/2) s².
    s = np.arange(L) * s_bar
    y = 0.5 * kappa_true * s * s
    for yi in y:
        kh = est2.update(yi)
    print(f"[smoke] κ̂ (noiseless arc) = {kh:.6f}, true = {kappa_true}")
    assert abs(kh - kappa_true) < 1e-9, f"κ̂ should recover true κ exactly"
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
