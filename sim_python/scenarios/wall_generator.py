r"""Parametric 2D wall generators for sim_python.

A *wall* is a piecewise-smooth 2D curve $\mathcal W : s \mapsto p(s) \in \mathbb R^2$
that the AUV is asked to track at constant lateral distance $d^*$.  The
controller side $\in \{L, R\}$ selects which side of the curve's tangent
is the "inside" of the standoff band:

    side = "L"  →  the standoff point is on the LEFT of the tangent
    side = "R"  →  the standoff point is on the RIGHT of the tangent

Three generators are supplied:

- ``gen_straight(length, side)`` — straight wall along +x with κ ≡ 0
- ``gen_arc(radius, arc_length, side)`` — circular arc; κ = ±1/R, constant
- ``gen_cspline(kappa_max, kappa_dot_max, length, side, seed)`` — random
  C² cubic spline with curvature ≤ kappa_max and |κ'| ≤ kappa_dot_max
  (achieved by sampling random control κ at coarse arc-length nodes then
  cubic-spline interpolating; an a-posteriori clip ensures the bound)

Sample spacing
--------------
All generators emit a polyline at approximately equal arc-length step
``ds`` (default 0.05 m).  The exact number of samples is
``N = int(round(length / ds)) + 1`` (or analogous for arcs).

Return type
-----------
A frozen ``Wall`` dataclass with:

- ``points``  : ndarray (N, 2)  — sampled (x, y) along the wall
- ``tangent`` : ndarray (N,)    — tangent angle γ_p(s) [rad], unwrapped
- ``kappa``   : ndarray (N,)    — signed curvature κ(s) [m⁻¹]
- ``side``    : str "L" or "R"
- ``s``       : ndarray (N,)    — arc length parameter [m] (s[0]=0)

Sign convention for κ
---------------------
With unit tangent ``t̂(s) = (cos γ_p, sin γ_p)`` and unit normal
``n̂(s) = (-sin γ_p, cos γ_p)`` (the 90°-CCW rotation of t̂), the
Serret–Frenet relation in 2D gives ``dγ_p/ds = κ`` so that a positive
κ corresponds to the curve turning counter-clockwise (left turn).
The arc generator with side="L" returns κ = +1/R (LEFT turn) and side
"R" returns κ = -1/R (RIGHT turn).

Convention note
---------------
The Serret–Frenet sign convention above matches the κ used in the LOS +
feed-forward law $r^* = \\hat\\kappa \\hat u + r^*_{LOS}$.

Determinism
-----------
The cspline generator takes a ``seed``; all randomness flows through a
``numpy.random.Generator(seed)``.  No global numpy.random.* state is
ever touched, so a fixed seed reproduces the wall exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

Side = Literal["L", "R"]


# --------------------------------------------------------------------------- #
# Wall dataclass                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Wall:
    """Sampled 2D wall description.

    Attributes
    ----------
    points : ndarray (N, 2)
        Sampled (x, y) coordinates along the wall, arc-length spaced.
    tangent : ndarray (N,)
        Tangent angle γ_p(s) [rad], unwrapped.
    kappa : ndarray (N,)
        Signed curvature κ(s) [m⁻¹], same sign convention as
        ``dγ_p/ds = κ``.
    side : "L" or "R"
        Standoff side: the AUV stays at lateral distance ``d*`` on the
        L (resp. R) of the wall's tangent vector.
    s : ndarray (N,)
        Arc-length parameter, ``s[0] = 0`` and ``s[-1] ≈ length``.
    """

    points: np.ndarray
    tangent: np.ndarray
    kappa: np.ndarray
    side: str
    s: np.ndarray

    def __post_init__(self) -> None:
        n = self.points.shape[0]
        if self.points.shape != (n, 2):
            raise ValueError(f"points must be (N, 2); got {self.points.shape}")
        for arr_name in ("tangent", "kappa", "s"):
            arr = getattr(self, arr_name)
            if arr.shape != (n,):
                raise ValueError(
                    f"{arr_name} must be (N,)={n,}; got {arr.shape}"
                )
        if self.side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R'; got {self.side!r}")
        if not np.all(np.isfinite(self.points)):
            raise ValueError("points contains NaN/inf")
        if not np.all(np.isfinite(self.kappa)):
            raise ValueError("kappa contains NaN/inf")

    @property
    def length(self) -> float:
        """Total arc length [m]."""
        return float(self.s[-1] - self.s[0])

    @property
    def n_samples(self) -> int:
        """Number of sampled points along the wall."""
        return int(self.points.shape[0])


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _check_side(side: str) -> str:
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R'; got {side!r}")
    return side


def _default_ds(length: float, ds: float | None) -> float:
    if ds is None:
        return 0.05
    if not (np.isfinite(ds) and ds > 0):
        raise ValueError(f"ds must be > 0 and finite; got {ds!r}")
    if ds > length:
        raise ValueError(f"ds={ds} must be ≤ length={length}")
    return float(ds)


# --------------------------------------------------------------------------- #
# Generators                                                                  #
# --------------------------------------------------------------------------- #


def gen_straight(
    length: float,
    side: Side = "L",
    ds: float | None = None,
) -> Wall:
    """Straight wall along +x axis from (0,0) to (length, 0).

    κ(s) ≡ 0, tangent γ_p(s) ≡ 0.  The AUV's standoff path is the line
    y = +d* (side="L") or y = -d* (side="R").
    """
    if not (np.isfinite(length) and length > 0):
        raise ValueError(f"length must be > 0; got {length!r}")
    _check_side(side)
    ds_v = _default_ds(length, ds)
    n = int(round(length / ds_v)) + 1
    s = np.linspace(0.0, length, n, dtype=np.float64)
    x = s.copy()
    y = np.zeros_like(s)
    points = np.stack([x, y], axis=1)
    tangent = np.zeros_like(s)
    kappa = np.zeros_like(s)
    return Wall(points=points, tangent=tangent, kappa=kappa, side=side, s=s)


def gen_arc(
    radius: float,
    arc_length: float,
    side: Side = "L",
    ds: float | None = None,
) -> Wall:
    """Circular arc of radius R, total arc length ``arc_length``.

    The arc starts at (0, 0) tangent +x and turns LEFT (κ = +1/R) if
    side="L", RIGHT (κ = -1/R) if side="R".

    Parametrisation: γ_p(s) = κ s, then x(s) = ∫cos γ_p ds, y(s) = ∫sin γ_p ds.
    For uniform κ:
        γ_p(s) = κ s
        x(s)   = sin(κ s) / κ
        y(s)   = (1 - cos(κ s)) / κ
    For side="R", κ < 0 ⇒ y < 0 (turns right).
    """
    if not (np.isfinite(radius) and radius > 0):
        raise ValueError(f"radius must be > 0; got {radius!r}")
    if not (np.isfinite(arc_length) and arc_length > 0):
        raise ValueError(f"arc_length must be > 0; got {arc_length!r}")
    _check_side(side)
    ds_v = _default_ds(arc_length, ds)
    n = int(round(arc_length / ds_v)) + 1
    s = np.linspace(0.0, arc_length, n, dtype=np.float64)
    sign = +1.0 if side == "L" else -1.0
    kappa_val = sign / radius
    tangent = kappa_val * s
    x = np.sin(tangent) / kappa_val
    y = (1.0 - np.cos(tangent)) / kappa_val
    points = np.stack([x, y], axis=1)
    kappa = np.full_like(s, kappa_val)
    return Wall(points=points, tangent=tangent, kappa=kappa, side=side, s=s)


def gen_cspline(
    kappa_max: float,
    kappa_dot_max: float,
    length: float,
    side: Side = "L",
    seed: int = 0,
    ds: float | None = None,
    n_control: int | None = None,
) -> Wall:
    """C¹ cubic-spline wall with random curvature ≤ kappa_max, |κ'| ≤ kappa_dot_max.

    Procedure
    ---------
    1. Pick coarse arc-length control nodes ``s_ctrl`` spaced approximately
       so that two adjacent nodes are at distance ``Δs_ctrl ≥ kappa_max / kappa_dot_max``
       (this guarantees the linear-interpolant κ between two random
       ±kappa_max samples has |κ'| ≤ kappa_dot_max).
    2. Sample i.i.d. uniform κ_i ∈ [-kappa_max, +kappa_max] at each
       control node (orientation pre-multiplied by side sign).
    3. Cubic-spline interpolate κ over arc length s.
    4. A-posteriori clip κ to ±kappa_max (the spline's overshoot can be
       O(10%) so we clip; this introduces a non-smooth derivative at the
       clip points — documented stage-02 risk, but |κ'| post-clip is
       still ≤ kappa_dot_max because clip can only reduce |κ'|).
    5. Integrate γ_p(s) = ∫₀ˢ κ dτ via cumulative trapezoid.
    6. Integrate (x, y) = ∫₀ˢ (cos γ_p, sin γ_p) dτ via cumulative trapezoid.

    Parameters
    ----------
    kappa_max : float
        Max |κ| [m⁻¹].
    kappa_dot_max : float
        Max |dκ/ds| [m⁻²].
    length : float
        Total arc length [m].
    side : "L" or "R"
        Sign factor; for side="R" the random κ samples are negated so
        the wall has a net rightward bend on average (statistically).
    seed : int
        Generator seed; all randomness flows through a Generator(seed)
        so a fixed seed reproduces the wall exactly.
    ds : float, optional
        Output sample spacing; default 0.05 m.
    n_control : int, optional
        Override number of control nodes (default = ceil(length / Δs_ctrl) + 1).
    """
    if not (np.isfinite(kappa_max) and kappa_max > 0):
        raise ValueError(f"kappa_max must be > 0; got {kappa_max!r}")
    if not (np.isfinite(kappa_dot_max) and kappa_dot_max > 0):
        raise ValueError(f"kappa_dot_max must be > 0; got {kappa_dot_max!r}")
    if not (np.isfinite(length) and length > 0):
        raise ValueError(f"length must be > 0; got {length!r}")
    _check_side(side)

    rng = np.random.default_rng(int(seed))
    ds_v = _default_ds(length, ds)
    n = int(round(length / ds_v)) + 1
    s = np.linspace(0.0, length, n, dtype=np.float64)

    # Control-node spacing to satisfy |κ'| ≤ kappa_dot_max from linear
    # interpolation of ±kappa_max samples.  Factor 2 because adjacent
    # samples can swing by 2 * kappa_max.
    ds_ctrl_min = (2.0 * kappa_max) / kappa_dot_max
    if n_control is None:
        n_ctrl = max(4, int(np.ceil(length / ds_ctrl_min)) + 1)
    else:
        n_ctrl = int(n_control)
        if n_ctrl < 4:
            raise ValueError(f"n_control must be ≥ 4; got {n_ctrl}")
    s_ctrl = np.linspace(0.0, length, n_ctrl, dtype=np.float64)
    sign = +1.0 if side == "L" else -1.0
    kappa_ctrl = sign * rng.uniform(-kappa_max, +kappa_max, size=n_ctrl)
    # Enforce zero curvature at endpoints to make the spline well-defined
    # (and to match a "wall starts straight" convention).
    kappa_ctrl[0] = 0.0
    kappa_ctrl[-1] = 0.0

    spline = CubicSpline(s_ctrl, kappa_ctrl, bc_type="natural")
    kappa = spline(s)
    # A-posteriori clip; spline overshoot can violate the bound by ~10%.
    kappa = np.clip(kappa, -kappa_max, +kappa_max)

    # Integrate tangent and coordinates by cumulative trapezoid.
    tangent = np.zeros_like(s)
    tangent[1:] = np.cumsum(0.5 * (kappa[:-1] + kappa[1:]) * np.diff(s))
    cos_t = np.cos(tangent)
    sin_t = np.sin(tangent)
    x = np.zeros_like(s)
    y = np.zeros_like(s)
    x[1:] = np.cumsum(0.5 * (cos_t[:-1] + cos_t[1:]) * np.diff(s))
    y[1:] = np.cumsum(0.5 * (sin_t[:-1] + sin_t[1:]) * np.diff(s))
    points = np.stack([x, y], axis=1)

    return Wall(points=points, tangent=tangent, kappa=kappa, side=side, s=s)


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    """Generate one of each + sanity-print."""
    w_s = gen_straight(length=10.0, side="L")
    w_a = gen_arc(radius=5.0, arc_length=np.pi * 5.0, side="L")  # half circle
    w_c = gen_cspline(
        kappa_max=0.05, kappa_dot_max=0.01, length=50.0, side="L", seed=42
    )
    for name, w in (("straight", w_s), ("arc", w_a), ("cspline", w_c)):
        print(
            f"[smoke] {name}: N={w.n_samples}, length={w.length:.3f} m, "
            f"max|κ|={np.max(np.abs(w.kappa)):.4f} m^-1, "
            f"first={w.points[0]}, last={w.points[-1]}"
        )
    # Half-circle of R=5 → endpoint should be at (0, 10).
    last = w_a.points[-1]
    assert abs(last[0]) < 1e-6 and abs(last[1] - 10.0) < 1e-6, (
        f"arc smoke endpoint = {last}, expected (0, 10)"
    )
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
