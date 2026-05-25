"""Re-acquire planner: ML prediction + Archimedean spiral search.

Implements Lemma 3's re-acquisition strategy: a short maximum-likelihood
wall extrapolation followed by an Archimedean spiral search.

Strategy
--------
On entry to R mode the planner is given the last known position +
heading + curvature estimate.  It produces a sequence of waypoints
the AUV should visit:

1. **ML prediction phase**: if the most recent κ-window LS fit has
   R² ≥ R²_min (default 0.7), extrapolate the wall along the tangent
   + κ for a short distance τ_predict · ū, and aim there as the first
   waypoint.  If R² < R²_min, the fit is not reliable and we skip
   prediction, going straight to spiral.

2. **Archimedean spiral phase** (Koopman search):
   spiral outward from the last-known location with

       r(φ) = ρ_0 + a · φ                    (r in m, φ in rad)

   The spiral pitch ``2π a`` must be ≤ w_FOV (lateral) to ensure
   coverage overlap between adjacent arms — this is the explicit
   constraint from the spec:

       2π · a ≤ w_FOV · ρ̄                    (large-ρ form)

   We default to the more conservative tight form ``2π a ≤ w_FOV``
   (interpreting w_FOV as a fractional angular sweep wider than
   needed for any ρ in practice).  Setting ``a = w_FOV / (2π)`` gives
   the maximum spacing satisfying the overlap constraint.

Detection probability per unit time
-----------------------------------
    λ(w_FOV, ū)  ≈  ū · w_FOV / (2π · ρ̄)

This is the *geometric* coverage rate (sensor sweep area per unit
time at radius ρ̄ ).  Used by the FSM to size the R-budget against
T*; computed here as a static helper.

API
---
``ReacquirePlanner(cfg)`` — constructed once per run.

``plan_waypoints(last_pose, kappa_hat, R2, side)`` →
    list of (x, y) waypoints in world frame, ordered.

``coverage_rate(rho_bar)`` → float (m²/s × 1/m² = 1/s coverage rate)

Determinism
-----------
Spiral is deterministic in (last_pose, cfg).  No RNG.  If a future
stochastic variant is needed (e.g. random rotation of the spiral
start angle) it must take a numpy.random.Generator(seed) so runs stay
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

_DEFAULT_R2_MIN: float = 0.7  # min LS-fit R² for ML prediction to be used
_DEFAULT_TAU_PREDICT: float = 2.0  # s — short forward extrapolation
_DEFAULT_RHO0: float = 0.5  # m — initial spiral radius
_DEFAULT_PHI_MAX: float = 6.0 * 2.0 * np.pi  # 6 turns
_DEFAULT_DPHI: float = np.pi / 12.0  # 15°-spaced spiral waypoints


@dataclass(frozen=True)
class ReacquireConfig:
    """ReacquirePlanner configuration.

    Attributes
    ----------
    w_fov : float
        Field-of-view full width [rad]; sets spiral pitch upper bound.
    u_min : float
        Lower bound on surge speed ū [m/s]; used in λ coverage rate.
    R2_min : float
        Minimum R² for ML prediction to be used; below this we go
        straight to spiral.  Default 0.7.
    tau_predict : float
        Forward extrapolation time [s] for ML prediction waypoint.
    rho_0 : float
        Initial spiral radius [m] (avoids degeneracy at φ=0).
    phi_max : float
        Maximum spiral angle [rad]; sets number of spiral waypoints.
    dphi : float
        Angular spacing between spiral waypoints [rad].
    rho_bar : float, default = rho_0 + π·a (mid-spiral)
        Reference radius for coverage-rate calculation [m].  If
        ``None`` (default) it is set internally to rho_0 + (phi_max/2)·a.
    """

    w_fov: float
    u_min: float
    R2_min: float = _DEFAULT_R2_MIN
    tau_predict: float = _DEFAULT_TAU_PREDICT
    rho_0: float = _DEFAULT_RHO0
    phi_max: float = _DEFAULT_PHI_MAX
    dphi: float = _DEFAULT_DPHI
    rho_bar: float | None = None

    def __post_init__(self) -> None:
        for name in ("w_fov", "u_min", "R2_min", "tau_predict", "rho_0",
                     "phi_max", "dphi"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"ReacquireConfig.{name} must be > 0; got {v!r}")
        if not (0 < self.R2_min < 1):
            raise ValueError(f"R2_min must be in (0,1); got {self.R2_min}")

    @property
    def spiral_pitch_a(self) -> float:
        """Archimedean spiral pitch ``a`` satisfying 2π · a ≤ w_FOV."""
        return float(self.w_fov / (2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def coverage_rate(w_fov: float, u_min: float, rho_bar: float) -> float:
    """λ(w_FOV, ū, ρ̄) = ū · w_FOV / (2π · ρ̄) [1/s].

    Geometric coverage rate of an Archimedean spiral at radius ρ̄.
    """
    if not (rho_bar > 0 and np.isfinite(rho_bar)):
        raise ValueError(f"rho_bar must be > 0; got {rho_bar!r}")
    if not (w_fov > 0 and u_min > 0 and np.isfinite(w_fov) and np.isfinite(u_min)):
        raise ValueError(f"w_fov, u_min must be > 0")
    return float(u_min * w_fov / (2.0 * np.pi * rho_bar))


@dataclass(frozen=True)
class Pose2D:
    """Minimal 2D pose."""
    x: float
    y: float
    psi: float  # heading [rad]


# --------------------------------------------------------------------------- #
# Planner                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ReacquirePlanner:
    """Two-phase search: ML predict (optional) then Archimedean spiral.

    Attributes
    ----------
    cfg : ReacquireConfig
    """

    cfg: ReacquireConfig

    def __post_init__(self) -> None:
        # Sanity-check pitch constraint.
        pitch = 2.0 * np.pi * self.cfg.spiral_pitch_a
        if pitch > self.cfg.w_fov + 1e-12:
            raise ValueError(
                f"spiral pitch {pitch} violates 2π·a ≤ w_FOV={self.cfg.w_fov}"
            )

    # ------------------------------------------------------------------- #
    # ML prediction waypoint                                              #
    # ------------------------------------------------------------------- #

    def _predict_waypoint(
        self,
        last_pose: Pose2D,
        kappa_hat: float,
    ) -> Tuple[float, float]:
        """Forward-extrapolate a short distance along the curved tangent.

        Uses the small-angle clothoid-like form:
            Δψ = κ̂ · ū · τ_predict
            Δs = ū · τ_predict
            (Δx, Δy) ≈ (sin(Δψ)/κ̂, (1 - cos(Δψ))/κ̂) if κ̂ != 0
                       (Δs · cos ψ, Δs · sin ψ)       otherwise

        rotated into world frame at the start by ψ.
        """
        ds = self.cfg.u_min * self.cfg.tau_predict
        if abs(kappa_hat) < 1e-9:
            dx_body = ds
            dy_body = 0.0
        else:
            dpsi = kappa_hat * ds
            dx_body = np.sin(dpsi) / kappa_hat
            dy_body = (1.0 - np.cos(dpsi)) / kappa_hat
        cp = np.cos(last_pose.psi)
        sp = np.sin(last_pose.psi)
        dx = cp * dx_body - sp * dy_body
        dy = sp * dx_body + cp * dy_body
        return (last_pose.x + dx, last_pose.y + dy)

    # ------------------------------------------------------------------- #
    # Spiral waypoints                                                    #
    # ------------------------------------------------------------------- #

    def _spiral_waypoints(self, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Archimedean spiral r(φ) = ρ_0 + a·φ around center."""
        a = self.cfg.spiral_pitch_a
        n = int(np.floor(self.cfg.phi_max / self.cfg.dphi)) + 1
        phi = np.arange(n, dtype=np.float64) * self.cfg.dphi
        rho = self.cfg.rho_0 + a * phi
        x = center[0] + rho * np.cos(phi)
        y = center[1] + rho * np.sin(phi)
        return [(float(x[i]), float(y[i])) for i in range(n)]

    # ------------------------------------------------------------------- #
    # Public planner                                                      #
    # ------------------------------------------------------------------- #

    def plan_waypoints(
        self,
        last_pose: Pose2D,
        kappa_hat: float,
        R2: float,
        side: str = "L",
    ) -> List[Tuple[float, float]]:
        """Return ordered waypoint list for R-mode search.

        Parameters
        ----------
        last_pose : Pose2D
            Last confident AUV pose at the moment of loss.
        kappa_hat : float
            Most recent curvature estimate [m⁻¹].
        R2 : float
            R² of most recent LS κ-fit (0 ≤ R² ≤ 1).  If ≥ R2_min,
            the ML prediction phase is used; otherwise it is skipped.
        side : "L" or "R"
            Standoff side; spiral chirality follows side ("L" → CCW).

        Returns
        -------
        list of (x, y)
            Waypoints in world frame, first-visited first.  Length
            = (1 if ML used else 0) + number of spiral nodes.
        """
        if side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R'; got {side!r}")
        if not (0 <= R2 <= 1):
            raise ValueError(f"R2 must be in [0,1]; got {R2!r}")
        waypoints: List[Tuple[float, float]] = []
        center = (last_pose.x, last_pose.y)

        if R2 >= self.cfg.R2_min:
            wp = self._predict_waypoint(last_pose, kappa_hat)
            waypoints.append(wp)
            center = wp

        spiral = self._spiral_waypoints(center)
        # For side="R" reflect the spiral around the local heading axis
        # (flip y of each point relative to center) so the chirality
        # is reversed.  Simple sign flip on Δy.
        if side == "R":
            cx, cy = center
            spiral = [(x, 2 * cy - y) for (x, y) in spiral]
        waypoints.extend(spiral)
        return waypoints

    # ------------------------------------------------------------------- #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------- #

    def coverage_rate(self, rho_bar: float | None = None) -> float:
        """λ at the supplied (or config-default) ρ̄ [1/s]."""
        if rho_bar is None:
            rho_bar = self.cfg.rho_bar
            if rho_bar is None:
                rho_bar = self.cfg.rho_0 + 0.5 * self.cfg.phi_max * self.cfg.spiral_pitch_a
        return coverage_rate(self.cfg.w_fov, self.cfg.u_min, rho_bar)


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    cfg = ReacquireConfig(w_fov=np.pi / 2, u_min=1.0)
    print(f"[smoke] spiral pitch a = {cfg.spiral_pitch_a:.4f} m/rad")
    assert 2 * np.pi * cfg.spiral_pitch_a <= cfg.w_fov + 1e-12

    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=0.0, y=0.0, psi=0.0)
    wps = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.9, side="L")
    print(f"[smoke] {len(wps)} waypoints; first 3 = {wps[:3]}")
    assert len(wps) > 5
    assert wps[0][0] > 0  # ML predict goes forward

    # Without ML prediction (R²<R²_min)
    wps2 = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.3, side="L")
    print(f"[smoke] (no-ML) {len(wps2)} waypoints; first = {wps2[0]}")
    assert len(wps2) == len(wps) - 1

    # Coverage rate
    lam = p.coverage_rate()
    print(f"[smoke] coverage rate λ = {lam:.6f} /s")
    assert lam > 0

    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
