"""Re-acquire planner: model predict + Archimedean spiral (Layer-2).

Does not import from sim_python. Logic identical to the Layer-1 original:

  1. Prediction phase: if the κ-window LS fit R² >= R²_min (default 0.7),
     extrapolate the wall along the curved tangent for τ_predict·ū and aim there.
     Else skip to spiral.
  2. Archimedean spiral: r(φ) = ρ_0 + a φ with pitch 2π a <= w_FOV, so
     a = w_FOV/(2π) gives the max coverage-overlap spacing.

Coverage rate: λ = ū w_FOV / (2π ρ̄) [1/s].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_DEFAULT_R2_MIN: float = 0.7
_DEFAULT_TAU_PREDICT: float = 2.0
_DEFAULT_RHO0: float = 0.5
_DEFAULT_PHI_MAX: float = 6.0 * 2.0 * np.pi
_DEFAULT_DPHI: float = np.pi / 12.0


@dataclass(frozen=True)
class ReacquireConfig:
    """ReacquirePlanner configuration."""

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
        return float(self.w_fov / (2.0 * np.pi))


def coverage_rate(w_fov: float, u_min: float, rho_bar: float) -> float:
    """λ = ū w_FOV / (2π ρ̄) [1/s]."""
    if not (rho_bar > 0 and np.isfinite(rho_bar)):
        raise ValueError(f"rho_bar must be > 0; got {rho_bar!r}")
    if not (w_fov > 0 and u_min > 0 and np.isfinite(w_fov) and np.isfinite(u_min)):
        raise ValueError("w_fov, u_min must be > 0")
    return float(u_min * w_fov / (2.0 * np.pi * rho_bar))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    psi: float  # heading [rad]


@dataclass
class ReacquirePlanner:
    """Two-phase search: ML predict (optional) then Archimedean spiral."""

    cfg: ReacquireConfig

    def __post_init__(self) -> None:
        pitch = 2.0 * np.pi * self.cfg.spiral_pitch_a
        if pitch > self.cfg.w_fov + 1e-12:
            raise ValueError(
                f"spiral pitch {pitch} violates 2pi*a <= w_FOV={self.cfg.w_fov}"
            )

    def _predict_waypoint(self, last_pose: Pose2D, kappa_hat: float) -> Tuple[float, float]:
        ds = self.cfg.u_min * self.cfg.tau_predict
        if abs(kappa_hat) < 1e-9:
            dx_body, dy_body = ds, 0.0
        else:
            dpsi = kappa_hat * ds
            dx_body = np.sin(dpsi) / kappa_hat
            dy_body = (1.0 - np.cos(dpsi)) / kappa_hat
        cp, sp = np.cos(last_pose.psi), np.sin(last_pose.psi)
        dx = cp * dx_body - sp * dy_body
        dy = sp * dx_body + cp * dy_body
        return (last_pose.x + dx, last_pose.y + dy)

    def _spiral_waypoints(self, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        a = self.cfg.spiral_pitch_a
        n = int(np.floor(self.cfg.phi_max / self.cfg.dphi)) + 1
        phi = np.arange(n, dtype=np.float64) * self.cfg.dphi
        rho = self.cfg.rho_0 + a * phi
        x = center[0] + rho * np.cos(phi)
        y = center[1] + rho * np.sin(phi)
        return [(float(x[i]), float(y[i])) for i in range(n)]

    def plan_waypoints(
        self, last_pose: Pose2D, kappa_hat: float, R2: float, side: str = "L",
    ) -> List[Tuple[float, float]]:
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
        if side == "R":
            cx, cy = center
            spiral = [(x, 2 * cy - y) for (x, y) in spiral]
        waypoints.extend(spiral)
        return waypoints

    def coverage_rate(self, rho_bar: float | None = None) -> float:
        if rho_bar is None:
            rho_bar = self.cfg.rho_bar
            if rho_bar is None:
                rho_bar = self.cfg.rho_0 + 0.5 * self.cfg.phi_max * self.cfg.spiral_pitch_a
        return coverage_rate(self.cfg.w_fov, self.cfg.u_min, rho_bar)


def _smoke() -> None:
    cfg = ReacquireConfig(w_fov=np.pi / 2, u_min=1.0)
    assert 2 * np.pi * cfg.spiral_pitch_a <= cfg.w_fov + 1e-12
    p = ReacquirePlanner(cfg=cfg)
    last = Pose2D(x=0.0, y=0.0, psi=0.0)
    wps = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.9, side="L")
    assert len(wps) > 5 and wps[0][0] > 0
    wps2 = p.plan_waypoints(last_pose=last, kappa_hat=0.0, R2=0.3, side="L")
    assert len(wps2) == len(wps) - 1
    assert p.coverage_rate() > 0
    print("[reacquire_planner smoke] OK")


if __name__ == "__main__":
    _smoke()
