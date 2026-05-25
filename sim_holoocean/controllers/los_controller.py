"""LOS guidance with curvature feed-forward (Layer-2, no sim_python import).

The control law is byte-for-byte equivalent to the Layer-1 original so the two
engines compare the SAME controller:

    r^*(t) = K_ff · κ̂(t) · u(t)  +  r^*_{LOS}(d - d^*, θ; Δ)  +  (- K_θ · θ)

Caharija 2016 saturated LOS (Fossen Handbook ch.10 §10.4.2):

    sin-style   :  r^*_{LOS} = - K_p (d - d^*) / sqrt((d - d^*)^2 + Δ^2)
    atan2-style :  r^*_{LOS} = - K_p · atan2(d - d^*, Δ)

The LOS term implements the standstill-error Lyapunov design (ω_LOS ≈ ū/Δ); the
feed-forward term is the curvature anticipation κ·û.

State convention:
  d   : signed lateral distance to wall [m]
  θ   : heading misalignment ψ - γ_p [rad]
  κ̂   : curvature estimate [m⁻¹]  (+ LEFT turn, - RIGHT turn)
  u   : surge speed [m/s]
  d*  : standoff set-point [m]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LOSConfig:
    """LOS + feed-forward parameters (see module docstring)."""

    Delta: float
    K_p: float
    K_ff: float = 1.0
    K_theta: float = 0.0
    style: str = "sin"
    d_star: float = 0.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.Delta) and self.Delta > 0):
            raise ValueError(f"Delta must be > 0; got {self.Delta!r}")
        if not (np.isfinite(self.K_p) and self.K_p > 0):
            raise ValueError(f"K_p must be > 0; got {self.K_p!r}")
        if not np.isfinite(self.K_ff):
            raise ValueError(f"K_ff must be finite; got {self.K_ff!r}")
        if not np.isfinite(self.K_theta) or self.K_theta < 0:
            raise ValueError(f"K_theta must be >= 0 and finite; got {self.K_theta!r}")
        if self.style not in ("sin", "atan2"):
            raise ValueError(f"style must be 'sin' or 'atan2'; got {self.style!r}")


def compute_r_star(
    d: float,
    theta: float,
    kappa_hat: float,
    u: float,
    cfg: LOSConfig,
) -> float:
    """Commanded yaw rate r^* [rad/s] given current error state.

    Hull-side saturation (|r| <= r_max) is applied downstream in the driver's
    rudder mapping; this function returns the unsaturated guidance command.
    """
    if not np.isfinite(d):
        raise ValueError(f"d must be finite; got {d!r}")
    if not np.isfinite(theta):
        raise ValueError(f"theta must be finite; got {theta!r}")
    if not np.isfinite(kappa_hat):
        raise ValueError(f"kappa_hat must be finite; got {kappa_hat!r}")
    if not np.isfinite(u):
        raise ValueError(f"u must be finite; got {u!r}")

    e = d - cfg.d_star
    if cfg.style == "sin":
        r_los = -cfg.K_p * e / np.sqrt(e * e + cfg.Delta * cfg.Delta)
    else:  # atan2-style
        r_los = -cfg.K_p * float(np.arctan2(e, cfg.Delta))

    r_ff = cfg.K_ff * kappa_hat * u
    r_th = -cfg.K_theta * theta
    return float(r_ff + r_los + r_th)


def _smoke() -> None:
    cfg = LOSConfig(Delta=2.0, K_p=0.5, d_star=5.0)
    r0 = compute_r_star(d=5.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r0) < 1e-12, r0
    r1 = compute_r_star(d=6.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r1 - (-0.5 / np.sqrt(5.0))) < 1e-12, r1
    r_inf = compute_r_star(d=1e6, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    assert abs(r_inf - (-0.5)) < 1e-6, r_inf
    r_ff = compute_r_star(d=5.0, theta=0.0, kappa_hat=0.1, u=2.0, cfg=cfg)
    assert abs(r_ff - 0.2) < 1e-12, r_ff
    print("[los_controller smoke] OK")


if __name__ == "__main__":
    _smoke()
