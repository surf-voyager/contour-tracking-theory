"""LOS guidance with curvature feed-forward (Caharija 2016 saturated form).

Implements the Serret–Frenet line-of-sight guidance law with curvature
feed-forward:

    r^*(t) = \\hat\\kappa(t) \\cdot \\hat u(t)  +  r^*_{LOS}(d - d^*, θ; Δ)

The LOS term has two equivalent saturated forms (Caharija 2016):

    sin-style   :  r^*_{LOS} = - K_p (d - d^*) / sqrt((d - d^*)^2 + Δ^2)
    atan2-style :  r^*_{LOS} = - K_p · atan2(d - d^*, Δ)

Both saturate as |d - d^*| → ∞: sin-style asymptotes to ±K_p, atan2-style
to ±K_p·π/2.  The sin-style is the standard "saturated LOS" of Caharija
2016 (Fossen Handbook ch.10 §10.4.2) and is the default; atan2-style is
exposed as an alternative for users who prefer the explicit angle form.

The feed-forward κ̂·û component anticipates the curvature of the wall so
that LOS only has to correct the lateral-distance error rather than also
having to drive the steady-state turn.  This is required for steady-state
zero error on a circular wall.

A simple θ-correction term is also included optionally (proportional to
heading misalignment); set ``K_theta = 0`` (default) to recover the
classical look-ahead distance Δ-only behaviour.

References
----------
- The saturated-LOS Lyapunov analysis gives ω_LOS ≈ ū/Δ for stability.
- The curvature feed-forward analysis gives the κ·û term.
- Caharija et al. 2016 IEEE TCST, "Integral LOS for path following of AUVs"
  — saturated sin form

State convention:

- d        : signed lateral distance to wall [m]
- θ        : heading misalignment ψ - γ_p [rad]  (γ_p = wall tangent angle)
- κ̂        : current curvature estimate [m⁻¹]   (may include side sign)
- u        : current surge speed [m/s]
- d*       : standoff distance (set-point) [m]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LOSStyle = Literal["sin", "atan2"]


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LOSConfig:
    """LOS + feed-forward parameters.

    Attributes
    ----------
    Delta : float
        Look-ahead distance [m].  ω_LOS ≈ ū/Δ so smaller Δ ⇒ more
        aggressive correction.  Typical 1–5 m for AUVs.
    K_p : float
        Lateral-error gain [rad/s] — the saturated LOS asymptote.
        ``r^*_{LOS}`` saturates at ±K_p (sin-style).
    K_ff : float, default 1.0
        Feed-forward gain on κ̂·û; nominally 1.0 (theory).  Exposed
        for sensitivity / robustness studies.
    K_theta : float, default 0.0
        Optional proportional heading-error gain.  When > 0, adds
        ``-K_theta · θ`` to the LOS term; default 0 keeps the classical
        Δ-only look-ahead behaviour.
    style : "sin" or "atan2", default "sin"
        Saturated LOS form (see module docstring).
    d_star : float, default 0.0
        Standoff set-point [m].  Convenience field so ``compute_r_star``
        does not need a separate d_star argument.
    """

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
            raise ValueError(f"K_theta must be ≥ 0 and finite; got {self.K_theta!r}")
        if self.style not in ("sin", "atan2"):
            raise ValueError(f"style must be 'sin' or 'atan2'; got {self.style!r}")


# --------------------------------------------------------------------------- #
# Core                                                                        #
# --------------------------------------------------------------------------- #


def compute_r_star(
    d: float,
    theta: float,
    kappa_hat: float,
    u: float,
    cfg: LOSConfig,
) -> float:
    """Commanded yaw rate r^* given current error state.

    Implements:

        r^* = K_ff · κ̂ · u  +  r^*_{LOS}(d - d^*, θ; Δ)  +  (- K_θ · θ)

    where r^*_{LOS} is the Caharija 2016 saturated LOS:

        sin-style   : - K_p (d - d^*) / sqrt((d - d^*)^2 + Δ^2)
        atan2-style : - K_p · atan2(d - d^*, Δ)

    Parameters
    ----------
    d : float
        Current lateral distance to wall [m].
    theta : float
        Heading misalignment ψ - γ_p [rad].  Only used if K_theta > 0.
    kappa_hat : float
        Curvature estimate [m⁻¹] (signed: + LEFT turn, - RIGHT turn,
        matching scenarios.wall_generator convention).
    u : float
        Current surge speed [m/s].  Multiplied with κ̂ for the
        feed-forward turn rate.
    cfg : LOSConfig
        Controller config.

    Returns
    -------
    r_cmd : float
        Commanded yaw rate [rad/s].  NB: hull-side saturation (|r| ≤
        r_max, |ṙ| ≤ ṙ_max) is applied downstream by the hull model
        (sim_python.models.torpedo_kinematic_2d).

    Notes
    -----
    - The sign convention: a positive (d - d^*) means the AUV is
      further from the wall than required ⇒ needs to turn toward the
      wall.  With side="L" the wall is to our LEFT, so to reduce d we
      turn left ⇒ r should be positive.  The minus sign in the LOS
      term gives r > 0 when (d - d^*) > 0; combined with side="L" sign
      convention of the caller (who passes the signed d), this is
      consistent.  For side="R", the caller should pass d with the
      opposite sign so the same controller works.
    - There is no integrator in this signature — the analysis uses a
      pure proportional sat-LOS for the Lyapunov certificate.  An
      integral-LOS variant can be added later for the full Caharija
      2016 I-LOS as a separate function.
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
        # Caharija 2016 saturated sin-style.
        r_los = -cfg.K_p * e / np.sqrt(e * e + cfg.Delta * cfg.Delta)
    else:  # atan2-style
        r_los = -cfg.K_p * float(np.arctan2(e, cfg.Delta))

    r_ff = cfg.K_ff * kappa_hat * u
    r_th = -cfg.K_theta * theta
    return float(r_ff + r_los + r_th)


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    cfg = LOSConfig(Delta=2.0, K_p=0.5, d_star=5.0)
    # On-track: d = d* = 5, θ = 0, κ̂ = 0, u = 1 ⇒ r* = 0.
    r0 = compute_r_star(d=5.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    print(f"[smoke] on-track r*={r0:.6e} (expected 0)")
    assert abs(r0) < 1e-12

    # d=6 (too far), κ̂=0 ⇒ r* < 0 (sin-style):
    # e = (d - d*) = 1; r*_LOS = -K_p · 1 / sqrt(1² + 2²) = -0.5/sqrt(5).
    r1 = compute_r_star(d=6.0, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    expected_r1 = -0.5 / np.sqrt(5.0)
    print(f"[smoke] d=6, r*={r1:.4f} (expected {expected_r1:.4f})")
    assert abs(r1 - expected_r1) < 1e-12

    # Saturation: d=1e6, expect r* → -K_p.
    r_inf = compute_r_star(d=1e6, theta=0.0, kappa_hat=0.0, u=1.0, cfg=cfg)
    print(f"[smoke] d=1e6, r*={r_inf:.4f} (expected → -0.5)")
    assert abs(r_inf - (-0.5)) < 1e-6

    # Feed-forward: on-track + κ̂=0.1, u=2 ⇒ r* = 0.2.
    r_ff = compute_r_star(d=5.0, theta=0.0, kappa_hat=0.1, u=2.0, cfg=cfg)
    print(f"[smoke] κ̂·u FF, r*={r_ff:.4f} (expected 0.2)")
    assert abs(r_ff - 0.2) < 1e-12

    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
