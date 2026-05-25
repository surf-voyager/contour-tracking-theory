"""Mode-management FSM {T, L_N, L_G, R} (Layer-2, no sim_python import).

Closed-form dwells and the χ² gate are identical to the Layer-1 original:

    T*_N = sqrt( 2 (d* - d_min) β_N / (σ_κ̂ ū²) ),  β_N = 1/sqrt(2 ln(1/δ_L))
    T*_G = (1/ū) · sqrt(w_FOV / κ'_max)
    χ² gate: reject (→ L_N) when η.T S⁻¹ η > χ²_α(ν)

T*_N is the noise-driven dwell (how long the controller may coast before standoff
uncertainty risks collision); T*_G is the geometric guaranteed-loss time (how long
a wall may stay occluded out of the FOV before re-acquisition is needed). The
driver supplies (dt, chi2_stat, visibility, reacquire_success) each tick.

χ² threshold uses scipy.stats.chi2 if available; otherwise a hard-coded table for
α=0.01 (ν=1..4) so the FSM still runs on a minimal env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

try:
    from scipy.stats import chi2 as _scipy_chi2
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback path
    _scipy_chi2 = None
    _HAVE_SCIPY = False

# χ²_{0.01}(ν) table (upper-tail), fallback when scipy is absent.
_CHI2_001_TABLE = {1: 6.634896601, 2: 9.210340372, 3: 11.344866730, 4: 13.276704135}

_DEFAULT_ALPHA: float = 0.01
_DEFAULT_DELTA_L: float = 0.01
_DEFAULT_WARMUP_S: float = 1.0


class Mode(Enum):
    T = "T"
    L_N = "L_N"
    L_G = "L_G"
    R = "R"


@dataclass(frozen=True)
class ModeConfig:
    """Mode-manager configuration."""

    d_star: float
    d_min: float
    sigma_kappa_hat: float
    u_min: float
    w_fov: float
    kappa_dot_max: float
    delta_L: float = _DEFAULT_DELTA_L
    alpha_chi2: float = _DEFAULT_ALPHA
    chi2_dof: int = 2
    warmup_s: float = _DEFAULT_WARMUP_S

    def __post_init__(self) -> None:
        for name in (
            "d_star", "d_min", "sigma_kappa_hat", "u_min", "w_fov",
            "kappa_dot_max", "delta_L", "alpha_chi2", "warmup_s",
        ):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"ModeConfig.{name} must be > 0 and finite; got {v!r}")
        if self.d_star <= self.d_min:
            raise ValueError(
                f"ModeConfig.d_star ({self.d_star}) must be > d_min ({self.d_min})"
            )
        if self.chi2_dof < 1:
            raise ValueError(f"chi2_dof must be >= 1; got {self.chi2_dof}")
        if not (0 < self.delta_L < 1):
            raise ValueError(f"delta_L must be in (0,1); got {self.delta_L}")
        if not (0 < self.alpha_chi2 < 1):
            raise ValueError(f"alpha_chi2 must be in (0,1); got {self.alpha_chi2}")


def beta_N(delta_L: float = _DEFAULT_DELTA_L) -> float:
    """β_N(δ_L) = 1 / sqrt(2 ln(1/δ_L))."""
    if not (0.0 < delta_L < 1.0):
        raise ValueError(f"delta_L must be in (0,1); got {delta_L!r}")
    return float(1.0 / np.sqrt(2.0 * np.log(1.0 / delta_L)))


def compute_T_star_N(
    d_star: float, d_min: float, sigma_kappa_hat: float, u_min: float,
    delta_L: float = _DEFAULT_DELTA_L,
) -> float:
    """T*_N [s] = sqrt(2 (d*-d_min) β_N / (σ_κ̂ ū²)) (noise-driven dwell)."""
    if d_star <= d_min:
        raise ValueError(f"d_star ({d_star}) must be > d_min ({d_min})")
    if not (sigma_kappa_hat > 0 and np.isfinite(sigma_kappa_hat)):
        raise ValueError(f"sigma_kappa_hat must be > 0; got {sigma_kappa_hat!r}")
    if not (u_min > 0 and np.isfinite(u_min)):
        raise ValueError(f"u_min must be > 0; got {u_min!r}")
    margin = d_star - d_min
    beta = beta_N(delta_L)
    return float(np.sqrt(2.0 * margin * beta / (sigma_kappa_hat * u_min * u_min)))


def compute_T_star_G(u_min: float, w_fov: float, kappa_dot_max: float) -> float:
    """T*_G [s] = (1/ū) sqrt(w_FOV / κ'_max) (geometric guaranteed-loss time)."""
    if not (u_min > 0 and np.isfinite(u_min)):
        raise ValueError(f"u_min must be > 0; got {u_min!r}")
    if not (w_fov > 0 and np.isfinite(w_fov)):
        raise ValueError(f"w_fov must be > 0; got {w_fov!r}")
    if not (kappa_dot_max > 0 and np.isfinite(kappa_dot_max)):
        raise ValueError(f"kappa_dot_max must be > 0; got {kappa_dot_max!r}")
    return float((1.0 / u_min) * np.sqrt(w_fov / kappa_dot_max))


def chi2_threshold(alpha: float = _DEFAULT_ALPHA, dof: int = 2) -> float:
    """Upper-tail χ² threshold χ²_α(dof)."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1); got {alpha!r}")
    if dof < 1:
        raise ValueError(f"dof must be >= 1; got {dof!r}")
    if _HAVE_SCIPY:
        return float(_scipy_chi2.ppf(1.0 - alpha, df=dof))
    if abs(alpha - 0.01) < 1e-9 and dof in _CHI2_001_TABLE:
        return _CHI2_001_TABLE[dof]
    raise RuntimeError(
        f"scipy unavailable and no fallback χ² value for alpha={alpha}, dof={dof}"
    )


@dataclass
class ModeManager:
    """4-state FSM controlling {T, L_N, L_G, R}."""

    cfg: ModeConfig
    mode: Mode = field(default=Mode.T, init=False)
    t_now: float = field(default=0.0, init=False)
    t_enter: float = field(default=0.0, init=False)
    _T_star_N: float = field(default=0.0, init=False, repr=False)
    _T_star_G: float = field(default=0.0, init=False, repr=False)
    _chi2_thresh: float = field(default=0.0, init=False, repr=False)
    history: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._T_star_N = compute_T_star_N(
            d_star=self.cfg.d_star, d_min=self.cfg.d_min,
            sigma_kappa_hat=self.cfg.sigma_kappa_hat, u_min=self.cfg.u_min,
            delta_L=self.cfg.delta_L,
        )
        self._T_star_G = compute_T_star_G(
            u_min=self.cfg.u_min, w_fov=self.cfg.w_fov,
            kappa_dot_max=self.cfg.kappa_dot_max,
        )
        self._chi2_thresh = chi2_threshold(
            alpha=self.cfg.alpha_chi2, dof=self.cfg.chi2_dof
        )
        self.mode = Mode.T
        self.t_now = 0.0
        self.t_enter = 0.0
        self.history = [(0.0, Mode.T)]

    @property
    def T_star_N(self) -> float:
        return self._T_star_N

    @property
    def T_star_G(self) -> float:
        return self._T_star_G

    @property
    def T_star(self) -> float:
        return float(max(self._T_star_N, self._T_star_G))

    @property
    def chi2_threshold_value(self) -> float:
        return self._chi2_thresh

    @property
    def dwell_in_current(self) -> float:
        return float(self.t_now - self.t_enter)

    @staticmethod
    def chi2_statistic(eta: np.ndarray, S: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=np.float64).reshape(-1)
        S = np.asarray(S, dtype=np.float64)
        if S.shape != (eta.size, eta.size):
            raise ValueError(f"S shape {S.shape} incompatible with eta {eta.shape}")
        try:
            x = np.linalg.solve(S, eta)
        except np.linalg.LinAlgError as exc:
            raise ValueError("S must be invertible (positive definite)") from exc
        return float(eta @ x)

    def step(
        self,
        dt: float,
        chi2_stat: Optional[float] = None,
        visibility: bool = True,
        reacquire_success: bool = False,
    ) -> Mode:
        """Advance time by dt and process triggers; return new mode."""
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt must be > 0; got {dt!r}")
        prev_mode = self.mode
        self.t_now += float(dt)
        gate_on = self.t_now >= self.cfg.warmup_s
        new_mode = self.mode

        if self.mode == Mode.T:
            if not visibility:
                new_mode = Mode.L_G
            elif gate_on and chi2_stat is not None and chi2_stat > self._chi2_thresh:
                new_mode = Mode.L_N
        elif self.mode == Mode.L_N:
            if self.dwell_in_current > self._T_star_N:
                new_mode = Mode.R
            elif visibility and chi2_stat is not None and chi2_stat <= self._chi2_thresh:
                new_mode = Mode.T
        elif self.mode == Mode.L_G:
            if self.dwell_in_current > self._T_star_G:
                new_mode = Mode.R
            elif visibility:
                new_mode = Mode.T
        elif self.mode == Mode.R:
            if reacquire_success:
                new_mode = Mode.T

        if new_mode != prev_mode:
            self.mode = new_mode
            self.t_enter = self.t_now
            self.history.append((self.t_now, new_mode))
        return self.mode

    def reset(self) -> None:
        self.mode = Mode.T
        self.t_now = 0.0
        self.t_enter = 0.0
        self.history = [(0.0, Mode.T)]


def _smoke() -> None:
    cfg_N = ModeConfig(
        d_star=5.0, d_min=1.0, sigma_kappa_hat=3.0e-3, u_min=0.5,
        w_fov=np.pi / 2, kappa_dot_max=1e-3,
    )
    mm_N = ModeManager(cfg=cfg_N)
    assert abs(mm_N.T_star_N - 59.3) < 0.5, mm_N.T_star_N
    for kpm, expected in ((1e-3, 19.8), (1e-2, 6.27), (1e-1, 1.98)):
        T = compute_T_star_G(u_min=2.0, w_fov=np.pi / 2, kappa_dot_max=kpm)
        assert abs(T - expected) < 0.05, (T, expected)
    assert abs(mm_N.chi2_threshold_value - 9.210340372) < 1e-3, mm_N.chi2_threshold_value
    assert abs(beta_N(0.01) - 0.3296) < 1e-3, beta_N(0.01)
    print(f"[mode_manager smoke] OK (scipy={_HAVE_SCIPY})")


if __name__ == "__main__":
    _smoke()
