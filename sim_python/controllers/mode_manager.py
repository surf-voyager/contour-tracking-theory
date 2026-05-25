"""Mode-management FSM for {T, L_N, L_G, R}.

Implements Lemma 2's mode-switching logic and the two closed-form
dwell budgets: the noise-driven dwell T*_N and the geometric-occlusion
dwell T*_G.

States
------
T   : Tracking (nominal closed-loop LOS + κ̂ feed-forward)
L_N : Lost-Noise (χ² gate triggered — noise-driven loss; dwell ≤ T*_N)
L_G : Lost-Geometry (visibility flag flipped 0 — wall exited FOV cone;
       dwell ≤ T*_G; nominal-side curvature feed-forward continues)
R   : Re-acquire (search/spiral after dwell budget exceeded)

Transitions
-----------
T → L_N   : χ² innovation statistic η.T S⁻¹ η > χ²_α(ν=2) with α=0.01
T → L_G   : visibility predicate vis_t = 0 (provided by sensing layer)
L_N → R   : dwell time in L_N exceeds T*_N
L_G → R   : dwell time in L_G exceeds T*_G
L_*  → T  : new valid measurement arrives + χ² OK + vis = 1
R   → T   : re-acquire reports SUCCESS (set by ReacquirePlanner)
R   → L_* : re-acquire continues without success up to ``budget_R``

Closed-form dwells
------------------
Noise-driven dwell T*_N (boxed closed form):

    T*_N = sqrt( 2 (d* - d_min) β_N / (σ_κ̂ ū²) )
    β_N = 1 / sqrt(2 · ln(1/δ_L))   (default δ_L = 0.01)

Geometric-occlusion dwell T*_G (worst-case onset, b = 0):

    T*_G = (1/ū) · sqrt(w_FOV / κ'_max)

These are *static* (per-config) numbers; the FSM computes them on
construction from the supplied ModeConfig.  The dwell-time clock is
reset on each entry into L_*.

χ² gate
-------
Uses scipy.stats.chi2.ppf(1 - α, df=ν) for the threshold; α=0.01 by
default.  Caller supplies the innovation 2-vector η (e.g. lateral
residual + curvature-fit residual) and the covariance S; this module
just computes the statistic and compares.

Warm-up gate
------------
A 1-second post-init warm-up disables the χ² gate (initial transient
can otherwise spuriously trigger Lost).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import chi2


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Default χ² significance level (2-tailed).
_DEFAULT_ALPHA: float = 0.01

# Default lost-budget probability δ_L (used in β_N(δ_L)).
_DEFAULT_DELTA_L: float = 0.01

# Default warm-up [s] during which the χ² gate is disabled.
_DEFAULT_WARMUP_S: float = 1.0


# --------------------------------------------------------------------------- #
# Mode enum + config                                                          #
# --------------------------------------------------------------------------- #


class Mode(Enum):
    """FSM states."""

    T = "T"
    L_N = "L_N"
    L_G = "L_G"
    R = "R"


@dataclass(frozen=True)
class ModeConfig:
    """Mode-manager configuration.

    Parameters
    ----------
    d_star : float
        Standoff distance set-point [m].
    d_min : float
        Minimum allowable distance to wall [m] (safety margin).
        d_star > d_min required.
    sigma_kappa_hat : float
        Standard deviation of LS curvature estimate [m⁻¹] used in
        T*_N.  Use ``CurvatureEstimator.sigma_kappa_hat(σ_η)`` to
        compute from primary noise σ_η.
    u_min : float
        Lower bound on surge speed ū [m/s] used in T*_N and T*_G.
    w_fov : float
        Field-of-view full width [rad] used in T*_G.
    kappa_dot_max : float
        Max wall curvature rate |dκ/ds| [m⁻²] used in T*_G.
    delta_L : float, default 0.01
        Lost-budget probability used in β_N(δ_L).
    alpha_chi2 : float, default 0.01
        χ² gate significance level.
    chi2_dof : int, default 2
        Innovation vector dimensionality (degrees of freedom).
    warmup_s : float, default 1.0
        Initial period [s] during which χ² gate is disabled.
    """

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
            raise ValueError(f"chi2_dof must be ≥ 1; got {self.chi2_dof}")
        if not (0 < self.delta_L < 1):
            raise ValueError(f"delta_L must be in (0,1); got {self.delta_L}")
        if not (0 < self.alpha_chi2 < 1):
            raise ValueError(f"alpha_chi2 must be in (0,1); got {self.alpha_chi2}")


# --------------------------------------------------------------------------- #
# Closed-form dwells                                                          #
# --------------------------------------------------------------------------- #


def beta_N(delta_L: float = _DEFAULT_DELTA_L) -> float:
    """β_N(δ_L) = 1 / sqrt(2 · ln(1/δ_L)).

    Comes from the Chernoff tail bound on the lateral error: set
    exp(-(d*-d_min)²/(2σ_e²)) = δ_L and solve σ_e = (d*-d_min) · β_N.
    """
    if not (0.0 < delta_L < 1.0):
        raise ValueError(f"delta_L must be in (0,1); got {delta_L!r}")
    return float(1.0 / np.sqrt(2.0 * np.log(1.0 / delta_L)))


def compute_T_star_N(
    d_star: float,
    d_min: float,
    sigma_kappa_hat: float,
    u_min: float,
    delta_L: float = _DEFAULT_DELTA_L,
) -> float:
    """Lost-N maximum dwell time T*_N [s].

    Square-root closed form:

        T*_N = sqrt( 2 · (d* - d_min) · β_N(δ_L) / (σ_κ̂ · ū²) )

    Reference algebra: d*-d_min=4, σ_κ̂=3e-3, ū=0.5, δ_L=0.01 ⇒
    β_N=0.3296 ⇒ T*_N = sqrt(2.6368 / 7.5e-4) ≈ 59.3 s.
    """
    if d_star <= d_min:
        raise ValueError(f"d_star ({d_star}) must be > d_min ({d_min})")
    if not (sigma_kappa_hat > 0 and np.isfinite(sigma_kappa_hat)):
        raise ValueError(
            f"sigma_kappa_hat must be > 0 and finite; got {sigma_kappa_hat!r}"
        )
    if not (u_min > 0 and np.isfinite(u_min)):
        raise ValueError(f"u_min must be > 0 and finite; got {u_min!r}")
    margin = d_star - d_min
    beta = beta_N(delta_L)
    return float(np.sqrt(2.0 * margin * beta / (sigma_kappa_hat * u_min * u_min)))


def compute_T_star_G(
    u_min: float,
    w_fov: float,
    kappa_dot_max: float,
) -> float:
    """Lost-G maximum dwell time T*_G [s].

    Geometric-occlusion dwell (worst-case onset, b=0):

        T*_G = (1/ū) · sqrt(w_FOV / κ'_max)

    Reference algebra: ū=2, w_FOV=π/2, then
        κ'_max = 1e-3 ⇒ T*_G = 0.5 · sqrt(1570.8) ≈ 19.8 s
        κ'_max = 1e-2 ⇒ T*_G = 0.5 · sqrt(157.08) ≈ 6.27 s
        κ'_max = 1e-1 ⇒ T*_G = 0.5 · sqrt(15.708) ≈ 1.98 s
    """
    if not (u_min > 0 and np.isfinite(u_min)):
        raise ValueError(f"u_min must be > 0 and finite; got {u_min!r}")
    if not (w_fov > 0 and np.isfinite(w_fov)):
        raise ValueError(f"w_fov must be > 0 and finite; got {w_fov!r}")
    if not (kappa_dot_max > 0 and np.isfinite(kappa_dot_max)):
        raise ValueError(
            f"kappa_dot_max must be > 0 and finite; got {kappa_dot_max!r}"
        )
    return float((1.0 / u_min) * np.sqrt(w_fov / kappa_dot_max))


def chi2_threshold(alpha: float = _DEFAULT_ALPHA, dof: int = 2) -> float:
    """Upper-tail χ² threshold χ²_α(dof) for the innovation gate.

    Reject (declare Lost-N) when statistic > threshold.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1); got {alpha!r}")
    if dof < 1:
        raise ValueError(f"dof must be ≥ 1; got {dof!r}")
    return float(chi2.ppf(1.0 - alpha, df=dof))


# --------------------------------------------------------------------------- #
# Mode-manager FSM                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class ModeManager:
    """4-state FSM controlling {T, L_N, L_G, R}.

    Attributes
    ----------
    cfg : ModeConfig
        Static thresholds + dwells.

    Internal state
    --------------
    mode : Mode
        Current state.
    t_now : float
        Current simulation time [s].
    t_enter : float
        Time stamp [s] of most recent entry into current state.
    """

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
            d_star=self.cfg.d_star,
            d_min=self.cfg.d_min,
            sigma_kappa_hat=self.cfg.sigma_kappa_hat,
            u_min=self.cfg.u_min,
            delta_L=self.cfg.delta_L,
        )
        self._T_star_G = compute_T_star_G(
            u_min=self.cfg.u_min,
            w_fov=self.cfg.w_fov,
            kappa_dot_max=self.cfg.kappa_dot_max,
        )
        self._chi2_thresh = chi2_threshold(
            alpha=self.cfg.alpha_chi2, dof=self.cfg.chi2_dof
        )
        self.mode = Mode.T
        self.t_now = 0.0
        self.t_enter = 0.0
        self.history = [(0.0, Mode.T)]

    # ------------------------------------------------------------------- #
    # Public dwell accessors                                              #
    # ------------------------------------------------------------------- #

    @property
    def T_star_N(self) -> float:
        """Pre-computed T*_N [s]."""
        return self._T_star_N

    @property
    def T_star_G(self) -> float:
        """Pre-computed T*_G [s]."""
        return self._T_star_G

    @property
    def T_star(self) -> float:
        """Joint dwell = max(T*_N, T*_G)."""
        return float(max(self._T_star_N, self._T_star_G))

    @property
    def chi2_threshold_value(self) -> float:
        """Pre-computed χ² threshold."""
        return self._chi2_thresh

    @property
    def dwell_in_current(self) -> float:
        """Time [s] spent in current mode since most recent entry."""
        return float(self.t_now - self.t_enter)

    # ------------------------------------------------------------------- #
    # χ² statistic helper                                                 #
    # ------------------------------------------------------------------- #

    @staticmethod
    def chi2_statistic(eta: np.ndarray, S: np.ndarray) -> float:
        """η.T S⁻¹ η innovation statistic.

        Parameters
        ----------
        eta : ndarray, shape (n,)
            Innovation vector.
        S : ndarray, shape (n, n)
            Innovation covariance (symmetric positive-definite).
        """
        eta = np.asarray(eta, dtype=np.float64).reshape(-1)
        S = np.asarray(S, dtype=np.float64)
        if S.shape != (eta.size, eta.size):
            raise ValueError(f"S shape {S.shape} incompatible with eta {eta.shape}")
        try:
            x = np.linalg.solve(S, eta)
        except np.linalg.LinAlgError as exc:
            raise ValueError("S must be invertible (positive definite)") from exc
        return float(eta @ x)

    # ------------------------------------------------------------------- #
    # Step                                                                #
    # ------------------------------------------------------------------- #

    def step(
        self,
        dt: float,
        chi2_stat: Optional[float] = None,
        visibility: bool = True,
        reacquire_success: bool = False,
    ) -> Mode:
        """Advance time by dt and process triggers; return new mode.

        Parameters
        ----------
        dt : float
            Step size [s], > 0.
        chi2_stat : float, optional
            Latest innovation statistic.  If None, no χ²-driven
            transition is considered this step.
        visibility : bool
            True iff wall is currently within FOV.  Lost-G is
            triggered by transition from True → False.
        reacquire_success : bool
            Set by the ReacquirePlanner when it captures a fresh
            track; triggers R → T.

        Returns
        -------
        Mode
            New (or unchanged) mode.
        """
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt must be > 0; got {dt!r}")
        prev_mode = self.mode
        self.t_now += float(dt)

        # Warm-up: χ² gate disabled for the first warmup_s seconds.
        gate_on = self.t_now >= self.cfg.warmup_s

        new_mode = self.mode

        if self.mode == Mode.T:
            # Visibility loss takes priority over χ² gate (Lost-G is an
            # instantaneous geometric event).
            if not visibility:
                new_mode = Mode.L_G
            elif (
                gate_on
                and chi2_stat is not None
                and chi2_stat > self._chi2_thresh
            ):
                new_mode = Mode.L_N

        elif self.mode == Mode.L_N:
            if self.dwell_in_current > self._T_star_N:
                new_mode = Mode.R
            elif (
                visibility
                and chi2_stat is not None
                and chi2_stat <= self._chi2_thresh
            ):
                # Recovered nominally.
                new_mode = Mode.T

        elif self.mode == Mode.L_G:
            if self.dwell_in_current > self._T_star_G:
                new_mode = Mode.R
            elif visibility:
                # Wall re-entered FOV without needing a search.
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
        """Reset to T at t=0."""
        self.mode = Mode.T
        self.t_now = 0.0
        self.t_enter = 0.0
        self.history = [(0.0, Mode.T)]


# --------------------------------------------------------------------------- #
# __main__ smoke                                                              #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    # Two reference examples drawn directly from the closed forms:
    #   - T*_N example: σ_κ̂=3e-3, ū=0.5, d*=5, d_min=1 → 59.3 s
    #   - T*_G example: ū=2, w_FOV=π/2, κ'_max=1e-3 → 19.8 s
    # The two examples don't share ū, so we test each in isolation.

    # T*_N example: ū=0.5
    cfg_N = ModeConfig(
        d_star=5.0, d_min=1.0, sigma_kappa_hat=3.0e-3, u_min=0.5,
        w_fov=np.pi / 2, kappa_dot_max=1e-3,
    )
    mm_N = ModeManager(cfg=cfg_N)
    print(f"[smoke] T*_N(ū=0.5) = {mm_N.T_star_N:.2f} s (expect ~59.3)")
    assert abs(mm_N.T_star_N - 59.3) < 0.5

    # T*_G example: ū=2, w_FOV=π/2, three κ'_max
    for kpm, expected in ((1e-3, 19.8), (1e-2, 6.27), (1e-1, 1.98)):
        T = compute_T_star_G(u_min=2.0, w_fov=np.pi / 2, kappa_dot_max=kpm)
        print(f"[smoke] T*_G(ū=2, κ'={kpm:.0e}) = {T:.3f} s (expect ~{expected})")
        assert abs(T - expected) < 0.05, f"{T} vs {expected}"

    print(f"[smoke] χ²_threshold (α=0.01, ν=2) = {mm_N.chi2_threshold_value:.3f} (expect ~9.21)")
    print(f"[smoke] β_N (δ_L=0.01) = {beta_N(0.01):.4f} (expect ~0.3296)")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke()
