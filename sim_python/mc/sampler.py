"""Latin-Hypercube sampler + adaptive-refinement helpers for the MC framework.

Public API
----------
- ``latin_hypercube_sample(ranges, N, seed) -> list[dict]``
  Latin-Hypercube sample N configurations over a dict of named axes.
  Each axis is specified as ``(lo, hi, "log"|"linear")``.  Log-scale
  axes are sampled uniformly in log space then exponentiated, so they
  cover the lo–hi interval log-uniformly.

- ``adaptive_boundary_refine(prior_results, N_refine, residual_window) -> list[dict]``
  Identify configurations whose (C1, C2) residuals fall inside the
  window (default [-0.1, 0.1]) and sample N_refine perturbed neighbours.
  Used to densify the sample near the feasibility boundary.

Determinism
-----------
All randomness flows through ``numpy.random.Generator(seed)`` and the
``scipy.stats.qmc.LatinHypercube(seed=seed)`` engine.  Two invocations
with the same (ranges, N, seed) produce identical lists.

References
----------
- scipy.stats.qmc.LatinHypercube — backend
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc

# --------------------------------------------------------------------------- #
# Allowed axis scales                                                         #
# --------------------------------------------------------------------------- #

_SCALES: Tuple[str, ...] = ("linear", "log")

AxisRange = Tuple[float, float, str]  # (lo, hi, scale)


# --------------------------------------------------------------------------- #
# Latin-hypercube sample                                                      #
# --------------------------------------------------------------------------- #


def latin_hypercube_sample(
    ranges: Dict[str, AxisRange],
    N: int,
    seed: int,
) -> List[Dict[str, float]]:
    """Latin-hypercube sample over named parameter ranges.

    Parameters
    ----------
    ranges : dict
        ``{param_name: (lo, hi, scale)}`` with ``scale in {"linear", "log"}``.
        ``lo < hi`` strictly.  For log scale, both endpoints must be > 0.
    N : int
        Number of samples (> 0).
    seed : int
        Seed for ``scipy.stats.qmc.LatinHypercube``.  Two calls with the
        same (ranges, N, seed) produce identical outputs.

    Returns
    -------
    list[dict]
        Length-N list of ``{param_name: float}`` dicts, in axis-order
        matching the input dict's insertion order.

    Raises
    ------
    ValueError
        On empty/invalid ranges, non-positive N or bad scale spec.
    """
    if not isinstance(ranges, dict) or len(ranges) == 0:
        raise ValueError("ranges must be a non-empty dict of {name: (lo, hi, scale)}")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError(f"N must be a positive int; got {N!r}")
    if seed is None or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an int; got {seed!r}")

    # Validate every axis up-front, build lo/hi/scale arrays.
    names: List[str] = list(ranges.keys())
    los: List[float] = []
    his: List[float] = []
    scales: List[str] = []
    for name in names:
        spec = ranges[name]
        if not (isinstance(spec, (tuple, list)) and len(spec) == 3):
            raise ValueError(
                f"ranges[{name!r}] must be (lo, hi, scale); got {spec!r}"
            )
        lo, hi, scale = spec
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"ranges[{name!r}] endpoints must be finite")
        if not (lo < hi):
            raise ValueError(
                f"ranges[{name!r}]: lo < hi required; got lo={lo}, hi={hi}"
            )
        if scale not in _SCALES:
            raise ValueError(
                f"ranges[{name!r}]: scale must be one of {_SCALES}; got {scale!r}"
            )
        if scale == "log" and lo <= 0:
            raise ValueError(
                f"ranges[{name!r}]: log scale requires lo > 0; got {lo}"
            )
        los.append(float(lo))
        his.append(float(hi))
        scales.append(scale)

    d = len(names)
    # Deterministic LHS engine seeded with `seed`.  scipy's LHS returns
    # values in [0, 1)^d which we then scale per axis.
    engine = qmc.LatinHypercube(d=d, seed=int(seed))
    unit = engine.random(n=int(N))  # shape (N, d)

    out: List[Dict[str, float]] = []
    for i in range(int(N)):
        row: Dict[str, float] = {}
        for j, name in enumerate(names):
            u = float(unit[i, j])
            lo = los[j]
            hi = his[j]
            if scales[j] == "linear":
                row[name] = lo + (hi - lo) * u
            else:  # log
                log_lo = np.log(lo)
                log_hi = np.log(hi)
                row[name] = float(np.exp(log_lo + (log_hi - log_lo) * u))
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Adaptive boundary refinement                                                #
# --------------------------------------------------------------------------- #


def adaptive_boundary_refine(
    prior_results,
    N_refine: int,
    residual_window: Tuple[float, float] = (-0.1, 0.1),
    *,
    residual_cols: Tuple[str, ...] = ("c1_residual", "c2_residual"),
    perturb_frac: float = 0.05,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Sample N_refine perturbed configurations around prior C1/C2 boundary.

    Parameters
    ----------
    prior_results : pandas.DataFrame
        Per-run summaries from a prior batch.  Must contain at least
        the columns in ``residual_cols`` plus the parameter columns to
        perturb.
    N_refine : int
        Number of refined samples to draw.
    residual_window : (lo, hi), default (-0.1, 0.1)
        Configurations are 'on the boundary' when at least one of the
        residual columns lies in ``[lo, hi]``.
    residual_cols : tuple of str, default ("c1_residual", "c2_residual")
        Column names treated as boundary residuals.
    perturb_frac : float, default 0.05
        Standard deviation of the multiplicative Gaussian perturbation
        applied to each parameter (5 % per axis).
    seed : int, default 0
        RNG seed (numpy.random.Generator) for the perturbations.

    Returns
    -------
    list[dict]
        Length-N_refine list of perturbed configurations.

    Notes
    -----
    Stage-03 only exercises the function with a small smoke; Stage-04
    will drive it with the full N=10⁴ → N=10³ refine pipeline.  The
    parameter axes detected here are every non-residual numeric column
    in ``prior_results``.  If no near-boundary rows exist, the function
    falls back to perturbing around the mean of the parameter columns.
    """
    import pandas as pd  # local import to keep top-level light

    if N_refine <= 0:
        raise ValueError(f"N_refine must be > 0; got {N_refine}")
    if not isinstance(prior_results, pd.DataFrame):
        raise TypeError("prior_results must be a pandas DataFrame")
    lo, hi = residual_window
    if not (lo < hi):
        raise ValueError(f"residual_window lo<hi required; got {residual_window}")
    rng = np.random.default_rng(int(seed))

    # Find which parameter columns to perturb: every numeric column
    # that's not a residual.
    param_cols = [
        c for c in prior_results.columns
        if c not in residual_cols
        and pd.api.types.is_numeric_dtype(prior_results[c])
    ]
    if not param_cols:
        raise ValueError(
            "prior_results has no numeric parameter columns to perturb"
        )

    # Select boundary rows (any residual in window).
    if all(c in prior_results.columns for c in residual_cols):
        mask = np.zeros(len(prior_results), dtype=bool)
        for c in residual_cols:
            col = prior_results[c].to_numpy(dtype=float)
            mask |= (col >= lo) & (col <= hi)
        boundary = prior_results.loc[mask, param_cols]
    else:
        boundary = prior_results.iloc[0:0][param_cols]

    if len(boundary) == 0:
        # Fallback: perturb around mean of every parameter column.
        means = prior_results[param_cols].mean(axis=0).to_numpy(dtype=float)
        base = np.tile(means, (N_refine, 1))
    else:
        # Sample with replacement from boundary rows.
        idx = rng.integers(low=0, high=len(boundary), size=N_refine)
        base = boundary.to_numpy(dtype=float)[idx]

    # Multiplicative Gaussian perturbation, but with clamping at 0 so
    # log-scale-friendly axes stay positive.
    noise = rng.normal(loc=0.0, scale=float(perturb_frac), size=base.shape)
    perturbed = base * (1.0 + noise)
    perturbed = np.where(perturbed <= 0.0, base * 0.5, perturbed)

    return [
        {col: float(perturbed[i, j]) for j, col in enumerate(param_cols)}
        for i in range(N_refine)
    ]


# --------------------------------------------------------------------------- #
# Standalone smoke                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sample = latin_hypercube_sample(
        ranges={
            "kappa_max": (0.01, 1.0, "log"),
            "v_star": (0.2, 2.0, "linear"),
            "f_s": (0.5, 20.0, "log"),
            "tau_d": (0.05, 0.5, "linear"),
        },
        N=5,
        seed=42,
    )
    for i, cfg in enumerate(sample):
        print(f"[{i}] {cfg}")
