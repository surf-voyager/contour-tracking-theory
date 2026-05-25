"""Dual-sonar (scanning vs forward) comparison helpers.

Builds a paired-comparison DataFrame and a side-by-side overlay plot
for the scanning-vs-forward sonar batches.

Public API
----------
- ``compare_metric(scanning_df, forward_df, metric, axis, *, n_bins=8) ->
  pd.DataFrame``
    Bin both batches along the same axis, compute the metric's mean and
    std per bin for each batch, return a tidy paired DataFrame:

        axis_bin_center | scanning_mean | scanning_std | scanning_n
                        | forward_mean  | forward_std  | forward_n
                        | abs_diff      | rel_diff_pct

- ``plot_pair_overlay(scanning_df, forward_df, *, axis, metric, ax,
  axis_log=False, **kw)``
    Plot both batches on the same matplotlib axes with distinct markers
    + colors (scanning: blue circles; forward: red squares).  Returns
    the artist handles so callers can attach legends.

Background
----------
- The geometric-occlusion dwell T*_G has the square-root closed form
  T*_G = (1/ū)·sqrt(w_FOV / κ'_max).
- For a scanning (≈360° FOV) sonar the wall never exits the field of
  view, so Lost-G → 0; a forward (narrow-FOV) sonar shows a finite
  T*_G that grows as κ'_max shrinks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover — matplotlib optional at module import
    plt = None  # type: ignore


# --------------------------------------------------------------------------- #
# compare_metric                                                              #
# --------------------------------------------------------------------------- #


def compare_metric(
    scanning_df: pd.DataFrame,
    forward_df: pd.DataFrame,
    metric: str,
    axis: str,
    *,
    n_bins: int = 8,
    axis_log: bool = False,
    axis_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Paired bin-wise comparison of a metric across two batches.

    Parameters
    ----------
    scanning_df, forward_df : DataFrame
        Per-run summaries from the two batches.  Must each contain
        ``axis`` and ``metric`` columns.
    metric : str
        The column to aggregate (mean / std).
    axis : str
        The independent variable (column) used for binning.
    n_bins : int, default 8
        Number of bins.
    axis_log : bool, default False
        If True, bin edges are log-spaced (axis values must be > 0).
    axis_range : (lo, hi), optional
        Bin range.  Defaults to the joint min/max of both batches.

    Returns
    -------
    DataFrame
        One row per bin with columns:
        ``axis_bin_center``, ``scanning_mean``, ``scanning_std``,
        ``scanning_n``, ``forward_mean``, ``forward_std``, ``forward_n``,
        ``abs_diff`` (scanning − forward), ``rel_diff_pct``
        (100·abs_diff/|forward_mean|, NaN if forward_mean ≈ 0).
    """
    for name, df in (("scanning", scanning_df), ("forward", forward_df)):
        if axis not in df.columns:
            raise KeyError(f"{name}_df missing axis column {axis!r}")
        if metric not in df.columns:
            raise KeyError(f"{name}_df missing metric column {metric!r}")

    s_x = scanning_df[axis].to_numpy(dtype=float)
    s_y = scanning_df[metric].to_numpy(dtype=float)
    f_x = forward_df[axis].to_numpy(dtype=float)
    f_y = forward_df[metric].to_numpy(dtype=float)

    if axis_range is None:
        lo = float(np.nanmin(np.concatenate([s_x, f_x])))
        hi = float(np.nanmax(np.concatenate([s_x, f_x])))
    else:
        lo, hi = axis_range

    if axis_log:
        if lo <= 0:
            lo = float(np.nanmin(
                np.concatenate([s_x[s_x > 0], f_x[f_x > 0]])
            ))
        edges = np.geomspace(lo, hi, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    for i in range(n_bins):
        s_mask = (s_x >= edges[i]) & (s_x < edges[i + 1])
        f_mask = (f_x >= edges[i]) & (f_x < edges[i + 1])
        if i == n_bins - 1:
            s_mask |= s_x == edges[-1]
            f_mask |= f_x == edges[-1]
        s_sub = s_y[s_mask]
        f_sub = f_y[f_mask]
        s_mean = float(np.nanmean(s_sub)) if s_sub.size else np.nan
        s_std = float(np.nanstd(s_sub)) if s_sub.size else np.nan
        f_mean = float(np.nanmean(f_sub)) if f_sub.size else np.nan
        f_std = float(np.nanstd(f_sub)) if f_sub.size else np.nan
        abs_diff = (s_mean - f_mean) if np.isfinite(s_mean) and np.isfinite(f_mean) else np.nan
        if np.isfinite(f_mean) and abs(f_mean) > 1e-12:
            rel = 100.0 * abs_diff / abs(f_mean)
        else:
            rel = np.nan
        rows.append({
            "axis_bin_center": float(centers[i]),
            "scanning_mean": s_mean,
            "scanning_std": s_std,
            "scanning_n": int(s_sub.size),
            "forward_mean": f_mean,
            "forward_std": f_std,
            "forward_n": int(f_sub.size),
            "abs_diff": abs_diff,
            "rel_diff_pct": rel,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plot_pair_overlay                                                           #
# --------------------------------------------------------------------------- #


def plot_pair_overlay(
    scanning_df: pd.DataFrame,
    forward_df: pd.DataFrame,
    *,
    axis: str,
    metric: str,
    ax: Optional["plt.Axes"] = None,
    axis_log: bool = False,
    n_bins: int = 8,
    scanning_label: str = "Scanning (360°)",
    forward_label: str = "Forward (90°)",
    **kw: Any,
) -> Dict[str, Any]:
    """Plot scanning vs forward metric overlay on a single axes.

    Returns dict ``{ax, scanning_line, forward_line, table}``.
    """
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib not available")

    if ax is None:
        fig, ax = plt.subplots(figsize=kw.get("figsize", (7.0, 5.0)))

    table = compare_metric(
        scanning_df, forward_df, metric=metric, axis=axis,
        n_bins=n_bins, axis_log=axis_log,
    )

    s_line = ax.errorbar(
        table["axis_bin_center"], table["scanning_mean"],
        yerr=table["scanning_std"],
        fmt="o-", color="C0", lw=2.0, ms=8, capsize=4,
        label=f"{scanning_label}",
    )
    f_line = ax.errorbar(
        table["axis_bin_center"], table["forward_mean"],
        yerr=table["forward_std"],
        fmt="s--", color="C3", lw=2.0, ms=8, capsize=4,
        label=f"{forward_label}",
    )
    if axis_log:
        ax.set_xscale("log")
    ax.set_xlabel(axis)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return {"ax": ax, "scanning_line": s_line, "forward_line": f_line, "table": table}


__all__ = ["compare_metric", "plot_pair_overlay"]
