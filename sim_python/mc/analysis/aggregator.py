"""Batch aggregator for MC outputs.

Reads every run parquet under ``batch_dir`` and provides
``compute_per_cell`` for 2-D grid aggregation (mean / std / pass_rate
per axis cell).

Two entry points:

- ``load_batch(batch_dir) -> pd.DataFrame``
  Joins ``_summary.parquet`` (per-run summaries) with optional
  trajectory metadata.  It is summary-only by default: the per-run
  trajectories live at ``<batch_dir>/run_<hash>/trajectory.parquet``
  and are not eagerly loaded — they would be 100 × ~30 KB ≈ 3 MB on a
  smoke batch but could be N × 50 KB ≈ 500 MB at full MC.

- ``compute_per_cell(df, axes, metric, nbins=8, agg='mean') -> pd.DataFrame``
  Bin the two named axes into ``nbins`` × ``nbins`` cells and
  aggregate ``metric`` per cell.  Returns long-form
  ``{x_axis, y_axis, agg_value, count}`` for downstream plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# load_batch                                                                  #
# --------------------------------------------------------------------------- #


def load_batch(batch_dir: str | Path) -> pd.DataFrame:
    """Load the per-run summary parquet emitted by the dispatcher.

    Parameters
    ----------
    batch_dir : path-like
        Directory containing ``_summary.parquet``.

    Returns
    -------
    DataFrame
        Per-run summary rows; one row per dispatched config.
    """
    p = Path(batch_dir)
    summary = p / "_summary.parquet"
    if not summary.exists():
        raise FileNotFoundError(
            f"no _summary.parquet under {p}; did dispatcher finish?"
        )
    df = pd.read_parquet(summary)
    return df


def load_trajectory(batch_dir: str | Path, config_hash: str) -> pd.DataFrame:
    """Load a single trajectory by config_hash from a batch dir."""
    p = Path(batch_dir) / f"run_{config_hash}" / "trajectory.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no trajectory at {p}")
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- #
# compute_per_cell                                                            #
# --------------------------------------------------------------------------- #


_AGG_FUNCS: dict[str, Callable] = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "count": len,
}


def compute_per_cell(
    df: pd.DataFrame,
    axes: Tuple[str, str],
    metric: str,
    nbins: int | Tuple[int, int] = 8,
    agg: str = "mean",
    *,
    log_axes: Sequence[str] = (),
) -> pd.DataFrame:
    """Bin two axes and aggregate ``metric`` per cell.

    Parameters
    ----------
    df : DataFrame
        Per-run summary table (from ``load_batch``).
    axes : (x_axis, y_axis)
        Column names to bin on.
    metric : str
        Column name to aggregate.
    nbins : int or (int, int), default 8
        Number of bins along each axis.
    agg : {"mean", "std", "median", "min", "max", "count"}, default "mean"
        Aggregation function name.
    log_axes : iterable of axis names
        Axes that should be binned in log-space (uniform in log of the
        column's positive values).

    Returns
    -------
    DataFrame with columns ``[x_axis, y_axis, agg_value, count]`` —
    one row per non-empty cell.  Coordinates are the cell *centres*.

    Notes
    -----
    For boolean metrics (e.g. ``trackable``), ``agg="mean"`` produces
    the cell pass-rate ∈ [0, 1].
    """
    if not isinstance(axes, (tuple, list)) or len(axes) != 2:
        raise ValueError(f"axes must be a 2-tuple of column names; got {axes!r}")
    x_axis, y_axis = axes
    for col in (x_axis, y_axis, metric):
        if col not in df.columns:
            raise KeyError(f"column {col!r} missing from df.columns={list(df.columns)}")
    if agg not in _AGG_FUNCS:
        raise ValueError(f"agg must be one of {list(_AGG_FUNCS)}; got {agg!r}")
    if isinstance(nbins, int):
        nx = ny = nbins
    else:
        nx, ny = nbins
    if nx <= 0 or ny <= 0:
        raise ValueError(f"nbins must be positive; got {(nx, ny)}")

    log_set = set(log_axes)

    def _edges(col: str, nb: int) -> np.ndarray:
        vals = df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ValueError(f"column {col!r} has no finite values")
        if col in log_set:
            pos = finite[finite > 0]
            if pos.size == 0:
                raise ValueError(f"log-axis {col!r} has no positive values")
            return np.exp(np.linspace(np.log(pos.min()), np.log(pos.max()), nb + 1))
        return np.linspace(finite.min(), finite.max(), nb + 1)

    x_edges = _edges(x_axis, nx)
    y_edges = _edges(y_axis, ny)

    # Bin indices (clip to last bin for the upper edge).
    xi = np.clip(np.digitize(df[x_axis].to_numpy(dtype=float), x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(df[y_axis].to_numpy(dtype=float), y_edges) - 1, 0, ny - 1)

    fn = _AGG_FUNCS[agg]
    rows: list[dict] = []
    metric_vals = df[metric]
    # Convert booleans → ints so mean returns pass-rate.
    if metric_vals.dtype == bool:
        metric_vals = metric_vals.astype(int)
    metric_arr = metric_vals.to_numpy(dtype=float)

    for i in range(nx):
        for j in range(ny):
            mask = (xi == i) & (yi == j)
            if not mask.any():
                continue
            xc = 0.5 * (x_edges[i] + x_edges[i + 1])
            yc = 0.5 * (y_edges[j] + y_edges[j + 1])
            vals = metric_arr[mask]
            v = float(fn(vals)) if len(vals) else float("nan")
            rows.append(
                {
                    x_axis: xc,
                    y_axis: yc,
                    "agg_value": v,
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)
