"""2-D phase-diagram plotting for MC batch results.

Provides the ``plot_2d_slice`` helper used to render the paper's phase
diagrams from a batch summary.

The default rendering is a hexbin colourmap, which copes well with a
small (N=100) smoke batch and scales to N≥10⁴ at full MC.  An optional
``mode="contour"`` draws a tricontourf on top of the scatter for
boolean / pass-rate metrics.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend; figures saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_2d_slice(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    metric: str,
    *,
    ax: Optional[plt.Axes] = None,
    mode: str = "hexbin",
    gridsize: int = 20,
    cmap: str = "viridis",
    title: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Axes:
    """Render a 2-D slice of a per-run summary DataFrame.

    Parameters
    ----------
    df : DataFrame
        Per-run summaries (long form).  Must contain x_axis, y_axis,
        metric columns.
    x_axis, y_axis, metric : str
        Column names.
    ax : matplotlib.Axes, optional
        Target axes; if None a new figure is created.
    mode : {"hexbin", "scatter", "contour"}, default "hexbin"
        Rendering style.  "contour" requires ≥ 3 finite points.
    gridsize : int, default 20
        Hexbin grid size.
    cmap : str, default "viridis"
        Colormap name.
    title : str, optional
        Plot title.
    xscale, yscale : {"linear", "log"}, default "linear"
        Axis scales.
    save_path : str, optional
        If given, ``fig.savefig(save_path, dpi=dpi)`` is called and the
        Axes returned without ``plt.show``.
    dpi : int, default 150
        Save DPI.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.figure
    for col in (x_axis, y_axis, metric):
        if col not in df.columns:
            raise KeyError(f"column {col!r} missing from df.columns={list(df.columns)}")

    x = df[x_axis].to_numpy(dtype=float)
    y = df[y_axis].to_numpy(dtype=float)
    m = df[metric]
    if m.dtype == bool:
        m = m.astype(int)
    z = m.to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[finite]; y = y[finite]; z = z[finite]

    if mode == "hexbin":
        # bins='log' option useful for highly skewed metrics; default linear.
        hb = ax.hexbin(
            x, y, C=z, gridsize=gridsize, cmap=cmap,
            xscale=xscale, yscale=yscale,
            reduce_C_function=np.mean,
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label(f"{metric} (mean per cell)")
    elif mode == "scatter":
        sc = ax.scatter(x, y, c=z, cmap=cmap, s=24, edgecolors="k", linewidths=0.3)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(metric)
    elif mode == "contour":
        if len(x) < 3:
            raise ValueError("contour mode requires ≥ 3 finite points")
        tcf = ax.tricontourf(x, y, z, levels=10, cmap=cmap)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        cb = fig.colorbar(tcf, ax=ax)
        cb.set_label(metric)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected hexbin/scatter/contour")

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    if title:
        ax.set_title(title)
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi)
    return ax
