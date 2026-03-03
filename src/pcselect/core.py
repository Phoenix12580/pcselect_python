from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _PCAStats:
    stdev: np.ndarray
    variance: np.ndarray
    total_variance: float
    pct_var: np.ndarray
    cumulative_pct: np.ndarray


def _extract_stdev(pca_like: Any) -> np.ndarray:
    """Extract PCA standard deviations from common Python objects.

    Supported inputs:
    - 1D iterable of stdev values
    - objects with ``stdev_`` or ``stdev``
    - scikit-learn PCA objects with ``explained_variance_``
    - AnnData-like object with ``uns['pca']['variance']``
    """
    if isinstance(pca_like, Iterable) and not hasattr(pca_like, "shape"):
        stdev = np.asarray(list(pca_like), dtype=float)
    elif isinstance(pca_like, np.ndarray):
        stdev = pca_like.astype(float)
    elif hasattr(pca_like, "stdev_"):
        stdev = np.asarray(getattr(pca_like, "stdev_"), dtype=float)
    elif hasattr(pca_like, "stdev"):
        stdev = np.asarray(getattr(pca_like, "stdev"), dtype=float)
    elif hasattr(pca_like, "explained_variance_"):
        stdev = np.sqrt(np.asarray(getattr(pca_like, "explained_variance_"), dtype=float))
    elif hasattr(pca_like, "uns") and isinstance(pca_like.uns, dict):
        variance = pca_like.uns.get("pca", {}).get("variance")
        if variance is None:
            raise ValueError("Could not find PCA variance in anndata.uns['pca']['variance'].")
        stdev = np.sqrt(np.asarray(variance, dtype=float))
    else:
        raise TypeError(
            "Unsupported pca_like input. Pass stdev iterable/array, sklearn PCA, or AnnData-like object."
        )

    if stdev.ndim != 1 or stdev.size == 0:
        raise ValueError("PCA standard deviations must be a non-empty 1D array.")
    if np.any(stdev < 0):
        raise ValueError("PCA standard deviations must be non-negative.")
    return stdev


def _compute_stats(stdev: np.ndarray) -> _PCAStats:
    variance = stdev**2
    total = float(np.sum(variance))
    pct_var = variance / total
    cumulative_pct = np.cumsum(pct_var)
    return _PCAStats(
        stdev=stdev,
        variance=variance,
        total_variance=total,
        pct_var=pct_var,
        cumulative_pct=cumulative_pct,
    )


def calculate_optimal_pcs(
    pca_like: Any,
    min_variance: float = 0.75,
    max_marginal_gain: float = 0.005,
) -> int:
    """Calculate the recommended number of PCs.

    Mirrors scPCselect logic by combining:
    1) minimum cumulative variance threshold,
    2) marginal gain threshold, and
    3) elbow point from curvature.

    Returns the median of available candidate cutoffs.
    """
    stdev = _extract_stdev(pca_like)
    stats = _compute_stats(stdev)

    pcs_variance = int(np.argmax(stats.cumulative_pct > min_variance) + 1)
    marginal_gain = np.concatenate(([stats.variance[0]], np.diff(stats.cumulative_pct * stats.total_variance))) / stats.total_variance
    below = np.where(marginal_gain < max_marginal_gain)[0]
    pcs_marginal = int(below[0] + 1) if below.size else len(stdev)

    diff1 = np.diff(stats.stdev)
    diff2 = np.diff(diff1)
    pcs_elbow = int(np.argmax(np.abs(diff2)) + 2) if diff2.size else len(stdev)

    candidates = np.array([pcs_variance, pcs_marginal, pcs_elbow], dtype=int)
    return int(np.round(np.median(candidates)))


def get_variance_summary(pca_like: Any, pc_range: Iterable[int] | None = None) -> pd.DataFrame:
    """Generate a variance summary table by selected PC cutoffs."""
    stdev = _extract_stdev(pca_like)
    stats = _compute_stats(stdev)

    if pc_range is None:
        pc_range = range(10, 51, 5)

    valid = np.array([n for n in pc_range if 1 <= int(n) <= len(stdev)], dtype=int)
    if valid.size == 0:
        raise ValueError("pc_range has no values in the valid interval [1, n_pcs].")

    marginal5 = []
    avg = []
    for n in valid:
        start = max(0, n - 5)
        marginal5.append(np.sum(stats.pct_var[start:n]) * 100)
        avg.append(np.mean(stats.pct_var[:n]) * 100)

    return pd.DataFrame(
        {
            "n_pcs": valid,
            "cumulative_variance": stats.cumulative_pct[valid - 1] * 100,
            "marginal_variance": np.array(marginal5),
            "avg_variance_per_pc": np.array(avg),
        }
    )


def visualize_pc_selection(
    pca_like: Any,
    max_pcs: int = 50,
    variance_thresholds: Iterable[float] = (0.60, 0.70, 0.80, 0.90),
    figsize: tuple[float, float] = (14, 10),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Create scPCselect-style diagnostic plots for PC selection.

    Returns a tuple of ``(plot_data, figure)``.
    """
    stdev = _extract_stdev(pca_like)
    max_pcs = max(3, min(max_pcs, len(stdev)))
    subset_stdev = stdev[:max_pcs]

    full = _compute_stats(stdev)
    sub = _compute_stats(subset_stdev)

    marginal_gain = np.concatenate(([sub.pct_var[0] * 100], np.diff(sub.cumulative_pct * 100)))

    diff1 = np.diff(sub.stdev)
    diff2 = np.diff(diff1)
    curvature = np.full(max_pcs, np.nan)
    if diff2.size:
        curvature[2 : 2 + diff2.size] = diff2

    plot_data = pd.DataFrame(
        {
            "pc": np.arange(1, max_pcs + 1),
            "stdev": subset_stdev,
            "individual_var": sub.pct_var * 100,
            "cumulative_var": sub.cumulative_pct * 100,
            "marginal_gain": marginal_gain,
            "curvature": curvature,
        }
    )

    thresholds = np.array(list(variance_thresholds), dtype=float)
    threshold_pcs = []
    for t in thresholds:
        idx = np.where(full.cumulative_pct > t)[0]
        threshold_pcs.append(int(idx[0] + 1) if idx.size else len(stdev))

    colors = ["red", "orange", "green", "blue", "purple"]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(plot_data["pc"], plot_data["stdev"], color="steelblue", marker="o", ms=3)
    for i, (pc, t) in enumerate(zip(threshold_pcs, thresholds)):
        if pc <= max_pcs:
            color = colors[i % len(colors)]
            ax1.axvline(pc, linestyle="--", color=color, alpha=0.6)
            ax1.text(pc + 0.2, np.nanmax(subset_stdev) * (1 - i * 0.08), f"{pc} PCs\n({t*100:.0f}%)", color=color, fontsize=8)
    ax1.set_title("1. Elbow Plot (Standard Deviation)")
    ax1.set_xlabel("Principal Component")

    ax2.plot(plot_data["pc"], plot_data["cumulative_var"], color="darkgreen", marker="o", ms=3)
    for i, t in enumerate(thresholds):
        ax2.axhline(t * 100, linestyle="--", color=colors[i % len(colors)], alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_title("2. Cumulative Variance Explained")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Variance (%)")

    ax3.bar(plot_data["pc"], plot_data["marginal_gain"], color="coral", alpha=0.7)
    ax3.axhline(0.5, linestyle="--", color="red")
    ax3.set_title("3. Marginal Gain per PC")
    ax3.set_xlabel("Principal Component")
    ax3.set_ylabel("Variance Gained (%)")

    valid_curv = plot_data.dropna(subset=["curvature"])
    ax4.plot(valid_curv["pc"], np.abs(valid_curv["curvature"]), color="purple", marker="o", ms=3)
    ax4.set_title("4. Curvature (2nd Derivative)")
    ax4.set_xlabel("Principal Component")
    ax4.set_ylabel("Absolute Curvature")

    fig.tight_layout()
    return plot_data, fig
