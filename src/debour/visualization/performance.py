"""Model learning-curve and bootstrap-confidence-interval plots.

Purpose:
    Graph Pearson r against training size and compare many models without the
    oversized legends that obscured earlier figures. Top models can be highlighted.

How to use:
    Import the plotting functions. ``workflows/evaluate_models.py`` calls the
    bootstrap plot automatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curve(results: pd.DataFrame, output: str | Path) -> None:
    ordered = results.sort_values("training_rows")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ordered["training_rows"], ordered["pearsonr"], marker="o", color="#125c8c", linewidth=2)
    ax.set(xlabel="Number of training sequences", ylabel="Pearson r", title="Heldout performance by training-set size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_bootstrap_summary(summary: pd.DataFrame, output: str | Path, *, highlight_top: int = 3) -> None:
    ordered = summary.sort_values("observed_pearsonr", ascending=True).reset_index(drop=True)
    colors = np.array(["#3274a1"] * len(ordered), dtype=object)
    if highlight_top:
        top_positions = ordered.nlargest(highlight_top, "observed_pearsonr").index
        colors[top_positions] = ["#d62728", "#ff7f0e", "#2ca02c"][: len(top_positions)]
    y = np.arange(len(ordered))
    lower = ordered["observed_pearsonr"] - ordered["ci_lower_95"]
    upper = ordered["ci_upper_95"] - ordered["observed_pearsonr"]
    height = max(4.5, 0.32 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.errorbar(ordered["observed_pearsonr"], y, xerr=[lower, upper], fmt="none", color="#999999", capsize=2)
    ax.scatter(ordered["observed_pearsonr"], y, c=colors, s=30, zorder=3)
    ax.set_yticks(y, ordered["model"])
    ax.set(xlabel="Pearson r on shared heldout set", ylabel="Model", title="Observed Pearson r with paired-bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
