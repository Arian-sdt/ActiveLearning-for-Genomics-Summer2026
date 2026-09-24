"""Performance, UMAP, and facility-coverage plotting.

Purpose: generate publication-ready figures from saved tabular artifacts.
How to use: import plot functions; workflows write PNGs without interactive display.
"""

from .performance import plot_bootstrap_summary, plot_learning_curve

__all__ = ["plot_bootstrap_summary", "plot_learning_curve"]

