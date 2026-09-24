"""Shared holdout prediction, regression metrics, and paired bootstrap analysis.

Purpose: make model comparisons statistically paired and leakage-aware.
How to use: import functions or run ``debour-evaluate --help``.
"""

from .bootstrap import paired_bootstrap_pearson
from .metrics import regression_metrics

__all__ = ["paired_bootstrap_pearson", "regression_metrics"]

