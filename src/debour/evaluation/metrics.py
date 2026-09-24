"""Regression metrics used in model validation and heldout evaluation.

Purpose: centralize Pearson r, MSE, RMSE, and MAE with edge-case handling.
How to use: call ``regression_metrics(observed, predicted)``.
"""

from __future__ import annotations

import numpy as np


def pearsonr(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted) - np.asarray(observed)
    mse = float(np.mean(residual**2))
    return {
        "pearsonr": pearsonr(observed, predicted),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
    }

