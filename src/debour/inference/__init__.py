"""Batched model and ensemble prediction utilities.

Purpose: share one prediction implementation across selection and evaluation.
How to use: import ``predict_ensemble`` from this package.
"""

from .ensemble_predictions import predict_ensemble

__all__ = ["predict_ensemble"]

