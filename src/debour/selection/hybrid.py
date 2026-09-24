"""Greedy hybrid acquisition combining uncertainty and facility gain.

Purpose:
    Balance predictive disagreement with representative coverage. Variance is
    min-max normalized once over the candidate pool. At every greedy round the
    current facility marginal gain is normalized by total ground weight, giving
    a bounded per-ground-element average gain. Lambda weights uncertainty:
    score = lambda * normalized_variance + (1-lambda) * normalized_facility_gain.

How to use:
    Call ``greedy_hybrid_selection`` with candidate variance and features. Set
    ``initial_coverage`` for conditional selection or leave it zero for
    unconditional selection.
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from debour.features.jensen_shannon import weighted_js_similarity
from .facility_location import SimilarityFunction, _similarity_to_ground


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def greedy_hybrid_selection(
    candidate_features: np.ndarray,
    prediction_variance: np.ndarray,
    ground_features: np.ndarray,
    rows: int,
    *,
    lambda_uncertainty: float,
    ground_weights: np.ndarray | None = None,
    initial_coverage: np.ndarray | None = None,
    similarity: SimilarityFunction = weighted_js_similarity,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    if not 0 <= lambda_uncertainty <= 1:
        raise ValueError("lambda_uncertainty must be in [0,1]")
    weights = np.ones(len(ground_features), dtype=np.float64) if ground_weights is None else np.asarray(ground_weights, dtype=np.float64)
    coverage = np.zeros(len(ground_features), dtype=np.float64) if initial_coverage is None else np.asarray(initial_coverage, dtype=np.float64).copy()
    normalized_variance = minmax(prediction_variance)
    available = np.ones(len(candidate_features), dtype=bool)
    selected: list[int] = []
    diagnostics: list[dict[str, float | int]] = []

    for rank in tqdm(range(1, rows + 1), desc="Hybrid greedy selection", unit="selected"):
        best_index = -1
        best_score = -float("inf")
        best_gain = 0.0
        best_similarity = None
        for index in np.flatnonzero(available):
            similarities = _similarity_to_ground(candidate_features[index], ground_features, similarity)
            raw_gain = float(np.dot(weights, np.maximum(similarities - coverage, 0.0)))
            normalized_gain = raw_gain / weights.sum()
            score = lambda_uncertainty * normalized_variance[index] + (1 - lambda_uncertainty) * normalized_gain
            if score > best_score:
                best_index, best_score, best_gain, best_similarity = index, score, raw_gain, similarities
        available[best_index] = False
        selected.append(best_index)
        coverage = np.maximum(coverage, best_similarity)
        diagnostics.append({
            "selection_rank": rank,
            "candidate_index": best_index,
            "hybrid_score": best_score,
            "normalized_variance": float(normalized_variance[best_index]),
            "raw_facility_gain": best_gain,
            "normalized_facility_gain": best_gain / weights.sum(),
            "normalized_coverage": float(np.dot(weights, coverage) / weights.sum()),
        })
    return np.asarray(selected), diagnostics

