"""Greedy facility-location selection over representative ground points.

Purpose:
    Select actual candidate sequences that maximize weighted coverage of a ground
    set. The ground may be all sequences, LSH bucket centroids, or synthetic
    family centroids. Conditional mode initializes coverage from a base training
    set; unconditional mode starts at zero after merely excluding base IDs.

Objective:
    F(S | T) = sum_i weight_i * max(max_{s in S} sim(i,s), max_{t in T} sim(i,t))

How to use:
    Call ``greedy_facility_location`` with candidate/ground feature matrices.
    Similarity defaults to 1 minus weighted Jensen-Shannon distance. The routine
    returns indices in selection order plus marginal-gain diagnostics.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np
from tqdm.auto import tqdm

from debour.features.jensen_shannon import weighted_js_similarity

SimilarityFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _similarity_to_ground(
    candidate: np.ndarray,
    ground: np.ndarray,
    similarity: SimilarityFunction,
) -> np.ndarray:
    tiled = np.broadcast_to(candidate, ground.shape)
    return np.asarray(similarity(ground, tiled), dtype=np.float64)


def initialize_conditional_coverage(
    ground: np.ndarray,
    base_features: np.ndarray | None,
    similarity: SimilarityFunction = weighted_js_similarity,
) -> np.ndarray:
    coverage = np.zeros(len(ground), dtype=np.float64)
    if base_features is None or len(base_features) == 0:
        return coverage
    for feature in tqdm(base_features, desc="Initializing base-set coverage", leave=False):
        coverage = np.maximum(coverage, _similarity_to_ground(feature, ground, similarity))
    return coverage


def greedy_facility_location(
    candidate_features: np.ndarray,
    ground_features: np.ndarray,
    rows: int,
    *,
    ground_weights: np.ndarray | None = None,
    initial_coverage: np.ndarray | None = None,
    similarity: SimilarityFunction = weighted_js_similarity,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Exact lazy-greedy maximization of a monotone facility objective."""

    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    ground_features = np.asarray(ground_features, dtype=np.float32)
    if rows > len(candidate_features):
        raise ValueError("rows exceeds candidate count")
    weights = np.ones(len(ground_features), dtype=np.float64) if ground_weights is None else np.asarray(ground_weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("ground_weights must be nonnegative with positive sum")
    coverage = np.zeros(len(ground_features), dtype=np.float64) if initial_coverage is None else np.asarray(initial_coverage, dtype=np.float64).copy()
    if coverage.shape != (len(ground_features),):
        raise ValueError("initial_coverage has the wrong shape")

    heap: list[tuple[float, int, int]] = []
    for index, feature in enumerate(tqdm(candidate_features, desc="Initial facility gains", unit="candidate")):
        similarities = _similarity_to_ground(feature, ground_features, similarity)
        gain = float(np.dot(weights, np.maximum(similarities - coverage, 0.0)))
        heapq.heappush(heap, (-gain, index, 0))

    selected: list[int] = []
    diagnostics: list[dict[str, float | int]] = []
    chosen: set[int] = set()
    progress = tqdm(total=rows, desc="Lazy-greedy facility selection", unit="selected")
    while len(selected) < rows:
        negative_bound, index, evaluated_round = heapq.heappop(heap)
        if index in chosen:
            continue
        similarities = _similarity_to_ground(candidate_features[index], ground_features, similarity)
        gain = float(np.dot(weights, np.maximum(similarities - coverage, 0.0)))
        if evaluated_round == len(selected):
            chosen.add(index)
            selected.append(index)
            coverage = np.maximum(coverage, similarities)
            diagnostics.append({
                "selection_rank": len(selected),
                "candidate_index": index,
                "marginal_facility_gain": gain,
                "normalized_coverage": float(np.dot(weights, coverage) / weights.sum()),
            })
            progress.update(1)
            progress.set_postfix(gain=f"{gain:.5g}", coverage=f"{diagnostics[-1]['normalized_coverage']:.4f}")
        else:
            heapq.heappush(heap, (-gain, index, len(selected)))
    progress.close()
    return np.asarray(selected, dtype=np.int64), diagnostics

