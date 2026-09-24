"""Purpose: verify lambda endpoints recover facility and uncertainty priorities.
How to run: ``python3 -m pytest tests/test_hybrid_selection.py``.
"""

import numpy as np

from debour.selection.hybrid import greedy_hybrid_selection


def dot_similarity(ground: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.sum(ground * candidates, axis=-1)


def test_lambda_one_selects_highest_variance() -> None:
    candidates = np.array([[1, 0], [0, 1]], dtype=np.float32)
    variance = np.array([0.1, 0.9])
    selected, _ = greedy_hybrid_selection(
        candidates, variance, candidates, 1,
        lambda_uncertainty=1.0, similarity=dot_similarity,
    )
    assert selected.tolist() == [1]

