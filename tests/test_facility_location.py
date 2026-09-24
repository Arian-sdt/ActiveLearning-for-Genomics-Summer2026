"""Purpose: verify greedy facility selection covers distinct ground regions.
How to run: ``python3 -m pytest tests/test_facility_location.py``.
"""

import numpy as np

from debour.selection.facility_location import greedy_facility_location


def dot_similarity(ground: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.sum(ground * candidates, axis=-1)


def test_selects_two_orthogonal_representatives() -> None:
    candidates = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    ground = np.array([[1, 0], [0, 1]], dtype=np.float32)
    selected, diagnostics = greedy_facility_location(candidates, ground, 2, similarity=dot_similarity)
    assert set(selected) == {0, 1}
    assert diagnostics[-1]["normalized_coverage"] == 1.0

