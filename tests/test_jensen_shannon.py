"""Purpose: verify symmetry, bounds, and identity of JS distance.
How to run: ``python3 -m pytest tests/test_jensen_shannon.py``.
"""

import numpy as np

from debour.features.jensen_shannon import js_distance, weighted_js_distance


def test_js_properties() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert js_distance(p, p) == 0
    assert np.isclose(js_distance(p, q), 1.0)
    assert js_distance(p, q) == js_distance(q, p)


def test_weighted_distance_identity() -> None:
    vector = np.concatenate([np.full(4, 0.25), np.full(16, 1 / 16), np.full(64, 1 / 64)])
    assert np.isclose(weighted_js_distance(vector, vector), 0.0)

