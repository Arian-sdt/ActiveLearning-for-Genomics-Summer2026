"""Purpose: verify known mutation families become correctly weighted centroids.
How to run: ``python3 -m pytest tests/test_family_selection.py``.
"""

import numpy as np
import pandas as pd

from debour.selection.families import family_centroids, family_summary


def test_family_centroids_use_means_and_remaining_family_sizes() -> None:
    frame = pd.DataFrame({
        "parent_id": ["b", "a", "a"],
        "k1_A": [0.0, 1.0, 0.5],
        "k1_C": [1.0, 0.0, 0.5],
    })
    labels, centroids, weights = family_centroids(frame, ["k1_A", "k1_C"])
    assert labels.tolist() == ["a", "b"]
    np.testing.assert_allclose(centroids, [[0.75, 0.25], [0.0, 1.0]])
    np.testing.assert_allclose(weights, [2.0, 1.0])

    summary = family_summary(labels, weights, centroids, ["k1_A", "k1_C"])
    assert summary["remaining_family_size"].tolist() == [2, 1]
    assert summary.columns.tolist() == [
        "parent_id", "remaining_family_size", "centroid_k1_A", "centroid_k1_C"
    ]

    conditional = family_summary(
        labels,
        weights,
        centroids,
        ["k1_A", "k1_C"],
        initial_coverage=np.array([0.4, 0.8]),
    )
    np.testing.assert_allclose(conditional["initial_base_coverage"], [0.4, 0.8])


def test_family_centroids_reject_missing_family_ids() -> None:
    frame = pd.DataFrame({"parent_id": ["a", None], "k1_A": [1.0, 0.5]})
    try:
        family_centroids(frame, ["k1_A"])
    except ValueError as error:
        assert "missing values" in str(error)
    else:
        raise AssertionError("Expected missing family IDs to fail")
