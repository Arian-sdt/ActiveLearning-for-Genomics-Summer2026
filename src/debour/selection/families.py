"""Build weighted ground-set centroids from known synthetic families.

Purpose:
    Replace approximate LSH buckets with the true ``parent_id`` groups available
    in controlled synthetic mutation data. Base-training rows are removed before
    grouping, so each centroid and weight describe only selectable, remaining
    copies from that family. The centroid is the mean normalized k-mer vector and
    the weight is the number of remaining family members it represents.

How to use:
    Call ``family_centroids(frame, feature_columns, family_column="parent_id")``
    after joining synthetic metadata to a normalized k-mer table and excluding
    base IDs. Pass the returned centroids and weights to the standard facility or
    hybrid selector. No LSH computation is involved.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def family_centroids(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    family_column: str = "parent_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable family labels, mean feature vectors, and family sizes."""

    if frame.empty:
        raise ValueError("Cannot build family centroids from an empty table")
    required = [family_column, *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing family-centroid columns: {missing}")
    if frame[family_column].isna().any():
        raise ValueError(f"{family_column} contains missing values")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    working = frame[required].copy()
    working[family_column] = working[family_column].astype(str)
    numeric = working[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Family feature columns contain missing or non-numeric values")
    working[list(feature_columns)] = numeric

    grouped = working.groupby(family_column, sort=True, observed=True)
    centroid_frame = grouped[list(feature_columns)].mean()
    weights = grouped.size().reindex(centroid_frame.index).to_numpy(np.float64)
    labels = centroid_frame.index.to_numpy(dtype=str)
    centroids = centroid_frame.to_numpy(np.float32)
    return labels, centroids, weights


def family_summary(
    labels: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
    feature_columns: Sequence[str],
    *,
    family_column: str = "parent_id",
    initial_coverage: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create an auditable table of family sizes and centroid features."""

    if len(labels) != len(weights) or len(labels) != len(centroids):
        raise ValueError("labels, weights, and centroids must have equal lengths")
    result = pd.DataFrame({family_column: labels, "remaining_family_size": weights.astype(int)})
    if initial_coverage is not None:
        initial_coverage = np.asarray(initial_coverage, dtype=np.float64)
        if initial_coverage.shape != (len(labels),):
            raise ValueError("initial_coverage must have one value per family")
        result["initial_base_coverage"] = initial_coverage
    for index, column in enumerate(feature_columns):
        result[f"centroid_{column}"] = centroids[:, index]
    return result
