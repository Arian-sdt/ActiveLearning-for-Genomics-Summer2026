"""Random-hyperplane locality-sensitive hashing for k-mer vectors.

Purpose:
    Compress a large sequence ground set into weighted local buckets. Each hash
    bit records which side of a seeded random hyperplane a vector lies on;
    centroids and bucket sizes then approximate sequence-level facility coverage.

How to use:
    ``assign_lsh_buckets(features, bits=22, seed=42)`` followed by
    ``bucket_centroids``. The selection workflow can consume these arrays.
"""

from __future__ import annotations

import numpy as np


def assign_lsh_buckets(features: np.ndarray, bits: int = 22, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if bits <= 0 or bits > 63:
        raise ValueError("bits must be between 1 and 63")
    projection = np.random.default_rng(seed).standard_normal((features.shape[1], bits)).astype(np.float32)
    binary = features @ projection >= 0
    powers = (1 << np.arange(bits, dtype=np.uint64))
    return (binary.astype(np.uint64) * powers).sum(axis=1), projection


def bucket_centroids(features: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, inverse, counts = np.unique(assignments, return_inverse=True, return_counts=True)
    centroids = np.zeros((len(labels), features.shape[1]), dtype=np.float64)
    np.add.at(centroids, inverse, features)
    centroids /= counts[:, None]
    # Each k-mer probability block remains normalized after averaging.
    return labels, centroids.astype(np.float32), counts.astype(np.float64)

