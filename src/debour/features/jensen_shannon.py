"""Jensen-Shannon distance for normalized k-mer distributions.

Purpose:
    Compute a bounded, symmetric sequence-composition distance. The project uses
    base-2 logarithms and combines k=1,2,3 distances with weights 0.1,0.3,0.6.

How to use:
    Import ``weighted_js_distance`` for vectors/matrices, or run
    ``python3 -m debour.features.jensen_shannon --help`` for two TSV rows.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DEFAULT_BLOCK_SIZES = (4, 16, 64)
DEFAULT_WEIGHTS = (0.1, 0.3, 0.6)


def js_distance(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return sqrt(JS divergence) in [0,1] for probability distributions."""

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_term = np.where(p > 0, p * np.log2(p / midpoint), 0.0)
        q_term = np.where(q > 0, q * np.log2(q / midpoint), 0.0)
    divergence = 0.5 * (p_term.sum(axis=axis) + q_term.sum(axis=axis))
    return np.sqrt(np.maximum(divergence, 0.0))


def weighted_js_distance(
    p: np.ndarray,
    q: np.ndarray,
    *,
    block_sizes: tuple[int, ...] = DEFAULT_BLOCK_SIZES,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    if len(block_sizes) != len(weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("block_sizes and weights must align, and weights must sum to 1")
    if p.shape[-1] != sum(block_sizes) or q.shape[-1] != sum(block_sizes):
        raise ValueError(f"Expected {sum(block_sizes)} features")
    distances = []
    start = 0
    for size in block_sizes:
        stop = start + size
        distances.append(js_distance(p[..., start:stop], q[..., start:stop]))
        start = stop
    return sum(weight * distance for weight, distance in zip(weights, distances))


def weighted_js_similarity(p: np.ndarray, q: np.ndarray, **kwargs: object) -> np.ndarray:
    return 1.0 - weighted_js_distance(p, q, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table")
    parser.add_argument("--id-a", required=True)
    parser.add_argument("--id-b", required=True)
    parser.add_argument("--id-column", default="IDs")
    args = parser.parse_args()
    frame = pd.read_csv(args.table, sep="\t")
    feature_columns = [column for column in frame.columns if column.startswith(("k1_", "k2_", "k3_"))]
    indexed = frame.set_index(args.id_column)
    p = indexed.loc[args.id_a, feature_columns].to_numpy(float)
    q = indexed.loc[args.id_b, feature_columns].to_numpy(float)
    print(float(weighted_js_distance(p, q)))


if __name__ == "__main__":
    main()

