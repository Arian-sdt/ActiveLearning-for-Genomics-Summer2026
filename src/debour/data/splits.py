"""Deterministic sampling, train/validation splitting, and leakage checks.

Purpose:
    Ensure model comparisons use explicit, reproducible ID sets and prevent test
    sequences from appearing in any training or selection set.

How to use:
    Import the helpers, or run ``python3 -m debour.data.splits --help`` to create
    a deterministic sampled ID file from a table.
"""

from __future__ import annotations

import argparse

import numpy as np

from .io import load_sequence_table, write_ids
from .schema import DEFAULT_SCHEMA


def split_indices(size: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two rows are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    permutation = np.random.default_rng(seed).permutation(size)
    validation_size = max(1, int(size * validation_fraction))
    return permutation[validation_size:], permutation[:validation_size]


def sample_ids(ids: list[str], rows: int, seed: int) -> list[str]:
    if rows > len(ids):
        raise ValueError(f"Requested {rows:,} IDs from a pool of {len(ids):,}")
    chosen = np.random.default_rng(seed).choice(len(ids), rows, replace=False)
    return [ids[index] for index in chosen]


def assert_disjoint(*named_sets: tuple[str, set[str]]) -> None:
    for index, (left_name, left) in enumerate(named_sets):
        for right_name, right in named_sets[index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"{left_name} and {right_name} overlap by {len(overlap):,} IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = load_sequence_table(args.table)
    write_ids(sample_ids(frame[DEFAULT_SCHEMA.id_column].tolist(), args.rows, args.seed), args.output)


if __name__ == "__main__":
    main()

