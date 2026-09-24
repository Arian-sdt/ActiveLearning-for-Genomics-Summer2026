"""Generate redundant families by mutating prototype DNA sequences.

Purpose:
    Create a controlled candidate pool where family membership and mutation count
    are known. Copies receive unique positions and a different canonical base at
    each position; copy counts and mutation levels are seeded and reproducible.

How to use:
    Call ``generate_mutated_families(prototypes, min_copies=50, max_copies=100,
    mutation_levels=[1,4,7,10,13,16,19], seed=42)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mutate_sequence(sequence: str, mutations: int, rng: np.random.Generator) -> str:
    sequence = list(sequence.upper())
    positions = rng.choice(len(sequence), size=mutations, replace=False)
    for position in positions:
        alternatives = [base for base in "ACGT" if base != sequence[position]]
        sequence[position] = rng.choice(alternatives)
    return "".join(sequence)


def generate_mutated_families(
    prototypes: pd.DataFrame,
    *,
    id_column: str = "IDs",
    sequence_column: str = "sequence",
    min_copies: int = 50,
    max_copies: int = 100,
    mutation_levels: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    mutation_levels = mutation_levels or [1, 2, 3, 4, 5]
    rng = np.random.default_rng(seed)
    rows = []
    for parent_index, record in prototypes.reset_index(drop=True).iterrows():
        copies = int(rng.integers(min_copies, max_copies + 1))
        for copy_index in range(copies):
            mutations = int(rng.choice(mutation_levels))
            rows.append({
                "synthetic_id": f"parent{parent_index:04d}_copy{copy_index:03d}",
                "parent_id": str(record[id_column]),
                "parent_index": parent_index,
                "num_mutations": mutations,
                "mutated_sequence": mutate_sequence(record[sequence_column], mutations, rng),
            })
    return pd.DataFrame(rows)

