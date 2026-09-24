"""Construct one shared heldout set that all compared models have not seen.

Purpose:
    Exclude the union of every model's sampled/training IDs before deterministic
    test sampling. This prevents favorable leakage for larger training sets.

How to use:
    Call ``build_shared_holdout(table, training_id_files, rows, seed)``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from debour.data.io import load_sequence_table, read_ids
from debour.data.schema import DEFAULT_SCHEMA


def build_shared_holdout(
    table: str | Path,
    training_id_files: list[str | Path],
    rows: int,
    seed: int,
) -> pd.DataFrame:
    excluded: set[str] = set()
    for path in training_id_files:
        excluded.update(read_ids(path))
    frame = load_sequence_table(table, exclude_ids=excluded)
    if rows > len(frame):
        raise ValueError(f"Requested {rows:,} test rows from {len(frame):,} eligible rows")
    return frame.sample(n=rows, random_state=seed).reset_index(drop=True)


def discover_training_ids(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    for name in ("sampled_ids.txt", "train_ids.txt"):
        candidate = run_dir / "splits" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No training ID file found in {run_dir / 'splits'}")

