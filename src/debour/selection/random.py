"""Uniform random selection without replacement.

Purpose: provide the essential label-budget baseline for active learning.
How to use: call ``select_random(frame, rows, seed)`` or use the selection CLI.
"""

from __future__ import annotations

import pandas as pd


def select_random(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if rows > len(frame):
        raise ValueError("Selection size exceeds candidate pool")
    selected = frame.sample(n=rows, random_state=seed).copy()
    selected.insert(0, "selection_rank", range(1, rows + 1))
    selected["selection_score"] = float("nan")
    return selected.reset_index(drop=True)

