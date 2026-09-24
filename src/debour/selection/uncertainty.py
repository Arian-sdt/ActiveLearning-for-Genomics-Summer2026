"""Rank sequences by ensemble prediction variance.

Purpose: select sequences on which independently seeded models disagree most.
How to use: pass a frame from ``predict_ensemble`` to ``select_uncertain``.
"""

from __future__ import annotations

import pandas as pd


def select_uncertain(scored: pd.DataFrame, rows: int) -> pd.DataFrame:
    if "prediction_variance" not in scored:
        raise ValueError("prediction_variance column is required")
    selected = scored.sort_values("prediction_variance", ascending=False).head(rows).copy()
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_score"] = selected["prediction_variance"]
    return selected.reset_index(drop=True)

