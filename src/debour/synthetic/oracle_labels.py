"""Assign synthetic activity labels with a trained BHI teacher model.

Purpose:
    Turn mutation-family sequences into a supervised benchmark. Outputs are
    unconstrained predictions on the K562 log2 fold-change scale, so negative
    values are valid and mean predicted activity below the reference baseline.

How to use:
    Rename synthetic ID/sequence columns to the standard schema, call
    ``label_with_oracle``, and save the returned Table S2-compatible frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from debour.data.schema import DEFAULT_SCHEMA
from debour.inference.ensemble_predictions import predict_ensemble


def label_with_oracle(
    frame: pd.DataFrame,
    checkpoint: str | Path,
    *,
    batch_size: int = 512,
    workers: int = 0,
    device: str = "cuda",
) -> pd.DataFrame:
    scored = predict_ensemble(frame, [checkpoint], batch_size=batch_size, workers=workers, device=device)
    result = frame.copy()
    result[DEFAULT_SCHEMA.target_column] = scored["prediction_mean"]
    return result

