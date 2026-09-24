"""Generate per-model predictions and ensemble uncertainty.

Purpose:
    Score a clean sequence frame with any number of compatible checkpoints and
    report each prediction, mean, sample variance, and standard deviation.

How to use:
    Import ``predict_ensemble``. Selection workflows call it automatically;
    model-run paths may point to run directories or checkpoint files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from debour.data.schema import DEFAULT_SCHEMA
from debour.data.sequences import normalize_sequence, one_hot_encode
from debour.models.checkpoints import load_model_checkpoint


class PredictionDataset(Dataset):
    def __init__(self, sequences: list[str], sequence_length: int):
        self.sequences = [normalize_sequence(sequence, sequence_length) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return one_hot_encode(self.sequences[index])


@torch.no_grad()
def predict_ensemble(
    frame: pd.DataFrame,
    model_runs: list[str | Path],
    *,
    sequence_length: int = 200,
    batch_size: int = 512,
    workers: int = 0,
    device: str = "cuda",
) -> pd.DataFrame:
    if len(model_runs) < 1:
        raise ValueError("At least one model is required")
    dataset = PredictionDataset(frame[DEFAULT_SCHEMA.sequence_column].tolist(), sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    columns: list[np.ndarray] = []
    for model_index, run in enumerate(model_runs, start=1):
        model = load_model_checkpoint(run, device=device)
        predictions = [
            model(inputs.to(device)).cpu().numpy()
            for inputs in tqdm(loader, desc=f"Model {model_index}/{len(model_runs)}", leave=False)
        ]
        columns.append(np.concatenate(predictions).astype(np.float64))
        del model
    matrix = np.column_stack(columns)
    result = frame.copy()
    for index in range(matrix.shape[1]):
        result[f"pred_model_{index + 1}"] = matrix[:, index]
    result["prediction_mean"] = matrix.mean(axis=1)
    if matrix.shape[1] > 1:
        result["prediction_variance"] = matrix.var(axis=1, ddof=1)
        result["prediction_std"] = matrix.std(axis=1, ddof=1)
    else:
        result["prediction_variance"] = 0.0
        result["prediction_std"] = 0.0
    return result

