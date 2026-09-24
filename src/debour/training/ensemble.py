"""Train multiple seeded models on one exact shared ID set.

Purpose:
    Create genuine deep ensembles where model initialization/order changes but
    the sampled examples remain identical across seeds.

How to use:
    Import ``train_ensemble`` or run ``workflows/train_ensemble.py``. The workflow
    can run locally or emit Slurm jobs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import TrainingConfig
from .trainer import train_model


def train_ensemble(
    frame: pd.DataFrame,
    output_root: str | Path,
    seeds: list[int],
    base_config: TrainingConfig,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    results = []
    for seed in seeds:
        results.append(train_model(frame, output_root / f"seed{seed}", replace(base_config, seed=seed)))
    return results

