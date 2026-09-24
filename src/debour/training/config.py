"""Serializable training hyperparameters for the BHI CNN+BiLSTM.

Purpose: make every model run explicit and reproducible.
How to use: construct ``TrainingConfig(...)`` or load matching YAML defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainingConfig:
    sequence_length: int = 200
    epochs: int = 80
    batch_size: int = 32
    validation_batch_size: int = 256
    validation_fraction: float = 0.1
    learning_rate: float = 0.005
    weight_decay: float = 0.01
    dropout: float = 0.1
    huber_delta: float = 1.0
    pct_start: float = 0.3
    div_factor: float = 25.0
    workers: int = 0
    seed: int = 42
    device: str = "cuda"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

