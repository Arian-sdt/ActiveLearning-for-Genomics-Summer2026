"""Training configuration, datasets, single-model training, and ensembles.

Purpose: expose reproducible training APIs used by local and Slurm workflows.
How to use: call ``train_model`` or run the workflow scripts in ``workflows/``.
"""

__all__ = ["TrainingConfig", "train_model"]


def __getattr__(name: str):
    """Lazily expose training APIs so the trainer CLI remains warning-free."""

    if name == "TrainingConfig":
        from .config import TrainingConfig

        return TrainingConfig
    if name == "train_model":
        from .trainer import train_model

        return train_model
    raise AttributeError(name)
