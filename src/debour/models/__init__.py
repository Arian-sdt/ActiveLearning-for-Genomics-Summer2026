"""Neural network architectures and checkpoint loading.

Purpose: keep model definitions independent from training and inference workflows.
How to use: import ``BhiCnnBiLstm`` or ``load_model_checkpoint``.
"""

__all__ = ["BhiCnnBiLstm", "load_model_checkpoint"]


def __getattr__(name: str):
    """Lazily expose model APIs so architecture self-checks run without warnings."""

    if name == "BhiCnnBiLstm":
        from .bhi_cnn_bilstm import BhiCnnBiLstm

        return BhiCnnBiLstm
    if name == "load_model_checkpoint":
        from .checkpoints import load_model_checkpoint

        return load_model_checkpoint
    raise AttributeError(name)
