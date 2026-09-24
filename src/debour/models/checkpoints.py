"""Checkpoint discovery and model reconstruction.

Purpose:
    Load either historical raw ``model_best.pth`` state dictionaries or richer
    training checkpoints without duplicating compatibility logic.

How to use:
    ``model = load_model_checkpoint(run_dir, device='cuda')``. This module is
    intended for import by inference/evaluation workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .bhi_cnn_bilstm import BhiCnnBiLstm


def find_checkpoint(run_or_checkpoint: str | Path) -> Path:
    path = Path(run_or_checkpoint)
    if path.is_file():
        return path
    for name in ("model_best.pth", "best.pt", "last.pt"):
        candidate = path / name
        if candidate.exists():
            return candidate
    named = sorted(path.glob("model_best_*.pth"))
    if named:
        return named[-1]
    raise FileNotFoundError(f"No model checkpoint found under {path}")


def load_model_checkpoint(
    run_or_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dropout: float | None = None,
) -> BhiCnnBiLstm:
    path = Path(run_or_checkpoint)
    run_dir = path if path.is_dir() else path.parent
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model = BhiCnnBiLstm(dropout=float(dropout if dropout is not None else config.get("dropout", 0.1)))
    state = torch.load(find_checkpoint(path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()

