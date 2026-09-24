"""Train one BHI-style CNN+BiLSTM regression model.

Purpose:
    Provide the canonical training implementation used by every real and
    synthetic experiment: Huber loss, AdamW, cosine OneCycleLR, and best-model
    selection by validation Pearson r. It writes exact sampled/train/validation
    IDs, config, history, best weights, and a resumable last checkpoint.

How to run:
    ``debour-train --table TABLE.tsv --output-dir RUN --sample-rows 10000``
    or ``python3 -m debour.training.trainer --help``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from debour.data.io import load_sequence_table, read_ids, write_ids
from debour.data.schema import DEFAULT_SCHEMA
from debour.data.sequences import normalize_sequence, one_hot_encode
from debour.data.splits import sample_ids, split_indices
from debour.models.bhi_cnn_bilstm import BhiCnnBiLstm
from .config import TrainingConfig


def json_safe(value: object) -> object:
    """Convert NumPy scalars and non-finite floats into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class SequenceRegressionDataset(Dataset):
    """In-memory clean sequence table encoded lazily per item."""

    def __init__(self, frame: pd.DataFrame, sequence_length: int):
        self.ids = frame[DEFAULT_SCHEMA.id_column].astype(str).tolist()
        self.sequences = [
            normalize_sequence(sequence, sequence_length)
            for sequence in frame[DEFAULT_SCHEMA.sequence_column].tolist()
        ]
        self.targets = frame[DEFAULT_SCHEMA.target_column].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return one_hot_encode(self.sequences[index]), torch.tensor(self.targets[index])


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for inputs, target in loader:
        predictions.append(model(inputs.to(device)).cpu().numpy())
        targets.append(target.numpy())
    predicted = np.concatenate(predictions)
    observed = np.concatenate(targets)
    return {
        "mse": float(np.mean((predicted - observed) ** 2)),
        "pearsonr": pearsonr(observed, predicted),
    }


def train_model(
    frame: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, object]:
    """Train one model and return a compact run summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)

    dataset = SequenceRegressionDataset(frame, config.sequence_length)
    train_indices, validation_indices = split_indices(
        len(dataset), config.validation_fraction, config.seed
    )
    train_dataset = Subset(dataset, train_indices.tolist())
    validation_dataset = Subset(dataset, validation_indices.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )

    model = BhiCnnBiLstm(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate / config.div_factor,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        anneal_strategy="cos",
    )
    criterion = nn.HuberLoss(delta=config.huber_delta)

    split_dir = output_dir / "splits"
    write_ids(dataset.ids, split_dir / "sampled_ids.txt")
    write_ids([dataset.ids[index] for index in train_indices], split_dir / "train_ids.txt")
    write_ids([dataset.ids[index] for index in validation_indices], split_dir / "val_ids.txt")
    run_config = {
        **config.to_dict(),
        "architecture": "BHI-style CNN+BiLSTM",
        "encoding_channels": ["A", "G", "C", "T"],
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR cosine",
        "loss": "HuberLoss",
        "model_selection": "best validation Pearson r",
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    history: list[dict[str, float | int]] = []
    best_pearson = -float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.epochs:03d}", unit="batch")
        for inputs, targets in progress:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
            progress.set_postfix(huber=f"{np.mean(losses[-25:]):.5f}", lr=f"{scheduler.get_last_lr()[0]:.3g}")

        metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_huber": float(np.mean(losses)),
            **metrics,
            "epoch_seconds": time.time() - started,
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2, allow_nan=False))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": row,
            "config": run_config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        score = metrics["pearsonr"]
        # Always leave a usable best checkpoint. Pearson can be undefined for a
        # tiny or constant validation set; a later finite score supersedes it.
        if epoch == 1 or (np.isfinite(score) and (not np.isfinite(best_pearson) or score > best_pearson)):
            best_pearson, best_epoch = score, epoch
            torch.save(model.state_dict(), output_dir / "model_best.pth")
        print(
            f"epoch {epoch:03d} | train_huber {row['train_huber']:.5f} | "
            f"val_mse {metrics['mse']:.5f} | val_pearson {score:.5f}"
        )

    summary = {"best_epoch": best_epoch, "best_validation_pearson": best_pearson, **run_config}
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-file")
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--div-factor", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_ids = set(read_ids(args.id_file)) if args.id_file else None
    frame = load_sequence_table(args.table, include_ids=include_ids)
    if args.sample_rows is not None:
        selected = set(sample_ids(frame[DEFAULT_SCHEMA.id_column].tolist(), args.sample_rows, args.selection_seed))
        frame = frame[frame[DEFAULT_SCHEMA.id_column].isin(selected)].reset_index(drop=True)
    config = TrainingConfig(
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        huber_delta=args.huber_delta,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
    )
    train_model(frame, args.output_dir, config)


if __name__ == "__main__":
    main()
