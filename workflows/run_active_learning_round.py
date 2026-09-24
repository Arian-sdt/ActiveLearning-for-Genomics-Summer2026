#!/usr/bin/env python3
"""Compose one uncertainty-selection plus retraining active-learning round.

Purpose:
    Make the recurring experiment explicit: score the unused pool with a supplied
    ensemble, select the top uncertain IDs, union them with the base IDs, then
    train a fresh ensemble on the expanded set. The model is never fine-tuned.

How to run:
    ``python3 workflows/run_active_learning_round.py --table TABLE
    --base-ids BASE.txt --model-runs RUN1 RUN2 RUN3 RUN4 RUN5 --add-rows 2000
    --seeds 42 43 44 45 46 --output-dir outputs/round1``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from debour.data.io import load_sequence_table, read_ids, write_ids
from debour.inference.ensemble_predictions import predict_ensemble
from debour.selection.base import save_selection
from debour.selection.uncertainty import select_uncertain
from debour.training.config import TrainingConfig
from debour.training.ensemble import train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--base-ids", required=True)
    parser.add_argument("--model-runs", nargs="+", required=True)
    parser.add_argument("--add-rows", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    base = set(read_ids(args.base_ids))
    candidates = load_sequence_table(args.table, exclude_ids=base)
    scores = predict_ensemble(candidates, args.model_runs, batch_size=args.batch_size, workers=args.workers, device=args.device)
    selected = select_uncertain(scores, args.add_rows)
    save_selection(selected, output / "selection", vars(args))
    combined = base | set(selected["IDs"].astype(str))
    write_ids(sorted(combined), output / "combined_training_ids.txt")
    training = load_sequence_table(args.table, include_ids=combined)
    config = TrainingConfig(epochs=args.epochs, workers=args.workers, device=args.device)
    train_ensemble(training, output / "models", args.seeds, config)


if __name__ == "__main__":
    main()

