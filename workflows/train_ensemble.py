#!/usr/bin/env python3
"""Train a seeded CNN+BiLSTM ensemble on one exact shared sequence set.

Purpose:
    Load/clean a real or synthetic Table S2-compatible table, choose IDs exactly
    once, save that shared set, and train each seed from scratch with identical
    architecture and hyperparameters.

How to run:
    ``python3 workflows/train_ensemble.py --table TABLE --sample-rows 10000
    --seeds 42 43 44 45 46 --output-dir outputs/ensemble10k``.
    Run inside a GPU allocation, or use the documented Slurm command template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from debour.data.io import load_sequence_table, read_ids, write_ids
from debour.data.schema import DEFAULT_SCHEMA
from debour.data.splits import sample_ids
from debour.training.config import TrainingConfig
from debour.training.ensemble import train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--id-file")
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    include = set(read_ids(args.id_file)) if args.id_file else None
    frame = load_sequence_table(args.table, include_ids=include)
    if args.sample_rows:
        chosen = sample_ids(frame[DEFAULT_SCHEMA.id_column].tolist(), args.sample_rows, args.selection_seed)
        chosen_set = set(chosen)
        frame = frame[frame[DEFAULT_SCHEMA.id_column].isin(chosen_set)].reset_index(drop=True)
    write_ids(frame[DEFAULT_SCHEMA.id_column].tolist(), output / "shared_ids.txt")
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, device=args.device)
    results = train_ensemble(frame, output, args.seeds, config)
    (output / "ensemble_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

