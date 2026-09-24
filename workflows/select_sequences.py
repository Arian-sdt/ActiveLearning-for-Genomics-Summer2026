#!/usr/bin/env python3
"""Select random or ensemble-uncertain sequences from an unused candidate pool.

Purpose:
    Streamline the two universally required acquisition baselines. It excludes
    all supplied IDs, preserves prediction variance for every eligible sequence,
    and saves the ordered selection with complete configuration metadata.

How to run:
    ``python3 workflows/select_sequences.py uncertainty --table TABLE
    --model-runs RUN1 RUN2 RUN3 RUN4 RUN5 --exclude-ids BASE.txt
    --select-rows 2000 --output-dir OUT``. Use ``random`` instead of
    ``uncertainty`` for the random baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from debour.data.io import load_sequence_table, read_ids
from debour.inference.ensemble_predictions import predict_ensemble
from debour.selection.base import save_selection
from debour.selection.random import select_random
from debour.selection.uncertainty import select_uncertain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["random", "uncertainty"])
    parser.add_argument("--table", required=True)
    parser.add_argument("--model-runs", nargs="*")
    parser.add_argument("--exclude-ids", nargs="+", required=True)
    parser.add_argument("--select-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    excluded: set[str] = set()
    for path in args.exclude_ids:
        excluded.update(read_ids(path))
    candidates = load_sequence_table(args.table, exclude_ids=excluded)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.method == "random":
        selected = select_random(candidates, args.select_rows, args.seed)
    else:
        if not args.model_runs:
            parser.error("--model-runs is required for uncertainty selection")
        scored = predict_ensemble(
            candidates, args.model_runs, batch_size=args.batch_size,
            workers=args.workers, device=args.device,
        )
        scored.sort_values("prediction_variance", ascending=False).to_csv(
            output / "all_candidate_variances.tsv.gz", sep="\t", index=False
        )
        selected = select_uncertain(scored, args.select_rows)
    save_selection(selected, output, vars(args))


if __name__ == "__main__":
    main()

