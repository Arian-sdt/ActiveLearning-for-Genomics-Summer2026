#!/usr/bin/env python3
"""Run stages from the real K562 uncertainty-learning-curve experiment.

Purpose:
    Preserve the concrete 120k base, +20k uncertainty-round, and shared-holdout
    evaluation study while delegating all implementation to reusable workflows.
    Historical defaults are five seeds (42-46), 80 epochs, 20k additions, and a
    37k heldout test. Fresh ensembles are trained after each addition.

How to run:
    Print the base-training command with
    ``python3 experiments/real_k562_active_learning/run.py train-base --table
    TABLE``. Put ``--execute`` before the stage to run it. Other stages are
    ``uncertainty-round`` and ``evaluate``; use ``--help`` for their path inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from debour.cluster.commands import run_command

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="Run instead of printing the assembled command.")
    stages = result.add_subparsers(dest="stage", required=True)

    base = stages.add_parser("train-base", help="Train the shared 120k five-seed ensemble.")
    base.add_argument("--table", required=True)
    base.add_argument("--rows", type=int, default=120000)
    base.add_argument("--output-dir", default="outputs/experiments/real_k562/base120k")
    base.add_argument("--device", default="cuda")

    round_parser = stages.add_parser("uncertainty-round", help="Select 20k and retrain a fresh expanded ensemble.")
    round_parser.add_argument("--table", required=True)
    round_parser.add_argument("--base-ids", required=True)
    round_parser.add_argument("--model-runs", nargs="+", required=True)
    round_parser.add_argument("--add-rows", type=int, default=20000)
    round_parser.add_argument("--output-dir", required=True)
    round_parser.add_argument("--device", default="cuda")

    evaluation = stages.add_parser("evaluate", help="Compare checkpoints on one shared unseen holdout.")
    evaluation.add_argument("--table", required=True)
    evaluation.add_argument("--model-runs", nargs="+", required=True)
    evaluation.add_argument("--model-names", nargs="+")
    evaluation.add_argument("--test-rows", type=int, default=37000)
    evaluation.add_argument("--test-seed", type=int, default=42)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--device", default="cuda")
    return result


def command(args: argparse.Namespace) -> list[str]:
    if args.stage == "train-base":
        return [
            sys.executable, "workflows/train_ensemble.py", "--table", args.table,
            "--sample-rows", str(args.rows), "--selection-seed", "42", "--seeds",
            *map(str, SEEDS), "--epochs", "80", "--device", args.device,
            "--output-dir", args.output_dir,
        ]
    if args.stage == "uncertainty-round":
        return [
            sys.executable, "workflows/run_active_learning_round.py", "--table", args.table,
            "--base-ids", args.base_ids, "--model-runs", *args.model_runs,
            "--add-rows", str(args.add_rows), "--seeds", *map(str, SEEDS),
            "--epochs", "80", "--device", args.device, "--output-dir", args.output_dir,
        ]
    assembled = [
        sys.executable, "workflows/evaluate_models.py", "--table", args.table,
        "--model-runs", *args.model_runs,
    ]
    if args.model_names:
        if len(args.model_names) != len(args.model_runs):
            raise ValueError("--model-names must align with --model-runs")
        assembled.extend(["--model-names", *args.model_names])
    assembled.extend([
        "--test-rows", str(args.test_rows), "--test-seed", str(args.test_seed),
        "--bootstrap-samples", "1000", "--device", args.device,
        "--output-dir", args.output_dir,
    ])
    return assembled


def main() -> None:
    args = parser().parse_args()
    run_command(command(args), execute=args.execute, cwd=ROOT)


if __name__ == "__main__":
    main()
