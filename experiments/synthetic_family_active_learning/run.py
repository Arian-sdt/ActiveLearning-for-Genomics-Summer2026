#!/usr/bin/env python3
"""Run stages from the controlled synthetic family active-learning study.

Purpose:
    Preserve the experiment with 1,000 diverse parents, 50-100 mutated copies,
    oracle labels, a shared random 10k base, and 2k additions selected by random,
    uncertainty, known-family facility, conditional family facility, or hybrid
    scoring. Known ``parent_id`` groups replace LSH in diversity stages.

How to run:
    Print a stage with ``python3 experiments/synthetic_family_active_learning/
    run.py generate --source-table TABLE``. Add ``--execute`` before the stage
    to launch it. Available stages are generate, train-base, select-unconditional,
    select-conditional, select-hybrid-unconditional, select-hybrid-conditional,
    and evaluate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from debour.cluster.commands import run_command

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]
STAGES = (
    "generate",
    "train-base",
    "select-unconditional",
    "select-conditional",
    "select-hybrid-unconditional",
    "select-hybrid-conditional",
    "evaluate",
)


def require(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"Stage {args.stage} requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--source-table")
    parser.add_argument("--oracle-table")
    parser.add_argument("--synthetic-table")
    parser.add_argument("--kmer-table")
    parser.add_argument("--base-ids")
    parser.add_argument("--candidate-ids")
    parser.add_argument("--variance-table")
    parser.add_argument("--training-ids", help="Precombined base-plus-selection IDs for training a selected model.")
    parser.add_argument("--model-runs", nargs="+")
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.5)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    output = args.output_dir or f"outputs/experiments/synthetic_family/{args.stage}"
    if args.stage == "generate":
        require(args, "source_table")
        return [
            sys.executable, "workflows/run_synthetic_benchmark.py",
            "--table", args.source_table, "--prototype-rows", "1000",
            "--min-copies", "50", "--max-copies", "100",
            "--mutation-levels", "1", "4", "7", "10", "13", "16", "19",
            "--seed", "42", "--output-dir", output,
        ]
    if args.stage == "train-base":
        require(args, "oracle_table")
        command = [
            sys.executable, "workflows/train_ensemble.py", "--table", args.oracle_table,
        ]
        if args.training_ids:
            command.extend(["--id-file", args.training_ids])
        else:
            command.extend(["--sample-rows", "10000", "--selection-seed", "2026"])
        command.extend([
            "--seeds", *map(str, SEEDS), "--epochs", "80", "--device", args.device,
            "--output-dir", output,
        ])
        return command
    if args.stage == "evaluate":
        require(args, "oracle_table", "model_runs")
        command = [
            sys.executable, "workflows/evaluate_models.py", "--table", args.oracle_table,
            "--model-runs", *args.model_runs,
        ]
        if args.model_names:
            if len(args.model_names) != len(args.model_runs):
                raise ValueError("--model-names must align with --model-runs")
            command.extend(["--model-names", *args.model_names])
        command.extend([
            "--test-rows", "10000", "--test-seed", str(args.test_seed),
            "--bootstrap-samples", "1000", "--device", args.device,
            "--output-dir", output,
        ])
        return command

    require(args, "synthetic_table", "kmer_table", "base_ids")
    hybrid = "hybrid" in args.stage
    conditional = args.stage.endswith("conditional") and not args.stage.endswith("unconditional")
    if hybrid:
        require(args, "variance_table")
    command = [
        sys.executable, "workflows/select_family_facility.py",
        "--synthetic-table", args.synthetic_table,
        "--kmer-table", args.kmer_table,
        "--base-ids", args.base_ids,
        "--method", "hybrid" if hybrid else "facility",
        "--select-rows", "2000", "--output-dir", output,
    ]
    if conditional:
        command.append("--conditional")
    if args.candidate_ids:
        command.extend(["--candidate-ids", args.candidate_ids])
    if hybrid:
        command.extend([
            "--variance-table", args.variance_table,
            "--lambda-uncertainty", str(args.lambda_uncertainty),
        ])
    return command


def main() -> None:
    args = parse_args()
    run_command(build_command(args), execute=args.execute, cwd=ROOT)


if __name__ == "__main__":
    main()
