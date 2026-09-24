#!/usr/bin/env python3
"""Run the real-data LSH centroid facility-selection experiment.

Purpose:
    Preserve the study that compressed unused real K562 k-mer vectors with
    random-hyperplane LSH and used weighted bucket centroids as the facility
    ground set. It supports unconditional and base-aware conditional facility
    selection plus matching hybrid variants.

How to run:
    ``python3 experiments/lsh_centroid_selection/run.py facility-conditional
    --kmer-table KMERS.tsv --base-ids BASE.txt`` prints the command. Add
    ``--execute`` before the stage to run it. Hybrid stages also require
    ``--variance-table``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from debour.cluster.commands import run_command

ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    "facility-unconditional",
    "facility-conditional",
    "hybrid-unconditional",
    "hybrid-conditional",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--kmer-table", required=True)
    parser.add_argument("--base-ids", required=True)
    parser.add_argument("--candidate-ids")
    parser.add_argument("--variance-table")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.5)
    parser.add_argument("--lsh-bits", type=int, default=22)
    parser.add_argument("--lsh-seed", type=int, default=42)
    parser.add_argument("--select-rows", type=int, default=20000)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hybrid = args.stage.startswith("hybrid")
    conditional = args.stage.endswith("conditional") and not args.stage.endswith("unconditional")
    if hybrid and not args.variance_table:
        raise ValueError("Hybrid stages require --variance-table")
    output = args.output_dir or f"outputs/experiments/lsh_centroid/{args.stage}"
    command = [
        sys.executable, "workflows/select_facility.py",
        "--kmer-table", args.kmer_table,
        "--base-ids", args.base_ids,
        "--method", "hybrid" if hybrid else "facility",
        "--lsh-bits", str(args.lsh_bits),
        "--lsh-seed", str(args.lsh_seed),
        "--select-rows", str(args.select_rows),
        "--output-dir", output,
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
    run_command(command, execute=args.execute, cwd=ROOT)


if __name__ == "__main__":
    main()
