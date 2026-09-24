#!/usr/bin/env python3
"""Run historical uncertainty-facility lambda sweeps.

Purpose:
    Generate comparable conditional and/or unconditional hybrid selections for
    a list of uncertainty weights. The runner supports LSH centroids for real
    data and known parent-family centroids for synthetic data. Lambda always
    weights normalized ensemble variance; ``1-lambda`` weights normalized
    facility marginal gain.

How to run:
    ``python3 experiments/hybrid_lambda_sweep/run.py --representation family
    --mode both --kmer-table KMERS.tsv --synthetic-table COPIES.tsv --base-ids
    BASE.txt --variance-table VARIANCE.tsv`` prints every command. Add
    ``--execute`` to run them sequentially.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from debour.cluster.commands import run_command

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--representation", choices=["lsh", "family"], required=True)
    parser.add_argument("--mode", choices=["conditional", "unconditional", "both"], default="both")
    parser.add_argument("--kmer-table", required=True)
    parser.add_argument("--synthetic-table", help="Required for family representation.")
    parser.add_argument("--base-ids", required=True)
    parser.add_argument("--variance-table", required=True)
    parser.add_argument("--candidate-ids")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--select-rows", type=int, default=2000)
    parser.add_argument("--lsh-bits", type=int, default=22)
    parser.add_argument("--output-root", default="outputs/experiments/hybrid_lambda_sweep")
    return parser.parse_args()


def lambda_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.representation == "family" and not args.synthetic_table:
        raise ValueError("Family representation requires --synthetic-table")
    if any(value < 0 or value > 1 for value in args.lambdas):
        raise ValueError("Every lambda must be in [0,1]")
    modes = ["unconditional", "conditional"] if args.mode == "both" else [args.mode]

    for mode in modes:
        for value in args.lambdas:
            workflow = (
                "workflows/select_family_facility.py"
                if args.representation == "family"
                else "workflows/select_facility.py"
            )
            output = Path(args.output_root) / args.representation / mode / f"lambda_{lambda_label(value)}"
            command = [
                sys.executable, workflow,
                "--kmer-table", args.kmer_table,
                "--base-ids", args.base_ids,
                "--method", "hybrid",
                "--variance-table", args.variance_table,
                "--lambda-uncertainty", str(value),
                "--select-rows", str(args.select_rows),
                "--output-dir", str(output),
            ]
            if args.representation == "family":
                command.extend(["--synthetic-table", args.synthetic_table])
            else:
                command.extend(["--lsh-bits", str(args.lsh_bits)])
            if mode == "conditional":
                command.append("--conditional")
            if args.candidate_ids:
                command.extend(["--candidate-ids", args.candidate_ids])
            run_command(command, execute=args.execute, cwd=ROOT)


if __name__ == "__main__":
    main()
