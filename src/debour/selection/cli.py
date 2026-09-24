"""Command-line interface for random and pre-scored uncertainty selection.

Purpose:
    Expose stable selection commands while keeping algorithms importable. More
    complex facility/hybrid runs are configured through workflows/select_sequences.py.

How to run:
    ``debour-select random --table TABLE --rows 2000 --output-dir OUT`` or
    ``debour-select uncertainty --scores predictions.tsv --rows 2000 --output-dir OUT``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .base import save_selection
from .random import select_random
from .uncertainty import select_uncertain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="method", required=True)
    random_parser = subparsers.add_parser("random")
    random_parser.add_argument("--table", required=True)
    random_parser.add_argument("--rows", type=int, required=True)
    random_parser.add_argument("--seed", type=int, default=42)
    random_parser.add_argument("--output-dir", required=True)
    uncertainty_parser = subparsers.add_parser("uncertainty")
    uncertainty_parser.add_argument("--scores", required=True)
    uncertainty_parser.add_argument("--rows", type=int, required=True)
    uncertainty_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.method == "random":
        frame = pd.read_csv(args.table, sep="\t")
        selected = select_random(frame, args.rows, args.seed)
    else:
        frame = pd.read_csv(args.scores, sep="\t")
        selected = select_uncertain(frame, args.rows)
    save_selection(selected, args.output_dir, vars(args))


if __name__ == "__main__":
    main()

