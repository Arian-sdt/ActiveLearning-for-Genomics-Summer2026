"""Evaluate prediction columns with paired bootstrap Pearson analysis.

Purpose: expose the statistical core independently of GPU inference.
How to run: ``debour-evaluate --predictions predictions.tsv --target target --output-dir OUT``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .bootstrap import paired_bootstrap_pearson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--target", default="K562_log2FC")
    parser.add_argument("--prediction-columns", nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, sep="\t")
    columns = args.prediction_columns or [column for column in frame if column.startswith("pred_")]
    summary, pairwise, distributions = paired_bootstrap_pearson(
        frame[args.target].to_numpy(),
        {column: frame[column].to_numpy() for column in columns},
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "bootstrap_summary.tsv", sep="\t", index=False)
    pairwise.to_csv(output / "pairwise_tests.tsv", sep="\t", index=False)
    distributions.to_csv(output / "bootstrap_distributions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

