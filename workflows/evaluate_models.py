#!/usr/bin/env python3
"""Evaluate any model runs on one unseen shared holdout with paired bootstrap.

Purpose:
    Discover each run's training IDs, exclude their union, sample one heldout set,
    predict with every model, report metrics/paired p-values, and save a compact
    confidence-interval figure with no legend.

How to run:
    ``python3 workflows/evaluate_models.py --table TABLE --model-runs RUN_A RUN_B
    --test-rows 10000 --bootstrap-samples 1000 --output-dir OUT``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from debour.data.schema import DEFAULT_SCHEMA
from debour.evaluation.bootstrap import paired_bootstrap_pearson
from debour.evaluation.holdout import build_shared_holdout, discover_training_ids
from debour.inference.ensemble_predictions import predict_ensemble
from debour.visualization.performance import plot_bootstrap_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--model-runs", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    names = args.model_names or [Path(run).name for run in args.model_runs]
    if len(names) != len(args.model_runs):
        parser.error("--model-names must match --model-runs length")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    holdout = build_shared_holdout(
        args.table,
        [discover_training_ids(run) for run in args.model_runs],
        args.test_rows,
        args.test_seed,
    )
    predictions = {}
    result = holdout.copy()
    for name, run in zip(names, args.model_runs):
        values = predict_ensemble(holdout, [run], batch_size=args.batch_size, workers=args.workers, device=args.device)["prediction_mean"]
        predictions[name] = values.to_numpy()
        result[f"pred_{name}"] = values
    result.to_csv(output / "predictions.tsv.gz", sep="\t", index=False)
    summary, pairwise, distributions = paired_bootstrap_pearson(
        holdout[DEFAULT_SCHEMA.target_column].to_numpy(), predictions,
        samples=args.bootstrap_samples, seed=args.bootstrap_seed,
    )
    summary.to_csv(output / "bootstrap_summary.tsv", sep="\t", index=False)
    pairwise.to_csv(output / "pairwise_tests.tsv", sep="\t", index=False)
    distributions.to_csv(output / "bootstrap_distributions.tsv.gz", sep="\t", index=False)
    plot_bootstrap_summary(summary, output / "bootstrap_pearson_summary.png")
    (output / "evaluation_config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()

