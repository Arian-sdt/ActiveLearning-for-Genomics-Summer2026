#!/usr/bin/env python3
"""Run centroid-approximated facility or hybrid sequence selection.

Purpose:
    Load one normalized 84-column k-mer table, exclude the base training IDs,
    build weighted LSH centroids from the remaining pool, and select actual rows.
    Conditional mode initializes centroid coverage from the base set. Hybrid mode
    adds normalized ensemble variance with lambda weighting uncertainty.

How to run:
    ``python3 workflows/select_facility.py --kmer-table KMERS.tsv
    --base-ids BASE.txt --method facility --conditional --select-rows 2000
    --output-dir OUT``. For hybrid selection also pass ``--variance-table`` and
    ``--lambda-uncertainty``. Use a variance-prefiltered candidate ID file for
    hybrid runs because exact reranking scans every candidate at every round.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from debour.data.io import read_ids, write_ids
from debour.selection.facility_location import greedy_facility_location, initialize_conditional_coverage
from debour.selection.hybrid import greedy_hybrid_selection
from debour.selection.lsh import assign_lsh_buckets, bucket_centroids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmer-table", required=True)
    parser.add_argument("--id-column", default="IDs")
    parser.add_argument("--base-ids", required=True)
    parser.add_argument("--candidate-ids")
    parser.add_argument("--method", choices=["facility", "hybrid"], default="facility")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--variance-table")
    parser.add_argument("--variance-column", default="prediction_variance")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.5)
    parser.add_argument("--lsh-bits", type=int, default=22)
    parser.add_argument("--lsh-seed", type=int, default=42)
    parser.add_argument("--select-rows", type=int, default=2000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.kmer_table, sep="\t")
    feature_columns = [column for column in frame if column.startswith(("k1_", "k2_", "k3_"))]
    if len(feature_columns) != 84:
        raise ValueError(f"Expected 84 normalized k-mer columns, found {len(feature_columns)}")
    frame[args.id_column] = frame[args.id_column].astype(str)
    base_ids = set(read_ids(args.base_ids))
    base = frame[frame[args.id_column].isin(base_ids)]
    if args.conditional and len(base) != len(base_ids):
        raise ValueError(f"Only {len(base):,}/{len(base_ids):,} base IDs were found in the k-mer table")
    pool = frame[~frame[args.id_column].isin(base_ids)].copy()
    if args.candidate_ids:
        candidates_requested = set(read_ids(args.candidate_ids))
        candidates = pool[pool[args.id_column].isin(candidates_requested)].copy()
    else:
        candidates = pool.copy()

    pool_features = pool[feature_columns].to_numpy(np.float32)
    assignments, _ = assign_lsh_buckets(pool_features, bits=args.lsh_bits, seed=args.lsh_seed)
    labels, ground, weights = bucket_centroids(pool_features, assignments)
    initial = initialize_conditional_coverage(
        ground,
        base[feature_columns].to_numpy(np.float32) if args.conditional else None,
    )
    candidate_features = candidates[feature_columns].to_numpy(np.float32)
    if args.method == "facility":
        indices, diagnostics = greedy_facility_location(
            candidate_features, ground, args.select_rows,
            ground_weights=weights, initial_coverage=initial,
        )
    else:
        if not args.variance_table:
            parser.error("--variance-table is required for hybrid selection")
        variance = pd.read_csv(args.variance_table, sep="\t", usecols=[args.id_column, args.variance_column])
        variance[args.id_column] = variance[args.id_column].astype(str)
        candidates = candidates.merge(variance, on=args.id_column, how="inner", validate="one_to_one")
        candidate_features = candidates[feature_columns].to_numpy(np.float32)
        indices, diagnostics = greedy_hybrid_selection(
            candidate_features,
            candidates[args.variance_column].to_numpy(float),
            ground,
            args.select_rows,
            lambda_uncertainty=args.lambda_uncertainty,
            ground_weights=weights,
            initial_coverage=initial,
        )
    selected = candidates.iloc[indices].reset_index(drop=True)
    details = pd.DataFrame(diagnostics)
    selected = pd.concat([details.drop(columns="candidate_index"), selected], axis=1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selection.tsv", sep="\t", index=False)
    write_ids(selected[args.id_column].tolist(), output / "selected_ids.txt")
    config = {**vars(args), "ground_buckets": len(labels), "ground_weight": float(weights.sum()), "candidate_rows": len(candidates)}
    (output / "selection_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

