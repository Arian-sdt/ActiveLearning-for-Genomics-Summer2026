#!/usr/bin/env python3
"""Select synthetic sequences using their known mutation-family structure.

Purpose:
    Run the synthetic-data facility-location selection used in this project
    without LSH. The workflow:

    1. excludes the exact shared base training IDs;
    2. treats each known parent mutation family as one ground-set group;
    3. computes one normalized k-mer centroid per remaining family;
    4. weights each centroid by its remaining family size;
    5. uses weighted Jensen-Shannon similarity (1 minus distance);
    6. runs lazy-greedy facility selection and saves 2,000 actual sequence IDs
       in selection order; or runs greedy hybrid selection when requested;
    7. records marginal gain and cumulative normalized coverage every round.

    Unconditional mode starts family coverage at zero. Conditional mode first
    measures how well every family centroid is already covered by the base set,
    then selects sequences that add coverage beyond that base. Hybrid mode adds
    min-max-normalized ensemble variance using ``lambda_uncertainty``; facility
    gain is divided by total family weight before the two terms are combined.

How to run:
    Unconditional facility selection::

        python3 workflows/select_family_facility.py \
          --synthetic-table mutated_copies.tsv \
          --kmer-table synthetic_kmers.tsv \
          --base-ids base_10000_ids.txt \
          --select-rows 2000 --output-dir outputs/family_unconditional_2k

    Add ``--conditional`` for base-aware selection. For hybrid selection, add
    ``--method hybrid --variance-table predictions.tsv
    --lambda-uncertainty 0.5``. Candidate IDs may optionally restrict the real
    rows eligible for selection, while all remaining families still define the
    representative ground set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from debour.data.io import read_ids, write_ids
from debour.features.kmers import feature_names
from debour.selection.facility_location import (
    greedy_facility_location,
    initialize_conditional_coverage,
)
from debour.selection.families import family_centroids, family_summary
from debour.selection.hybrid import greedy_hybrid_selection


def kmer_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = feature_names()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"K-mer table is missing {len(missing)} canonical feature columns")
    return columns


def validate_normalized_features(frame: pd.DataFrame, columns: list[str]) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("K-mer features must be finite and nonnegative")
    start = 0
    for size, label in zip((4, 16, 64), ("k=1", "k=2", "k=3")):
        totals = values[:, start : start + size].sum(axis=1)
        if not np.allclose(totals, 1.0, atol=1e-5):
            raise ValueError(f"{label} columns are not independently normalized to one")
        start += size


def load_joined_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(
        args.synthetic_table,
        sep="\t",
        usecols=[args.synthetic_id_column, args.family_column],
    )
    kmers = pd.read_csv(args.kmer_table, sep="\t")
    features = kmer_feature_columns(kmers)
    if args.kmer_id_column not in kmers:
        raise ValueError(f"Missing k-mer ID column: {args.kmer_id_column}")

    metadata = metadata.rename(columns={args.synthetic_id_column: "_selection_id"})
    kmers = kmers.rename(columns={args.kmer_id_column: "_selection_id"})
    metadata["_selection_id"] = metadata["_selection_id"].astype(str)
    kmers["_selection_id"] = kmers["_selection_id"].astype(str)
    if metadata["_selection_id"].duplicated().any():
        raise ValueError("Synthetic metadata contains duplicate sequence IDs")
    if kmers["_selection_id"].duplicated().any():
        raise ValueError("K-mer table contains duplicate sequence IDs")

    joined = metadata.merge(
        kmers[["_selection_id", *features]],
        on="_selection_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(metadata):
        missing = len(metadata) - len(joined)
        raise ValueError(f"K-mer table is missing {missing:,} synthetic sequence IDs")
    validate_normalized_features(joined, features)
    return joined, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-table", required=True, help="TSV containing sequence IDs and parent-family IDs.")
    parser.add_argument("--kmer-table", required=True, help="Normalized k=1,2,3 k-mer TSV for synthetic sequences.")
    parser.add_argument("--synthetic-id-column", default="synthetic_id")
    parser.add_argument("--kmer-id-column", default="IDs")
    parser.add_argument("--family-column", default="parent_id")
    parser.add_argument("--base-ids", required=True, help="Exact existing training IDs to exclude.")
    parser.add_argument("--candidate-ids", help="Optional ID file restricting selectable rows after base exclusion.")
    parser.add_argument("--method", choices=["facility", "hybrid"], default="facility")
    parser.add_argument("--conditional", action="store_true", help="Initialize coverage from the base training set.")
    parser.add_argument("--variance-table", help="TSV containing candidate ensemble variances for hybrid selection.")
    parser.add_argument("--variance-id-column", default="IDs")
    parser.add_argument("--variance-column", default="prediction_variance")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.5)
    parser.add_argument("--select-rows", type=int, default=2000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, features = load_joined_data(args)
    base_ids = set(read_ids(args.base_ids))
    found_ids = set(frame["_selection_id"])
    missing_base = base_ids - found_ids
    if missing_base:
        raise ValueError(f"Synthetic/k-mer tables are missing {len(missing_base):,} base IDs")

    base = frame[frame["_selection_id"].isin(base_ids)].copy()
    pool = frame[~frame["_selection_id"].isin(base_ids)].copy()
    if pool.empty:
        raise ValueError("No candidate-pool rows remain after excluding base IDs")

    if args.candidate_ids:
        requested = set(read_ids(args.candidate_ids))
        missing_candidates = requested - set(pool["_selection_id"])
        if missing_candidates:
            raise ValueError(f"Candidate pool is missing {len(missing_candidates):,} requested IDs")
        candidates = pool[pool["_selection_id"].isin(requested)].copy()
    else:
        candidates = pool.copy()
    if args.select_rows > len(candidates):
        raise ValueError(f"Requested {args.select_rows:,} rows from only {len(candidates):,} candidates")

    family_labels, ground, weights = family_centroids(
        pool,
        features,
        family_column=args.family_column,
    )
    initial = initialize_conditional_coverage(
        ground,
        base[features].to_numpy(np.float32) if args.conditional else None,
    )

    if args.method == "facility":
        indices, diagnostics = greedy_facility_location(
            candidates[features].to_numpy(np.float32),
            ground,
            args.select_rows,
            ground_weights=weights,
            initial_coverage=initial,
        )
    else:
        if not args.variance_table:
            raise ValueError("--variance-table is required when --method hybrid")
        variance = pd.read_csv(
            args.variance_table,
            sep="\t",
            usecols=[args.variance_id_column, args.variance_column],
        ).rename(columns={args.variance_id_column: "_selection_id"})
        variance["_selection_id"] = variance["_selection_id"].astype(str)
        if variance["_selection_id"].duplicated().any():
            raise ValueError("Variance table contains duplicate sequence IDs")
        before = len(candidates)
        candidates = candidates.merge(variance, on="_selection_id", how="inner", validate="one_to_one")
        if len(candidates) != before:
            raise ValueError(f"Variance table is missing {before - len(candidates):,} candidate IDs")
        indices, diagnostics = greedy_hybrid_selection(
            candidates[features].to_numpy(np.float32),
            candidates[args.variance_column].to_numpy(float),
            ground,
            args.select_rows,
            lambda_uncertainty=args.lambda_uncertainty,
            ground_weights=weights,
            initial_coverage=initial,
        )

    selected = candidates.iloc[indices].reset_index(drop=True)
    selected = selected.rename(columns={"_selection_id": args.synthetic_id_column})
    details = pd.DataFrame(diagnostics).drop(columns="candidate_index")
    selected = pd.concat([details, selected], axis=1)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selection.tsv", sep="\t", index=False)
    write_ids(selected[args.synthetic_id_column].tolist(), output / "selected_ids.txt")
    family_summary(
        family_labels,
        weights,
        ground,
        features,
        family_column=args.family_column,
        initial_coverage=initial,
    ).to_csv(output / "family_ground_summary.tsv", sep="\t", index=False)

    config = {
        **vars(args),
        "base_rows": len(base),
        "remaining_pool_rows": len(pool),
        "candidate_rows": len(candidates),
        "ground_families": len(family_labels),
        "ground_weight": float(weights.sum()),
        "initial_normalized_coverage": float(np.dot(weights, initial) / weights.sum()),
        "ground_representation": "known_parent_family_centroids",
        "similarity": "one_minus_weighted_js_distance_k1_0.1_k2_0.3_k3_0.6",
    }
    (output / "selection_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
