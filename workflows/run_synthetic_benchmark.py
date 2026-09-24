#!/usr/bin/env python3
"""Create a controlled redundant DNA pool from diverse prototypes.

Purpose:
    Select farthest-first prototypes from a source table, generate mutation
    families with known parent identity/redundancy, and save Table-like artifacts
    ready for oracle labeling and active-learning experiments.

How to run:
    ``python3 workflows/run_synthetic_benchmark.py --table TABLE
    --prototype-rows 1000 --mutation-levels 1 4 7 10 13 16 19
    --output-dir outputs/synthetic``. Add oracle labels separately after choosing
    an explicit teacher checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from debour.data.io import load_sequence_table
from debour.data.schema import DEFAULT_SCHEMA
from debour.features.kmers import feature_names, kmer_feature_matrix
from debour.synthetic.mutations import generate_mutated_families
from debour.synthetic.prototypes import farthest_first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--prototype-rows", type=int, default=1000)
    parser.add_argument("--min-copies", type=int, default=50)
    parser.add_argument("--max-copies", type=int, default=100)
    parser.add_argument("--mutation-levels", type=int, nargs="+", default=[1, 4, 7, 10, 13, 16, 19])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_sequence_table(args.table)
    features = kmer_feature_matrix(frame[DEFAULT_SCHEMA.sequence_column].tolist())
    indices, _ = farthest_first(features, args.prototype_rows, args.seed)
    prototypes = frame.iloc[indices].reset_index(drop=True)
    prototypes.to_csv(output / "prototypes.tsv", sep="\t", index=False)
    copies = generate_mutated_families(
        prototypes,
        min_copies=args.min_copies,
        max_copies=args.max_copies,
        mutation_levels=args.mutation_levels,
        seed=args.seed,
    )
    copies.to_csv(output / "mutated_copies.tsv.gz", sep="\t", index=False)
    (output / "synthetic_config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()

