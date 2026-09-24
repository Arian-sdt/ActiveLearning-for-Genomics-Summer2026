"""Count and normalize overlapping DNA k-mers for k=1,2,3.

Purpose:
    Convert sequences into 84 interpretable probability features: 4 one-mers,
    16 two-mers, and 64 three-mers. Each k block is normalized independently.

How to use:
    ``python3 -m debour.features.kmers input.tsv output.tsv`` where the input has
    IDs and sequence columns. Import ``kmer_feature_matrix`` in other modules.
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from debour.data.schema import DEFAULT_SCHEMA

ALPHABET = "ACGT"


def enumerate_kmers(k: int) -> list[str]:
    return ["".join(chars) for chars in product(ALPHABET, repeat=k)]


def feature_layout(ks: tuple[int, ...] = (1, 2, 3)) -> dict[int, list[str]]:
    return {k: enumerate_kmers(k) for k in ks}


def kmer_distribution(sequence: str, k: int) -> np.ndarray:
    kmers = enumerate_kmers(k)
    lookup = {word: index for index, word in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    sequence = str(sequence).upper()
    for start in range(max(0, len(sequence) - k + 1)):
        word = sequence[start : start + k]
        index = lookup.get(word)
        if index is not None:
            counts[index] += 1
    total = counts.sum()
    return counts / total if total else counts


def kmer_feature_vector(sequence: str, ks: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    return np.concatenate([kmer_distribution(sequence, k) for k in ks])


def kmer_feature_matrix(sequences: list[str], ks: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    return np.vstack([kmer_feature_vector(sequence, ks) for sequence in sequences]).astype(np.float32)


def feature_names(ks: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    return [f"k{k}_{word}" for k in ks for word in enumerate_kmers(k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--id-column", default=DEFAULT_SCHEMA.id_column)
    parser.add_argument("--sequence-column", default=DEFAULT_SCHEMA.sequence_column)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep="\t", usecols=[args.id_column, args.sequence_column])
    features = np.vstack([
        kmer_feature_vector(sequence)
        for sequence in tqdm(frame[args.sequence_column], desc="Counting normalized k-mers", unit="sequence")
    ])
    output = pd.DataFrame(features, columns=feature_names())
    output.insert(0, args.id_column, frame[args.id_column].astype(str).to_numpy())
    output.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()

