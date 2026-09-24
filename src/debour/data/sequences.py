"""DNA normalization, validation, one-hot encoding, and GC content.

Purpose:
    Reproduce the model input contract in one location. Channels are ordered
    A,G,C,T to match the original DeBour/BHI training scripts. N maps to zeros.

How to use:
    ``normalize_sequence(seq, 200)`` then ``one_hot_encode(seq)``. Run
    ``python3 -m debour.data.sequences ACGT --length 8`` for a small check.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch

VALID_BASES = re.compile(r"^[ACGTN]+$")
BASE_TO_CHANNEL = {"A": 0, "G": 1, "C": 2, "T": 3}


def is_valid_sequence(sequence: str) -> bool:
    return bool(VALID_BASES.fullmatch(str(sequence).upper()))


def normalize_sequence(sequence: str, length: int = 200) -> str:
    """Uppercase and return exactly ``length`` bases via left pad/center crop."""

    if length <= 0:
        raise ValueError("length must be positive")
    sequence = str(sequence).upper()
    if len(sequence) < length:
        return "N" * (length - len(sequence)) + sequence
    if len(sequence) > length:
        start = (len(sequence) - length) // 2
        return sequence[start : start + length]
    return sequence


def one_hot_encode(sequence: str, *, as_tensor: bool = True) -> torch.Tensor | np.ndarray:
    """Encode a sequence as shape ``(4, length)`` in A,G,C,T channel order."""

    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    for position, base in enumerate(str(sequence).upper()):
        channel = BASE_TO_CHANNEL.get(base)
        if channel is not None:
            encoded[channel, position] = 1.0
    return torch.from_numpy(encoded) if as_tensor else encoded


def gc_content(sequence: str) -> float:
    sequence = str(sequence).upper()
    canonical = [base for base in sequence if base in "ACGT"]
    if not canonical:
        return float("nan")
    return sum(base in "GC" for base in canonical) / len(canonical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence")
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    normalized = normalize_sequence(args.sequence, args.length)
    print(normalized)
    print("shape:", tuple(one_hot_encode(normalized).shape), "GC:", gc_content(normalized))


if __name__ == "__main__":
    main()

