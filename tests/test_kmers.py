"""Purpose: verify independent probability normalization of k-mer blocks.
How to run: ``python3 -m pytest tests/test_kmers.py``.
"""

import numpy as np

from debour.features.kmers import kmer_distribution, kmer_feature_vector


def test_kmer_blocks_sum_to_one() -> None:
    vector = kmer_feature_vector("ACGTACGT")
    assert vector.shape == (84,)
    for start, stop in ((0, 4), (4, 20), (20, 84)):
        assert np.isclose(vector[start:stop].sum(), 1.0)


def test_overlapping_counts() -> None:
    distribution = kmer_distribution("AAAA", 2)
    assert np.isclose(distribution[0], 1.0)

