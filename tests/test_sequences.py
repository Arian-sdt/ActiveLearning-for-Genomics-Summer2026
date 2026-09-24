"""Purpose: verify sequence normalization and A,G,C,T encoding contracts.
How to run: ``python3 -m pytest tests/test_sequences.py``.
"""

import numpy as np

from debour.data.sequences import normalize_sequence, one_hot_encode


def test_normalization_is_exact_length() -> None:
    assert normalize_sequence("AC", 4) == "NNAC"
    assert normalize_sequence("AACCGG", 4) == "ACCG"
    assert len(normalize_sequence("AACCG", 4)) == 4


def test_channel_order_and_unknown() -> None:
    encoded = one_hot_encode("AGCTN", as_tensor=False)
    assert encoded.shape == (4, 5)
    np.testing.assert_array_equal(encoded[:, :4], np.eye(4, dtype=np.float32))
    assert encoded[:, 4].sum() == 0

