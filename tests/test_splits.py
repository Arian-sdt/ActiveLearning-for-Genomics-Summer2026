"""Purpose: verify deterministic, disjoint split and leakage helpers.
How to run: ``python3 -m pytest tests/test_splits.py``.
"""

import numpy as np
import pytest

from debour.data.splits import assert_disjoint, split_indices


def test_split_reproducible_and_disjoint() -> None:
    train_a, val_a = split_indices(100, 0.1, 42)
    train_b, val_b = split_indices(100, 0.1, 42)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert set(train_a).isdisjoint(val_a)


def test_overlap_raises() -> None:
    with pytest.raises(ValueError):
        assert_disjoint(("train", {"a"}), ("test", {"a"}))

