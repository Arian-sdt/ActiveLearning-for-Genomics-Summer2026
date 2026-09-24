"""Purpose: verify paired bootstrap output shapes and superior-model direction.
How to run: ``python3 -m pytest tests/test_bootstrap.py``.
"""

import numpy as np

from debour.evaluation.bootstrap import paired_bootstrap_pearson


def test_paired_bootstrap() -> None:
    observed = np.arange(20, dtype=float)
    predictions = {"good": observed.copy(), "bad": observed[::-1].copy()}
    summary, pairwise, distributions = paired_bootstrap_pearson(observed, predictions, samples=50, seed=1)
    assert len(summary) == 2
    assert len(pairwise) == 1
    assert distributions.shape == (50, 3)
    assert summary.set_index("model").loc["good", "observed_pearsonr"] > 0.99

