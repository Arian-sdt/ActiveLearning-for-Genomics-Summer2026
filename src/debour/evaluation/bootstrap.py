"""Paired bootstrap uncertainty for Pearson correlation comparisons.

Purpose:
    Repeatedly resample one fixed heldout set with replacement. The same sampled
    indices are used for every model in each replicate, enabling confidence
    intervals and pairwise probabilities for correlation differences.

How to use:
    ``paired_bootstrap_pearson(y, {'model_a': a, 'model_b': b}, samples=1000)``.
    A 10k heldout set with 1,000 bootstrap samples means 1,000 resamples of size
    10k, not ten samples and not 10,000 independent test sets.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import pearsonr


def paired_bootstrap_pearson(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = np.asarray(observed)
    if any(len(values) != len(observed) for values in predictions.values()):
        raise ValueError("Every prediction vector must match observed length")
    rng = np.random.default_rng(seed)
    bootstrap = {name: np.empty(samples) for name in predictions}
    for replicate in range(samples):
        indices = rng.integers(0, len(observed), size=len(observed))
        for name, values in predictions.items():
            bootstrap[name][replicate] = pearsonr(observed[indices], np.asarray(values)[indices])

    rows = []
    for name, values in predictions.items():
        distribution = bootstrap[name]
        rows.append({
            "model": name,
            "observed_pearsonr": pearsonr(observed, values),
            "bootstrap_mean": float(np.nanmean(distribution)),
            "ci_lower_95": float(np.nanpercentile(distribution, 2.5)),
            "ci_upper_95": float(np.nanpercentile(distribution, 97.5)),
        })
    pairwise = []
    for left, right in combinations(predictions, 2):
        difference = bootstrap[left] - bootstrap[right]
        pairwise.append({
            "model_a": left,
            "model_b": right,
            "mean_pearson_difference_a_minus_b": float(np.nanmean(difference)),
            "two_sided_bootstrap_p": float(2 * min(np.mean(difference <= 0), np.mean(difference >= 0))),
        })
    distributions = pd.DataFrame(bootstrap)
    distributions.insert(0, "bootstrap_replicate", range(1, samples + 1))
    return pd.DataFrame(rows), pd.DataFrame(pairwise), distributions

