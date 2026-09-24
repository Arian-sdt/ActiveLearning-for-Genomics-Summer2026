"""Greedy farthest-first selection of diverse prototype sequences.

Purpose:
    Choose parent sequences whose nearest selected-prototype weighted JS distance
    is maximized, creating distinct starting families for redundancy experiments.

How to use:
    Call ``farthest_first(features, rows=1000, seed=42)``. Features must contain
    normalized k=1,2,3 distributions in the standard 84-column order.
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from debour.features.jensen_shannon import weighted_js_distance


def farthest_first(features: np.ndarray, rows: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if rows > len(features):
        raise ValueError("rows exceeds candidate count")
    first = int(np.random.default_rng(seed).integers(len(features)))
    selected = [first]
    nearest = weighted_js_distance(features, np.broadcast_to(features[first], features.shape))
    nearest[first] = -np.inf
    for _ in tqdm(range(1, rows), desc="Selecting farthest-first prototypes", unit="prototype"):
        index = int(np.argmax(nearest))
        selected.append(index)
        distance = weighted_js_distance(features, np.broadcast_to(features[index], features.shape))
        nearest = np.minimum(nearest, distance)
        nearest[selected] = -np.inf
    return np.asarray(selected), nearest

