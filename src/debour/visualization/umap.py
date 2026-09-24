"""Fit/reuse a UMAP embedding and plot equal-style selection overlays.

Purpose:
    Visualize where random, uncertainty, diversity, conditional diversity, and
    hybrid methods sample the same sequence feature space. Selected/background
    point sizes and alpha are explicit so panels remain visually comparable.

How to use:
    Install ``.[umap]`` then import ``fit_umap`` and ``plot_umap_facets``.
    Embeddings should be saved and reused when only plot styling changes.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_umap(features: np.ndarray, *, seed: int = 42, neighbors: int = 30, min_dist: float = 0.05) -> np.ndarray:
    try:
        import umap
    except ImportError as error:
        raise ImportError("Install UMAP support with: python3 -m pip install -e '.[umap]'") from error
    return umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, random_state=seed, low_memory=False).fit_transform(features)


def plot_umap_facets(
    embedding: np.ndarray,
    ids: list[str],
    selections: dict[str, set[str]],
    output: str | Path,
    *,
    columns: int = 4,
    background_size: float = 0.4,
    selected_size: float = 0.8,
    alpha: float = 0.55,
) -> None:
    names = ["Full data", *selections]
    rows = ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    id_array = np.asarray(ids)
    for axis, name in zip(axes.flat, names):
        axis.scatter(embedding[:, 0], embedding[:, 1], s=background_size, alpha=alpha, color="#a9b2bb", rasterized=True)
        if name != "Full data":
            mask = np.isin(id_array, list(selections[name]))
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=selected_size, alpha=alpha, color="#08306b", rasterized=True)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
