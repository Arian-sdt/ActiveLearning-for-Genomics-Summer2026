"""Plot normalized facility coverage as sequences are selected.

Purpose: compare acquisition methods using selection-time coverage diagnostics.
How to use: pass a mapping of method names to selection TSV files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_coverage(selection_files: dict[str, str | Path], output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, path in selection_files.items():
        frame = pd.read_csv(path, sep="\t")
        ax.plot(frame["selection_rank"], frame["normalized_coverage"], label=name)
    ax.set(xlabel="Number selected", ylabel="Normalized facility coverage", title="Coverage gained during selection")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
