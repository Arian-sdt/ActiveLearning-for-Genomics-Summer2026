"""Shared selection result validation and persistence.

Purpose: ensure every acquisition method saves ordered IDs, scores, and config.
How to use: call ``save_selection`` from a selector or workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from debour.data.io import write_ids


def save_selection(frame: pd.DataFrame, output_dir: str | Path, config: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "IDs" not in frame or frame["IDs"].duplicated().any():
        raise ValueError("Selection must contain one unique IDs column")
    frame.to_csv(output_dir / "selection.tsv", sep="\t", index=False)
    write_ids(frame["IDs"].astype(str).tolist(), output_dir / "selected_ids.txt")
    (output_dir / "selection_config.json").write_text(json.dumps(config, indent=2, default=str))

