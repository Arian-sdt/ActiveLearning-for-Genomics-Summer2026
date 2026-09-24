"""Chunked sequence-table and ID-file input/output helpers.

Purpose:
    Apply identical cleaning rules to large Table S2 and synthetic tables while
    avoiding accidental leakage, duplicate IDs, or silent invalid targets.

How to use:
    Import ``load_sequence_table`` for moderate tables or ``iter_clean_chunks``
    for streaming workflows. This file is a library module, not a standalone CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import DEFAULT_SCHEMA, SequenceTableSchema
from .sequences import is_valid_sequence


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_ids(ids: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, ids)) + "\n")


def clean_sequence_frame(
    frame: pd.DataFrame,
    schema: SequenceTableSchema = DEFAULT_SCHEMA,
    *,
    require_target: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    required = [schema.id_column, schema.sequence_column]
    if require_target:
        required.append(schema.target_column)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = frame.dropna(subset=required).copy()
    cleaned[schema.id_column] = cleaned[schema.id_column].astype(str)
    cleaned[schema.sequence_column] = cleaned[schema.sequence_column].astype(str).str.upper()
    if require_target:
        cleaned[schema.target_column] = pd.to_numeric(cleaned[schema.target_column], errors="coerce")
        cleaned = cleaned.dropna(subset=[schema.target_column])
    cleaned = cleaned[cleaned[schema.sequence_column].map(is_valid_sequence)]
    if deduplicate:
        cleaned = cleaned.drop_duplicates(schema.id_column, keep="first")
    return cleaned.reset_index(drop=True)


def iter_clean_chunks(
    path: str | Path,
    schema: SequenceTableSchema = DEFAULT_SCHEMA,
    *,
    chunksize: int = 250_000,
    require_target: bool = True,
) -> Iterator[pd.DataFrame]:
    usecols = [schema.id_column, schema.sequence_column]
    if require_target:
        usecols.append(schema.target_column)
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False):
        yield clean_sequence_frame(chunk, schema, require_target=require_target)


def load_sequence_table(
    path: str | Path,
    schema: SequenceTableSchema = DEFAULT_SCHEMA,
    *,
    require_target: bool = True,
    include_ids: set[str] | None = None,
    exclude_ids: set[str] | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    seen: set[str] = set()
    for chunk in iter_clean_chunks(path, schema, chunksize=chunksize, require_target=require_target):
        id_values = chunk[schema.id_column]
        if include_ids is not None:
            chunk = chunk[id_values.isin(include_ids)]
        if exclude_ids:
            chunk = chunk[~chunk[schema.id_column].isin(exclude_ids)]
        chunk = chunk[~chunk[schema.id_column].isin(seen)]
        seen.update(chunk[schema.id_column])
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        raise ValueError(f"No valid rows loaded from {path}")
    result = pd.concat(pieces, ignore_index=True)
    if include_ids is not None:
        missing = include_ids - set(result[schema.id_column])
        if missing:
            raise ValueError(f"{len(missing):,} requested IDs were not found")
    return result

