"""Column contracts for DNA sequence/activity tables.

Purpose:
    Avoid scattering the historical Table S2 names (IDs, sequence,
    K562_log2FC) throughout training and analysis code.

How to use:
    ``schema = SequenceTableSchema()`` or pass custom column names to workflow
    arguments when adapting another table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceTableSchema:
    """Names of the identifier, sequence, and regression-target columns."""

    id_column: str = "IDs"
    sequence_column: str = "sequence"
    target_column: str = "K562_log2FC"

    @property
    def required_columns(self) -> tuple[str, str, str]:
        return self.id_column, self.sequence_column, self.target_column


DEFAULT_SCHEMA = SequenceTableSchema()

