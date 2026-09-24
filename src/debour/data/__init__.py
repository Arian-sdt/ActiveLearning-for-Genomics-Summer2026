"""Sequence-table loading, validation, normalization, and deterministic splits.

Purpose: centralize data contracts shared by every workflow.
How to use: import from ``debour.data`` after installing the package.
"""

__all__ = ["DEFAULT_SCHEMA", "SequenceTableSchema", "normalize_sequence", "one_hot_encode"]


def __getattr__(name: str):
    """Load public objects only when requested, keeping ``python -m`` warning-free."""

    if name in {"DEFAULT_SCHEMA", "SequenceTableSchema"}:
        from .schema import DEFAULT_SCHEMA, SequenceTableSchema

        return {"DEFAULT_SCHEMA": DEFAULT_SCHEMA, "SequenceTableSchema": SequenceTableSchema}[name]
    if name in {"normalize_sequence", "one_hot_encode"}:
        from .sequences import normalize_sequence, one_hot_encode

        return {"normalize_sequence": normalize_sequence, "one_hot_encode": one_hot_encode}[name]
    raise AttributeError(name)
