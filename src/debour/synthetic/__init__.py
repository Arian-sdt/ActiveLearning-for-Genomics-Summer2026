"""Prototype selection, mutation families, and oracle labeling.

Purpose: build controlled redundant sequence pools for acquisition benchmarks.
How to use: import helpers or run ``workflows/run_synthetic_benchmark.py``.
"""

from .mutations import generate_mutated_families
from .prototypes import farthest_first

__all__ = ["farthest_first", "generate_mutated_families"]

