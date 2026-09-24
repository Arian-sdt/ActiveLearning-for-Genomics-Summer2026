"""Normalized k-mer features and weighted Jensen-Shannon distance.

Purpose: provide the shared representation used by LSH and submodular selection.
How to use: import feature and distance functions from ``debour.features``.
"""

__all__ = ["kmer_distribution", "weighted_js_distance"]


def __getattr__(name: str):
    """Lazily expose feature helpers so their modules remain directly runnable."""

    if name == "kmer_distribution":
        from .kmers import kmer_distribution

        return kmer_distribution
    if name == "weighted_js_distance":
        from .jensen_shannon import weighted_js_distance

        return weighted_js_distance
    raise AttributeError(name)
