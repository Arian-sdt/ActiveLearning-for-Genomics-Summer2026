"""Random, uncertainty, facility-location, hybrid, family, and LSH selection.

Purpose: provide auditable acquisition methods with common result contracts.
How to use: import selectors or run ``debour-select --help``.
"""

from .facility_location import greedy_facility_location
from .families import family_centroids
from .hybrid import greedy_hybrid_selection

__all__ = ["family_centroids", "greedy_facility_location", "greedy_hybrid_selection"]
