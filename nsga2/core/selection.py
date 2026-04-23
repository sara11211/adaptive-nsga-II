"""
Binary tournament selection for NSGA-II.

Uses the crowded-comparison operator from Section III-B:
  1. Prefer lower (better) rank.
  2. If ranks are equal, prefer larger crowding distance (sparser region).
"""

from __future__ import annotations
import random
from typing import List
from nsga2.core.individual import Individual


def tournament_selection(population: List[Individual]) -> Individual:
    """Select one individual via binary tournament using crowded-comparison.

    Two individuals are picked uniformly at random. The winner is chosen
    by the crowded-comparison operator (lower rank wins; tie-break by
    larger crowding distance).

    Args:
        population: The current population (must have rank and
                    crowding_distance assigned).

    Returns:
        The winning Individual.
    """
    i, j = random.sample(range(len(population)), 2)
    a, b = population[i], population[j]

    # Crowded-comparison: prefer lower rank
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b

    # Same rank: prefer larger crowding distance
    if a.crowding_distance > b.crowding_distance:
        return a
    return b
