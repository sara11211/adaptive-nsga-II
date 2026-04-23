"""
Crowding distance assignment for NSGA-II.

Implements the density estimation metric from Section III-B of the paper.
For each front, the crowding distance of individual i is the average
side-length of the cuboid formed by its nearest neighbours along each
objective axis. Boundary solutions receive infinite distance.
"""

from __future__ import annotations
from typing import List
import numpy as np
from nsga2.core.individual import Individual


def crowding_distance_assignment(front: List[Individual]) -> None:
    """Assign crowding distance to every individual in *front* **in-place**.

    The crowding distance estimates the density of solutions surrounding
    a particular point. Larger values indicate sparser (better) regions.

    Algorithm (from the paper):
      1. Sort the front by each objective.
      2. Boundary solutions get infinite distance.
      3. Intermediate solutions get normalised gaps summed over all objectives.

    Complexity: O(MN log N) due to sorting, where M = objectives, N = front size.

    Args:
        front: List of evaluated Individuals belonging to the same nondominated front.
    """
    n = len(front)
    if n <= 2:
        # With 0-2 individuals everyone gets infinite distance
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    n_obj = len(front[0].objectives)

    # Reset distances
    for ind in front:
        ind.crowding_distance = 0.0

    # Accumulate crowding distance per objective
    for m in range(n_obj):
        # Sort by objective m
        front.sort(key=lambda ind: ind.objectives[m])

        # Boundary solutions always get infinite distance
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        # Objective range for normalisation
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min

        # If all values are identical, skip this objective (no contribution)
        if obj_range < 1e-12:
            continue

        # Accumulate normalised gap for intermediate solutions
        for i in range(1, n - 1):
            if front[i].crowding_distance == float("inf"):
                # Already a boundary in another objective — keep inf
                continue
            gap = front[i + 1].objectives[m] - front[i - 1].objectives[m]
            front[i].crowding_distance += gap / obj_range
