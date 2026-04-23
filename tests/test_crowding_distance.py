"""Tests for crowding distance assignment."""

import pytest
import numpy as np
from nsga2.core.individual import Individual
from nsga2.core.crowding_distance import crowding_distance_assignment


def _make_front(objectives: list) -> list:
    """Create a front of evaluated individuals."""
    front = []
    for obj in objectives:
        ind = Individual(np.array([0.0]), n_objectives=len(obj))
        ind.objectives = np.array(obj, dtype=float)
        front.append(ind)
    return front


class TestCrowdingDistance:
    def test_boundary_infinite(self):
        """Boundary solutions (min and max) must have infinite crowding distance."""
        front = _make_front([
            (0.0, 1.0),
            (0.5, 0.5),
            (1.0, 0.0),
        ])
        crowding_distance_assignment(front)

        # The two boundary solutions should have infinite distance
        sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
        assert sorted_front[0].crowding_distance == float("inf")
        assert sorted_front[1].crowding_distance == float("inf")

    def test_single_objective(self):
        """With identical second objective, crowding is based on first only."""
        front = _make_front([
            (0.0, 0.0),
            (0.3, 0.0),
            (0.6, 0.0),
            (1.0, 0.0),
        ])
        crowding_distance_assignment(front)

        # Middle two should have finite, positive distance
        mid = [ind for ind in front if ind.crowding_distance != float("inf")]
        for ind in mid:
            assert ind.crowding_distance > 0

    def test_two_solutions(self):
        """Two solutions both get infinite distance."""
        front = _make_front([(0.0, 1.0), (1.0, 0.0)])
        crowding_distance_assignment(front)
        for ind in front:
            assert ind.crowding_distance == float("inf")

    def test_single_solution(self):
        """Single solution gets infinite distance."""
        front = _make_front([(0.5, 0.5)])
        crowding_distance_assignment(front)
        assert front[0].crowding_distance == float("inf")

    def test_identical_objectives(self):
        """When all objectives are identical, crowding distance is zero for interiors."""
        front = _make_front([(1.0, 1.0)] * 5)
        crowding_distance_assignment(front)
        # All are boundaries or have zero range contribution
        interior = [ind for ind in front if ind.crowding_distance != float("inf")]
        for ind in interior:
            assert ind.crowding_distance == 0.0
