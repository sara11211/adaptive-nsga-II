"""Tests for fast nondominated sorting."""

import pytest
import numpy as np
from nsga2.core.individual import Individual
from nsga2.core.nondominated_sort import fast_non_dominated_sort, _dominates


class TestDominates:
    """Unit tests for the _dominates helper."""

    def test_dominates_all_better(self):
        """a strictly dominates b on all objectives."""
        a = np.array([1.0, 1.0])
        b = np.array([2.0, 2.0])
        assert _dominates(a, b) is True

    def test_dominates_partial(self):
        """a dominates b: better on one, equal on another."""
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 2.0])
        assert _dominates(a, b) is True

    def test_does_not_dominates(self):
        """a does not dominate b: better on one, worse on another."""
        a = np.array([1.0, 3.0])
        b = np.array([2.0, 1.0])
        assert _dominates(a, b) is False

    def test_equal(self):
        """Identical objectives: neither dominates."""
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        assert _dominates(a, b) is False

    def test_worse(self):
        """a is strictly worse than b."""
        a = np.array([3.0, 3.0])
        b = np.array([1.0, 1.0])
        assert _dominates(a, b) is False


class TestFastNondominatedSort:
    """Tests for fast_non_dominated_sort."""

    def _make_pop(self, objectives: list) -> list:
        """Helper: create a population from a list of objective tuples."""
        pop = []
        for obj in objectives:
            ind = Individual(np.array([0.0]), n_objectives=len(obj))
            ind.objectives = np.array(obj, dtype=float)
            pop.append(ind)
        return pop

    def test_single_front(self):
        """All solutions on one front (incomparable)."""
        pop = self._make_pop([
            (1.0, 5.0),
            (5.0, 1.0),
            (3.0, 3.0),
        ])
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self):
        """Clear two-front structure."""
        pop = self._make_pop([
            (1.0, 1.0),   # front 0 (dominates both below)
            (2.0, 2.0),   # front 1 (dominates the one below)
            (3.0, 3.0),   # front 2
        ])
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 3
        assert len(fronts[0]) == 1
        assert len(fronts[1]) == 1
        assert len(fronts[2]) == 1

    def test_rank_assignment(self):
        """Each individual gets the correct rank."""
        pop = self._make_pop([
            (1.0, 4.0),   # rank 0
            (4.0, 1.0),   # rank 0
            (2.0, 2.0),   # rank 1
            (3.0, 3.0),   # rank 2
        ])
        fronts = fast_non_dominated_sort(pop)
        for i, front in enumerate(fronts):
            for ind in front:
                assert ind.rank == i

    def test_empty_population(self):
        """Empty input returns empty list."""
        fronts = fast_non_dominated_sort([])
        assert fronts == []

    def test_all_identical(self):
        """All identical solutions belong to the same front."""
        pop = self._make_pop([
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
        ])
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_constrained_dominance(self):
        """Feasible solution dominates infeasible one."""
        feasible = Individual(np.array([0.0]), n_objectives=2, n_constraints=1)
        feasible.objectives = np.array([3.0, 3.0])
        feasible.constraints = np.array([1.0])  # satisfied

        infeasible = Individual(np.array([0.0]), n_objectives=2, n_constraints=1)
        infeasible.objectives = np.array([1.0, 1.0])  # better objectives
        infeasible.constraints = np.array([-1.0])  # violated

        fronts = fast_non_dominated_sort([feasible, infeasible])
        assert len(fronts[0]) == 1
        assert fronts[0][0] is feasible
