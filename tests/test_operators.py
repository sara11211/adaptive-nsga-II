"""Tests for selection, crossover, and mutation operators."""

import pytest
import numpy as np
from nsga2.core.individual import Individual
from nsga2.core.selection import tournament_selection
from nsga2.core.crossover import sbx_crossover
from nsga2.core.mutation import polynomial_mutation


class TestTournamentSelection:
    def _make_pop(self, ranks_and_cd):
        pop = []
        for rank, cd in ranks_and_cd:
            ind = Individual(np.array([0.0]), n_objectives=2)
            ind.rank = rank
            ind.crowding_distance = cd
            pop.append(ind)
        return pop

    def test_lower_rank_wins(self):
        """Individual with lower rank should win."""
        pop = self._make_pop([(0, 1.0), (1, 100.0)])
        winner = tournament_selection(pop)
        assert winner.rank == 0

    def test_same_rank_crowding_wins(self):
        """When ranks are equal, higher crowding distance wins."""
        pop = self._make_pop([(0, 50.0), (0, 10.0)])
        # Run many times — winner should always be the one with cd=50
        wins_a, wins_b = 0, 0
        for _ in range(100):
            w = tournament_selection(pop)
            if w.crowding_distance == 50.0:
                wins_a += 1
            else:
                wins_b += 1
        assert wins_a > wins_b


class TestSBXCrossover:
    def test_shape_preserved(self):
        """Children have the same shape as parents."""
        p1 = np.array([0.2, 0.5, 0.8])
        p2 = np.array([0.4, 0.6, 0.9])
        bounds = np.array([[0.0, 1.0]] * 3)
        c1, c2 = sbx_crossover(p1, p2, bounds=bounds)
        assert c1.shape == p1.shape
        assert c2.shape == p2.shape

    def test_within_bounds(self):
        """Children respect the variable bounds."""
        bounds = np.array([[0.0, 1.0]] * 10)
        for _ in range(50):
            p1 = np.random.rand(10)
            p2 = np.random.rand(10)
            c1, c2 = sbx_crossover(p1, p2, bounds=bounds)
            assert np.all(c1 >= bounds[:, 0]) and np.all(c1 <= bounds[:, 1])
            assert np.all(c2 >= bounds[:, 0]) and np.all(c2 <= bounds[:, 1])

    def test_no_crossover_respects_probability(self):
        """When random exceeds prob_crossover, children == parents."""
        p1 = np.array([0.1, 0.2])
        p2 = np.array([0.9, 0.8])
        # Run without crossing (using prob=0)
        c1, c2 = sbx_crossover(p1, p2, prob_crossover=0.0)
        np.testing.assert_array_equal(c1, p1)
        np.testing.assert_array_equal(c2, p2)


class TestPolynomialMutation:
    def test_shape_preserved(self):
        """Mutation preserves array shape."""
        x = np.array([0.5, 0.5, 0.5])
        bounds = np.array([[0.0, 1.0]] * 3)
        result = polynomial_mutation(x.copy(), prob_mutation=1.0, bounds=bounds)
        assert result.shape == x.shape

    def test_within_bounds(self):
        """Mutated values stay within bounds."""
        bounds = np.array([[0.0, 1.0]] * 20)
        for _ in range(30):
            x = np.random.rand(20)
            polynomial_mutation(x, prob_mutation=1.0, eta_m=20.0, bounds=bounds)
            assert np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1])

    def test_no_mutation(self):
        """With prob_mutation=0, nothing changes."""
        x = np.array([0.3, 0.7])
        original = x.copy()
        bounds = np.array([[0.0, 1.0]] * 2)
        polynomial_mutation(x, prob_mutation=0.0, bounds=bounds)
        np.testing.assert_array_equal(x, original)
