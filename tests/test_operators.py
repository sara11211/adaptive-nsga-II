import pytest
import numpy as np
from nsga2.core.individual import Individual
from nsga2.core.selection import tournament_selection
from nsga2.core.crossover import uniform_crossover
from nsga2.core.mutation import discrete_mutation


class TestTournamentSelection:
    def _make_pop(self, ranks_and_cd):
        pop = []
        for rank, cd in ranks_and_cd:
            ind = Individual(np.array([0]), n_objectives=2)
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
        wins_high, wins_low = 0, 0
        for _ in range(100):
            w = tournament_selection(pop)
            if w.crowding_distance == 50.0:
                wins_high += 1
            else:
                wins_low += 1
        assert wins_high > wins_low


class TestUniformCrossover:
    def test_shape_preserved(self):
        """Children have the same shape as parents."""
        p1 = np.array([0, 1, 0])
        p2 = np.array([1, 0, 1])
        c1, c2 = uniform_crossover(p1, p2, prob_crossover=1.0)
        assert c1.shape == p1.shape
        assert c2.shape == p2.shape

    def test_no_crossover_respects_probability(self):
        """When prob=0, children == parents."""
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        # Run without crossing (using prob=0)
        c1, c2 = uniform_crossover(p1, p2, prob_crossover=0.0)
        np.testing.assert_array_equal(c1, p1)
        np.testing.assert_array_equal(c2, p2)

    def test_genes_are_swapped(self):
        """With prob=1.0, genes are mixed between parents."""
        p1 = np.zeros(20, dtype=int)
        p2 = np.ones(20, dtype=int)
        # Force crossover
        c1, c2 = uniform_crossover(p1, p2, prob_crossover=1.0)
        
        # Children should be a mix of 0s and 1s 
        assert not np.array_equal(c1, p1), "Child 1 was not modified"
        assert not np.array_equal(c2, p2), "Child 2 was not modified"


class TestDiscreteMutation:
    def test_shape_preserved(self):
        """Mutation preserves array shape."""
        x = np.array([0, 1, 0])
        num_choices = np.array([3, 3, 3])
        result = discrete_mutation(x.copy(), prob_mutation=1.0, num_choices=num_choices)
        assert result.shape == x.shape

    def test_within_valid_options(self):
        """Mutated values stay within valid choices (0 to k-1)."""
        # Gene 0: 2 choices (0, 1)
        # Gene 1: 5 choices (0..4)
        # Gene 2: 10 choices (0..9)
        num_choices = np.array([2, 5, 10])
        
        for _ in range(30):
            x = np.array([0, 0, 0])
            discrete_mutation(x, prob_mutation=1.0, num_choices=num_choices)
            
            assert x[0] < 2, "Gene 0 exceeded bounds"
            assert x[1] < 5, "Gene 1 exceeded bounds"
            assert x[2] < 10, "Gene 2 exceeded bounds"

    def test_no_mutation(self):
        """With prob_mutation=0, nothing changes."""
        x = np.array([1, 2])
        original = x.copy()
        num_choices = np.array([5, 5])
        discrete_mutation(x, prob_mutation=0.0, num_choices=num_choices)
        np.testing.assert_array_equal(x, original)