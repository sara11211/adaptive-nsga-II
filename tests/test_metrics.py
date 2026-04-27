import pytest
import numpy as np
from nsga2.metrics.gd import generational_distance
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.hv import hypervolume
from nsga2.metrics.spread import spread


class TestGenerationalDistance:
    def test_perfect_convergence(self):
        """GD = 0 when obtained front is exactly the true front."""
        true_pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        gd = generational_distance(true_pf, true_pf)
        assert gd < 1e-10

    def test_worse_convergence(self):
        """Shifted front should have positive GD."""
        true_pf = np.array([[0.0, 1.0], [1.0, 0.0]])
        obtained = np.array([[0.1, 1.1], [1.1, 0.1]])
        gd = generational_distance(obtained, true_pf)
        assert gd > 0


class TestInvertedGenerationalDistance:
    def test_perfect(self):
        true_pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        igd = inverted_generational_distance(true_pf, true_pf)
        assert igd < 1e-10

    def test_far_solutions(self):
        true_pf = np.array([[0.0, 1.0], [1.0, 0.0]])
        obtained = np.array([[5.0, 5.0]])  # very far
        igd = inverted_generational_distance(obtained, true_pf)
        assert igd > 1.0


class TestHypervolume:
    def test_zero_solutions(self):
        hv = hypervolume(np.empty((0, 2)), np.array([1.0, 1.0]))
        assert hv == 0.0

    def test_positive_for_valid_front(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        ref = np.array([2.0, 2.0])
        hv = hypervolume(front, ref)
        assert hv > 0

    def test_larger_front_larger_hv(self):
        ref = np.array([2.0, 2.0])
        hv_small = hypervolume(np.array([[0.5, 0.5]]), ref)
        hv_large = hypervolume(np.array([[0.0, 1.0], [1.0, 0.0]]), ref)
        assert hv_large > hv_small


class TestSpread:
    def test_perfect_spread(self):
        """Uniformly spaced obtained front should have good spread."""
        # Obtained: 50 uniformly spaced points on f2 = 1 - f1
        f1 = np.linspace(0.01, 0.99, 50)
        f2 = 1.0 - f1
        front = np.column_stack([f1, f2])
        # True front: slightly wider range
        true_f1 = np.linspace(0.0, 1.0, 100)
        true_f2 = 1.0 - true_f1
        true_front = np.column_stack([true_f1, true_f2])
        sp = spread(front, true_front)
        assert np.isfinite(sp)

    def test_single_point(self):
        sp = spread(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
        assert sp == float("inf")
