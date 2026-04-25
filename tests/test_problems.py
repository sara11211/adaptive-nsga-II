import pytest
import numpy as np
from nsga2.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


class TestZDT1:
    def setup_method(self):
        self.problem = ZDT1()

    def test_evaluate_shape(self):
        obj, con = self.problem.evaluate(np.zeros(30))
        assert obj.shape == (2,)
        assert len(con) == 0

    def test_optimal_at_x0(self):
        """When x[1:] = 0, g = 1, and we should be on the true Pareto front."""
        x = np.zeros(30)
        x[0] = 0.5
        obj, _ = self.problem.evaluate(x)
        f1, f2 = obj
        expected_f2 = 1.0 - np.sqrt(f1)
        assert abs(f2 - expected_f2) < 1e-10

    def test_pareto_front_shape(self):
        pf = self.problem.pareto_front()
        assert pf.shape == (1000, 2)
        assert np.all(pf[:, 0] >= 0) and np.all(pf[:, 0] <= 1)


class TestZDT2:
    def test_optimal_at_x0(self):
        problem = ZDT2()
        x = np.zeros(30)
        x[0] = 0.5
        obj, _ = problem.evaluate(x)
        f1, f2 = obj
        expected_f2 = 1.0 - f1 ** 2
        assert abs(f2 - expected_f2) < 1e-10


class TestZDT4:
    def test_bounds(self):
        problem = ZDT4()
        assert problem.n_var == 10
        assert problem.bounds.shape == (10, 2)
        assert problem.bounds[0, 0] == 0.0
        assert problem.bounds[0, 1] == 1.0
        assert problem.bounds[1, 0] == -5.0


class TestZDT6:
    def test_n_var(self):
        assert ZDT6().n_var == 10
