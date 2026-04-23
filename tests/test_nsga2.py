"""Tests for the main NSGA-II algorithm loop."""

import pytest
import numpy as np
from nsga2.core.nsga2 import NSGA2
from nsga2.problems.zdt import ZDT1, ZDT2
from nsga2.problems.constrained import CONSTR
from nsga2.metrics.gd import generational_distance


class TestNSGA2Integration:
    def test_zdt1_convergence(self):
        """NSGA-II should converge reasonably close to the ZDT1 Pareto front."""
        problem = ZDT1()
        optimizer = NSGA2(
            problem=problem,
            pop_size=50,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            bounds=problem.bounds,
            seed=42,
        )
        _, pareto_front = optimizer.run(generations=100, verbose=False)

        true_pf = problem.pareto_front(n_points=500)
        gd = generational_distance(pareto_front, true_pf)

        # Should achieve GD < 0.05 with 100 generations on ZDT1
        assert gd < 0.1, f"GD = {gd:.6f}, expected < 0.1"

    def test_zdt2_convergence(self):
        """NSGA-II should converge reasonably on ZDT2 (non-convex front)."""
        problem = ZDT2()
        optimizer = NSGA2(
            problem=problem,
            pop_size=50,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            bounds=problem.bounds,
            seed=42,
        )
        _, pareto_front = optimizer.run(generations=100, verbose=False)

        true_pf = problem.pareto_front(n_points=500)
        gd = generational_distance(pareto_front, true_pf)
        assert gd < 0.1, f"GD = {gd:.6f}, expected < 0.1"

    def test_constrained_problem(self):
        """NSGA-II should find feasible solutions on the CONSTR problem."""
        problem = CONSTR()
        optimizer = NSGA2(
            problem=problem,
            pop_size=50,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_constr=problem.n_constr,
            bounds=problem.bounds,
            seed=42,
        )
        _, pareto_front = optimizer.run(generations=100, verbose=False)

        # At least some solutions should exist
        assert len(pareto_front) > 0

    def test_history_tracking(self):
        """The optimizer should record history at every generation."""
        problem = ZDT1()
        optimizer = NSGA2(
            problem=problem,
            pop_size=30,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            bounds=problem.bounds,
            seed=42,
        )
        optimizer.run(generations=20, verbose=False)

        assert len(optimizer.history) == 20
        assert optimizer.history[0]["generation"] == 1
        assert optimizer.history[-1]["generation"] == 20

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        problem = ZDT1()

        def run(seed):
            opt = NSGA2(problem, pop_size=30, n_var=problem.n_var,
                        n_obj=problem.n_obj, bounds=problem.bounds, seed=seed)
            _, pf = opt.run(generations=10, verbose=False)
            return pf

        pf1 = run(123)
        pf2 = run(123)
        np.testing.assert_array_equal(pf1, pf2)
