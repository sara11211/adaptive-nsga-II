import pytest
import numpy as np
from nsga2.core.nsga2 import NSGA2
from nsga2.problems.discrete import OneZeroMax

class TestNSGA2Integration:
    
    def test_discrete_convergence(self):
        """NSGA-II should find a diverse set of solutions on OneZeroMax."""
        problem = OneZeroMax(n_var=30, max_val=1)
        
        optimizer = NSGA2(
            problem=problem,
            pop_size=50,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            num_choices=problem.num_choices, 
            seed=42,
        )
        
        pareto_set, pareto_front = optimizer.run(generations=100, verbose=False)

        # 1. Check that we found solutions
        assert len(pareto_set) > 0, "No solutions found"
        
        # 2. Check variable integrity (must be integers 0 or 1)
        assert np.all(pareto_set >= 0)
        assert np.all(pareto_set <= 1)
        # Check they are actually integers
        assert np.all(pareto_set == pareto_set.astype(int))

        # 3. Check diversity (OneZeroMax should produce a spread of sums)
        # If converged correctly, we should have solutions with sum_x near 0 and near 30
        sums = np.sum(pareto_set, axis=1)
        assert np.min(sums) < 5, "Did not find solutions near the 'zero' objective"
        assert np.max(sums) > 25, "Did not find solutions near the 'max' objective"

    def test_history_tracking(self):
        """The optimizer record history at every generation."""
        problem = OneZeroMax(n_var=10)
        optimizer = NSGA2(
            problem=problem,
            pop_size=30,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            num_choices=problem.num_choices,
            seed=42,
        )
        optimizer.run(generations=20, verbose=False)

        assert len(optimizer.history) == 20
        assert optimizer.history[0]["generation"] == 1
        assert optimizer.history[-1]["generation"] == 20

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        problem = OneZeroMax(n_var=10)

        def run(seed):
            opt = NSGA2(
                problem, 
                pop_size=30, 
                n_var=problem.n_var, 
                n_obj=problem.n_obj,
                num_choices=problem.num_choices,
                seed=seed
            )
            _, pf = opt.run(generations=10, verbose=False)
            return pf

        pf1 = run(123)
        pf2 = run(123)
        np.testing.assert_array_equal(pf1, pf2)