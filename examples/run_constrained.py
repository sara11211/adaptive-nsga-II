"""
Example: Run NSGA-II on a constrained problem (CONSTR).

Usage:
    python examples/run_constrained.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsga2.core.nsga2 import NSGA2
from nsga2.problems.constrained import CONSTR
from nsga2.visualization.pareto_front import plot_pareto_front


def main():
    problem = CONSTR()

    optimizer = NSGA2(
        problem=problem,
        pop_size=100,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        n_constr=problem.n_constr,
        bounds=problem.bounds,
        seed=42,
    )

    pareto_set, pareto_front = optimizer.run(generations=500)

    print(f"\nConstrained NSGA-II on CONSTR")
    print(f"  Front size: {len(pareto_front)}")
    print(f"  f1 range: [{pareto_front[:, 0].min():.4f}, {pareto_front[:, 0].max():.4f}]")
    print(f"  f2 range: [{pareto_front[:, 1].min():.4f}, {pareto_front[:, 1].max():.4f}]")

    plot_pareto_front(
        pareto_front,
        title="CONSTR — Constrained NSGA-II",
        save_path="output/constr_pareto.png",
    )


if __name__ == "__main__":
    main()
