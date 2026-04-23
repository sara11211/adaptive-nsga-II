"""
Example: Run NSGA-II on ZDT1 and plot the Pareto front.

Usage:
    python examples/run_zdt1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nsga2.core.nsga2 import NSGA2
from nsga2.problems.zdt import ZDT1
from nsga2.metrics.gd import generational_distance
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.hv import hypervolume
from nsga2.metrics.spread import spread
from nsga2.visualization.pareto_front import plot_pareto_front
from nsga2.visualization.convergence import plot_convergence


def main():
    problem = ZDT1()

    optimizer = NSGA2(
        problem=problem,
        pop_size=100,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        bounds=problem.bounds,
        seed=42,
    )

    pareto_set, pareto_front = optimizer.run(generations=250)

    # Compute metrics
    true_pf = problem.pareto_front()
    gd = generational_distance(pareto_front, true_pf)
    igd = inverted_generational_distance(pareto_front, true_pf)
    ref = np.max(true_pf, axis=0) + 0.1
    hv = hypervolume(pareto_front, ref)
    sp = spread(pareto_front)

    print(f"\nPerformance Metrics:")
    print(f"  GD:    {gd:.6f}")
    print(f"  IGD:   {igd:.6f}")
    print(f"  HV:    {hv:.6f}")
    print(f"  Spread:{sp:.6f}")
    print(f"  Front size: {len(pareto_front)}")

    # Plot Pareto front
    plot_pareto_front(
        pareto_front,
        true_front=true_pf,
        title="ZDT1 — NSGA-II",
        save_path="output/zdt1_pareto.png",
    )

    # Plot convergence (GD across generations)
    gd_history = []
    for entry in optimizer.history:
        gd_val = generational_distance(entry["objectives"], true_pf)
        gd_history.append(gd_val)

    plot_convergence(
        gd_history,
        metric_name="GD",
        title="ZDT1 — Convergence",
        save_path="output/zdt1_convergence.png",
    )


if __name__ == "__main__":
    main()
