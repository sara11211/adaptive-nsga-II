"""
Example: Run NSGA-II on HW-NAS-201 across multiple hardware targets.

Usage:
    python examples/run_hw_nas201.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nsga2.core.nsga2 import NSGA2
# The benchmarks package is at the project root level
from benchmarks.hw_nas_201.problem import HWNAS201, AVAILABLE_HARDWARES
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.hv import hypervolume
from nsga2.visualization.pareto_front import plot_pareto_front


def main():
    os.makedirs("output/hw_nas_201", exist_ok=True)

    for hw in AVAILABLE_HARDWARES:
        print(f"\n--- HW-NAS-201 on {hw} ---")
        problem = HWNAS201(hardware=hw)
        true_pf = problem.pareto_front()

        optimizer = NSGA2(
            problem=problem,
            pop_size=50,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            bounds=problem.bounds,
            seed=42,
            eta_c=5.0,
            eta_m=10.0,
        )

        _, pareto_front = optimizer.run(generations=100)

        igd = inverted_generational_distance(pareto_front, true_pf)
        ref = np.max(true_pf, axis=0) + 0.1
        hv = hypervolume(pareto_front, ref)
        print(f"  IGD: {igd:.6f} | HV: {hv:.6f} | Front: {len(pareto_front)}")

        plot_pareto_front(
            pareto_front,
            true_front=true_pf,
            title=f"HW-NAS-201 — {hw}",
            xlabel="Negative Accuracy",
            ylabel="Latency (ms)",
            save_path=f"output/hw_nas_201/{hw}_pareto.png",
        )


if __name__ == "__main__":
    main()
