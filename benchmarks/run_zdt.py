"""
Run NSGA-II on all ZDT benchmark problems.

Usage:
    python -m benchmarks.run_zdt --output-dir output/
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsga2.core.nsga2 import NSGA2
from nsga2.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from nsga2.metrics.gd import generational_distance
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.hv import hypervolume
from nsga2.metrics.spread import spread
from nsga2.visualization.pareto_front import plot_pareto_front

PROBLEMS = {
    "ZDT1": ZDT1,
    "ZDT2": ZDT2,
    "ZDT3": ZDT3,
    "ZDT4": ZDT4,
    "ZDT6": ZDT6,
}


def run_all_zdt(
    n_runs: int = 10,
    generations: int = 250,
    pop_size: int = 100,
    seed_base: int = 42,
    output_dir: str = "output",
) -> dict:
    """Run NSGA-II on all ZDT problems and collect metrics.

    Args:
        n_runs: Number of independent runs per problem.
        generations: Generations per run.
        pop_size: Population size.
        seed_base: Base random seed (each run increments it).
        output_dir: Directory to save Pareto front images.

    Returns:
        Dictionary mapping problem name to metric statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, ProblemClass in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        problem = ProblemClass()
        gd_values = []
        igd_values = []
        hv_values = []
        spread_values = []

        for run in range(n_runs):
            seed = seed_base + run

            optimizer = NSGA2(
                problem=problem,
                pop_size=pop_size,
                n_var=problem.n_var,
                n_obj=problem.n_obj,
                bounds=problem.bounds,
                seed=seed,
            )
            pareto_set, pareto_front = optimizer.run(generations=generations, verbose=False)

            true_pf = problem.pareto_front()

            gd = generational_distance(pareto_front, true_pf)
            igd = inverted_generational_distance(pareto_front, true_pf)

            # Reference point for HV: slightly worse than worst true Pareto point
            ref = np.max(true_pf, axis=0) + 0.1
            hv = hypervolume(pareto_front, ref)
            sp = spread(pareto_front)

            gd_values.append(gd)
            igd_values.append(igd)
            hv_values.append(hv)
            spread_values.append(sp)

        # Aggregate
        gd_arr = np.array(gd_values)
        igd_arr = np.array(igd_values)
        hv_arr = np.array(hv_values)
        sp_arr = np.array(spread_values)

        results[name] = {
            "GD": {"mean": float(np.mean(gd_arr)), "std": float(np.std(gd_arr))},
            "IGD": {"mean": float(np.mean(igd_arr)), "std": float(np.std(igd_arr))},
            "HV": {"mean": float(np.mean(hv_arr)), "std": float(np.std(hv_arr))},
            "Spread": {"mean": float(np.mean(sp_arr)), "std": float(np.std(sp_arr))},
        }

        # Print summary
        print(f"  GD:    {np.mean(gd_arr):.6f} +/- {np.std(gd_arr):.6f}")
        print(f"  IGD:   {np.mean(igd_arr):.6f} +/- {np.std(igd_arr):.6f}")
        print(f"  HV:    {np.mean(hv_arr):.6f} +/- {np.std(hv_arr):.6f}")
        print(f"  Spread:{np.mean(sp_arr):.6f} +/- {np.std(sp_arr):.6f}")

        # Save Pareto front plot for the last run
        plot_path = os.path.join(output_dir, f"{name.lower()}_pareto.png")
        optimizer = NSGA2(
            problem=problem, pop_size=pop_size, n_var=problem.n_var,
            n_obj=problem.n_obj, bounds=problem.bounds, seed=seed_base,
        )
        _, pf = optimizer.run(generations=generations, verbose=False)
        plot_pareto_front(
            pf,
            true_front=problem.pareto_front(),
            title=f"{name} — NSGA-II",
            save_path=plot_path,
            show=False,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZDT benchmarks with NSGA-II")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--gens", type=int, default=250, help="Generations per run")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    results = run_all_zdt(
        n_runs=args.runs,
        generations=args.gens,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
    )
