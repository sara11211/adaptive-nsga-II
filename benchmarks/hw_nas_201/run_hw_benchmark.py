"""
Run NSGA-II on the HW-NAS-201 benchmark across different hardware targets.

For each hardware, NSGA-II searches for architectures that maximise accuracy
while minimising latency. Results and Pareto front images are saved.

Usage:
    python -m benchmarks.hw_nas_201.run_hw_benchmark
"""

from __future__ import annotations
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nsga2.core.nsga2 import NSGA2
from nsga2.benchmarks.hw_nas_201.problem import HWNAS201, AVAILABLE_HARDWARES
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.hv import hypervolume
from nsga2.metrics.spread import spread
from nsga2.visualization.pareto_front import plot_pareto_front


def run_hw_benchmark(
    hardwares: list | None = None,
    n_runs: int = 5,
    generations: int = 100,
    pop_size: int = 50,
    seed_base: int = 42,
    output_dir: str = "output/hw_nas_201",
) -> dict:
    """Run NSGA-II on HW-NAS-201 across multiple hardware platforms.

    Args:
        hardwares: List of hardware names. Defaults to all available.
        n_runs: Independent runs per hardware.
        generations: Generations per run.
        pop_size: Population size.
        seed_base: Base random seed.
        output_dir: Directory for results and images.

    Returns:
        Dictionary mapping hardware name to metric statistics.
    """
    if hardwares is None:
        hardwares = AVAILABLE_HARDWARES

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for hw in hardwares:
        print(f"\n{'='*60}")
        print(f"  HW-NAS-201 on {hw}")
        print(f"{'='*60}")

        problem = HWNAS201(hardware=hw)
        true_pf = problem.pareto_front()

        igd_values = []
        hv_values = []
        spread_values = []
        best_front = None

        for run in range(n_runs):
            seed = seed_base + run
            optimizer = NSGA2(
                problem=problem,
                pop_size=pop_size,
                n_var=problem.n_var,
                n_obj=problem.n_obj,
                bounds=problem.bounds,
                seed=seed,
                eta_c=5.0,   # Wider crossover for discrete-like search
                eta_m=10.0,  # Moderate mutation
            )
            pareto_set, pareto_front = optimizer.run(generations=generations, verbose=False)

            if best_front is None or len(pareto_front) > len(best_front):
                best_front = pareto_front

            igd_values.append(inverted_generational_distance(pareto_front, true_pf))
            ref = np.max(true_pf, axis=0) + 0.1
            hv_values.append(hypervolume(pareto_front, ref))
            spread_values.append(spread(pareto_front))

        # Aggregate
        results[hw] = {
            "IGD": {"mean": float(np.mean(igd_values)), "std": float(np.std(igd_values))},
            "HV": {"mean": float(np.mean(hv_values)), "std": float(np.std(hv_values))},
            "Spread": {"mean": float(np.mean(spread_values)), "std": float(np.std(spread_values))},
        }

        print(f"  IGD:    {np.mean(igd_values):.6f} +/- {np.std(igd_values):.6f}")
        print(f"  HV:     {np.mean(hv_values):.6f} +/- {np.std(hv_values):.6f}")
        print(f"  Spread: {np.mean(spread_values):.6f} +/- {np.std(spread_values):.6f}")

        # Save Pareto front
        save_path = os.path.join(output_dir, f"{hw}_pareto.png")
        plot_pareto_front(
            best_front,
            true_front=true_pf,
            title=f"HW-NAS-201 — {hw}",
            xlabel="Negative Accuracy",
            ylabel="Latency (ms)",
            save_path=save_path,
            show=False,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-NAS-201 benchmark with NSGA-II")
    parser.add_argument("--hardware", type=str, nargs="*", default=None,
                        help="Hardware targets (default: all)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="output/hw_nas_201")
    args = parser.parse_args()

    run_hw_benchmark(
        hardwares=args.hardware,
        n_runs=args.runs,
        generations=args.gens,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
    )
