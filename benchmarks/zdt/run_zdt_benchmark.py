"""Run NSGA-II on ZDT benchmark problems.

Two-phase per problem:
  Phase 1 — tracking run for convergence/diversity history.
  Phase 2 — N_RUNS independent runs for statistics.

Usage:
    python -m benchmarks.zdt.run_zdt
    python -m benchmarks.zdt.run_zdt --problems ZDT1 ZDT3
"""

from __future__ import annotations
import sys
import os
import argparse
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from nsga2.core.nsga2 import NSGA2
from nsga2.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from benchmarks.zdt.config import ALGORITHM, OUTPUT, PROBLEMS
from benchmarks.zdt.utils import (
    extract_front, compute_metrics, aggregate_metrics, print_metrics,
    create_problem_dirs, save_results_txt, save_history_csv, generate_plots,
)

_PROBLEM_CLASSES = {
    "ZDT1": ZDT1, "ZDT2": ZDT2,
    "ZDT3": ZDT3, "ZDT4": ZDT4, "ZDT6": ZDT6,
}


def _make_optimizer(problem, seed, pop_size):
    """Instantiate NSGA-II with standard benchmark parameters."""
    return NSGA2(
        problem=problem,
        pop_size=pop_size,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        bounds=problem.bounds,
        seed=seed,
        prob_crossover=ALGORITHM["prob_crossover"],
        eta_c=ALGORITHM["eta_c"],
        eta_m=ALGORITHM["eta_m"],
    )


def run_zdt_benchmark(
    problems=None,
    n_runs=None, generations=None,
    pop_size=None, seed_base=None,
    output_dir=None,
):
    """Execute the full ZDT benchmark suite."""

    # Set defaults
    n_runs = n_runs or ALGORITHM["n_runs"]
    generations = generations or ALGORITHM["generations"]
    pop_size = pop_size or ALGORITHM["pop_size"]
    seed_base = seed_base or ALGORITHM["seed_base"]
    output_dir = output_dir or OUTPUT["base_dir"]

    if problems is None:
        problems = PROBLEMS

    params = dict(ALGORITHM)
    params["pop_size"] = pop_size
    params["generations"] = generations
    params["n_runs"] = n_runs
    params["seed_base"] = seed_base

    zdt_dir = os.path.join(output_dir, OUTPUT["subfolder"])
    os.makedirs(zdt_dir, exist_ok=True)
    all_results = {}

    for name in problems:
        if name not in _PROBLEM_CLASSES:
            raise ValueError(f"Unknown problem '{name}'. Choose from: {list(_PROBLEM_CLASSES.keys())}")

        prob_dir = os.path.join(zdt_dir, name.lower())
        dirs = create_problem_dirs(prob_dir)

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        problem = _PROBLEM_CLASSES[name]()
        true_pf = problem.pareto_front()

        # ── Phase 1: Tracking run ──────────────────────────────────────────────
        print("  [1/2] Tracking run...")
        history = {"gd": [], "igd": [], "igd_plus": [], "hv": [], "spread": []}

        def _track(gen, population):
            front = extract_front(population)
            if len(front) == 0 or len(true_pf) == 0:
                return
            m = compute_metrics(front, true_pf)
            history["gd"].append(m["GD"])
            history["igd"].append(m["IGD"])
            history["igd_plus"].append(m["IGD+"])
            history["hv"].append(m["HV"])
            history["spread"].append(m["Spread"])

        _make_optimizer(problem, seed_base, pop_size).run(
            generations=generations, verbose=False, callback=_track,
        )

        # ── Phase 2: Statistical runs ─────────────────────────────────────────
        print(f"  [2/2] Statistical runs ({n_runs} runs)...")
        run_metrics = []
        best_front = None
        best_set = None

        for run in range(n_runs):
            opt = _make_optimizer(problem, seed_base + run, pop_size)
            ps, pf = opt.run(generations=generations, verbose=False)

            if best_front is None or len(pf) > len(best_front):
                best_front = pf
                best_set = ps

            run_metrics.append(compute_metrics(pf, true_pf))

        all_results[name] = aggregate_metrics(run_metrics)
        all_results[name]["n_pareto"] = len(best_front)

        print_metrics({name: all_results[name]})

        # Save outputs
        save_results_txt({name: all_results[name]}, params, prob_dir)
        save_history_csv(history, prob_dir)
        last_metrics = {k: v[-1] for k, v in history.items() if v}
        generate_plots(name, best_front, true_pf, history, dirs, metrics=last_metrics)

        print(f"    Saved to {prob_dir}/")

    save_results_txt(all_results, params, zdt_dir)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZDT benchmark with NSGA-II")
    parser.add_argument("--problems", type=str, nargs="*", default=None,
                        help="Problems to run (default: all)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_zdt_benchmark(
        problems=args.problems,
        n_runs=args.runs,
        generations=args.gens,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
    )