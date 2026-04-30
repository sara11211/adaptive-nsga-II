"""Run NSGA-II on the HW-NAS-201 benchmark.

Two-phase per hardware:
  Phase 1 — tracking run for convergence/diversity history.
  Phase 2 — N_RUNS independent runs for statistics.

Usage:
    python -m benchmarks.hw_nas_201.run_hw_benchmark
    python -m benchmarks.hw_nas_201.run_hw_benchmark --hardware edgegpu edgetpu
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
from benchmarks.hw_nas_201.config import ALGORITHM, DATA, OUTPUT, DEFAULT_DATASET, ADAPTIVE_MUTATION
from benchmarks.hw_nas_201.problem import HWNAS201, discover_hardware
from benchmarks.hw_nas_201.utils import (
    extract_front,  compute_metrics, count_on_true_front, aggregate_metrics, print_metrics,
    create_hw_dirs, save_results_txt, save_architectures_csv,
    save_history_csv, generate_plots,
)


def _make_optimizer(problem, seed, pop_size):
    """Instantiate NSGA-II with standard benchmark parameters."""
    return NSGA2(
        problem=problem,
        pop_size=pop_size,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        num_choices=problem.num_choices,
        seed=seed,
        prob_crossover=ALGORITHM["prob_crossover"],
        adaptive_mutation=ADAPTIVE_MUTATION,
    )

def run_hw_benchmark(
    hardwares=None, dataset=None,  
    n_runs=None, generations=None,
    pop_size=None, seed_base=None,
    output_dir=None,
):
    """Execute the full HW-NAS-201 benchmark suite.""" 

    # Set defaults
    dataset = dataset or DEFAULT_DATASET
    n_runs = n_runs or ALGORITHM["n_runs"]
    generations = generations or ALGORITHM["generations"]
    pop_size = pop_size or ALGORITHM["pop_size"]
    seed_base = seed_base or ALGORITHM["seed_base"]
    output_dir = output_dir or OUTPUT["base_dir"]

    if hardwares is None:
        hardwares = discover_hardware(dataset, DATA["lookup_dir"])

    params = dict(ALGORITHM)
    params["dataset"] = dataset
    params["lookup_dir"] = DATA["lookup_dir"]
    params["pop_size"] = pop_size
    params["generations"] = generations
    params["n_runs"] = n_runs
    params["seed_base"] = seed_base

    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    all_results = {}

    for hw in hardwares:
        hw_dir = os.path.join(dataset_dir, hw)
        dirs = create_hw_dirs(hw_dir)

        print(f"\n{'='*60}")
        print(f"  HW-NAS-201 on {hw}  ({dataset})")
        print(f"{'='*60}")

        problem = HWNAS201(hardware=hw, dataset=dataset)
        true_pf = problem.pareto_front()
        ref = np.max(true_pf, axis=0) + ALGORITHM["ref_offset"]

        # ── Phase 1: Tracking run ──────────────────────────────────────────────
        print("  [1/2] Tracking run...")
        history = {"gd": [], "igd": [], "igd_plus": [], "hv": [], "spread": []}

        def _track(gen, population):
            front = extract_front(population)
            if len(front) == 0 or len(true_pf) == 0:
                return
            m = compute_metrics(front, true_pf, ref)
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

            run_metrics.append(compute_metrics(pf, true_pf, ref))

        all_results[hw] = aggregate_metrics(run_metrics)
        all_results[hw]["n_pareto"] = len(best_front)
        all_results[hw]["n_true_pareto"] = len(true_pf)
        all_results[hw]["n_on_true_pf"] = count_on_true_front(best_front, true_pf)

        print_metrics({hw: all_results[hw]})

        # Save outputs
        hw_params = {**params, "hardware": hw}
        save_results_txt({hw: all_results[hw]}, hw_params, hw_dir)
        save_results_txt({hw: all_results[hw]}, hw_params, hw_dir)
        save_architectures_csv(problem, best_set, best_front, hw_dir)
        save_history_csv(history, hw_dir)
        last_metrics = {k: v[-1] for k, v in history.items() if v}
        generate_plots(hw, best_front, true_pf, history, dirs, metrics=last_metrics)

        print(f"    Saved to {hw_dir}/")

    save_results_txt(all_results, params, dataset_dir)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-NAS-201 benchmark with NSGA-II")
    parser.add_argument("--hardware", type=str, nargs="*", default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_hw_benchmark(
        hardwares=args.hardware,
        dataset=args.dataset,
        n_runs=args.runs,
        generations=args.gens,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
    )