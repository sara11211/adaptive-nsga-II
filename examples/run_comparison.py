"""Run comparative benchmark: Baseline NSGA-II vs Adaptive NSGA-II (SaMuNet).

Runs both variants on the same hardware targets with identical seeds,
then generates side-by-side comparison plots, reports, and CSV.

Usage:
    python examples/run_comparison.py
    python examples/run_comparison.py --dataset cifar100 --hardware edgegpu
    python examples/run_comparison.py --gens 50 --runs 3
"""

from __future__ import annotations
import sys
import os
import csv
import argparse
from datetime import datetime
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nsga2.core.nsga2 import NSGA2
from benchmarks.hw_nas_201.config import (
    ALGORITHM, DATA, OUTPUT, DATASETS, ADAPTIVE_MUTATION,
)
from benchmarks.hw_nas_201.problem import HWNAS201, discover_hardware
from benchmarks.hw_nas_201.utils import (
    extract_front, compute_metrics, aggregate_metrics, count_on_true_front,
    generate_all_comparison_plots,
    save_comparison_report,
    save_comparison_csv,
)

BASELINE_LABEL = "Baseline"
ADAPTIVE_LABEL = "Adaptive"


def _make_optimizer(problem, seed, pop_size, adaptive_config):
    """Create NSGA-II optimizer with given adaptive mutation config."""
    return NSGA2(
        problem=problem,
        pop_size=pop_size,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        num_choices=problem.num_choices,
        seed=seed,
        prob_crossover=ALGORITHM["prob_crossover"],
        adaptive_mutation=adaptive_config,
    )


def _run_variant(problem, true_pf, pop_size, generations, n_runs,
                 seed_base, adaptive_config, label=""):
    """Run one variant and return all data needed for comparison.

    Two-phase: Phase 1 tracking run (convergence history),
    Phase 2 statistical runs (mean +/- std metrics).
    """
    ref = np.max(true_pf, axis=0) + ALGORITHM["ref_offset"]

    # ── Phase 1: Tracking run ───────────────────────────────────────────
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

    print(f"    [{label}] Phase 1: Tracking run...")
    opt = _make_optimizer(problem, seed_base, pop_size, adaptive_config)
    opt.run(generations=generations, verbose=False, callback=_track)

    # ── Phase 2: Statistical runs ───────────────────────────────────────
    print(f"    [{label}] Phase 2: Statistical runs ({n_runs} runs)...")
    run_metrics = []
    best_front = None
    best_set = None

    for run in range(n_runs):
        opt = _make_optimizer(problem, seed_base + run, pop_size, adaptive_config)
        ps, pf = opt.run(generations=generations, verbose=False)
        if best_front is None or len(pf) > len(best_front):
            best_front = pf
            best_set = ps
        run_metrics.append(compute_metrics(pf, true_pf, ref))

    agg = aggregate_metrics(run_metrics)
    agg["n_pareto"] = len(best_front)
    agg["n_true_pareto"] = len(true_pf)
    agg["n_on_true_pf"] = count_on_true_front(best_front, true_pf)

    # Keep adaptive pool for probability evolution plot
    adaptive_pool = getattr(opt, "_adaptive_pool", None)

    return {
        "history": history,
        "run_metrics": run_metrics,
        "aggregate": agg,
        "best_front": best_front,
        "best_set": best_set,
        "adaptive_pool": adaptive_pool,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main comparison loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_comparison_for_dataset(
    dataset, hardwares,
    n_runs, generations, pop_size, seed_base, output_dir,
):
    """Run baseline vs adaptive comparison for one dataset."""
    params = dict(ALGORITHM)
    params["dataset"] = dataset
    params["lookup_dir"] = DATA["lookup_dir"]
    params["pop_size"] = pop_size
    params["generations"] = generations
    params["n_runs"] = n_runs
    params["seed_base"] = seed_base

    if hardwares is None:
        hardwares = discover_hardware(dataset, DATA["lookup_dir"])

    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    overall_csv_path = os.path.join(dataset_dir, "overall_comparison.csv")
    overall_rows = []

    for hw in hardwares:
        hw_dir = os.path.join(dataset_dir, hw)
        os.makedirs(hw_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Comparison on {hw}  ({dataset})")
        print(f"{'='*60}")

        problem = HWNAS201(hardware=hw, dataset=dataset)
        true_pf = problem.pareto_front()

        baseline = _run_variant(
            problem, true_pf, pop_size, generations, n_runs,
            seed_base, adaptive_config=None, label=BASELINE_LABEL,
        )
        adaptive = _run_variant(
            problem, true_pf, pop_size, generations, n_runs,
            seed_base, adaptive_config=ADAPTIVE_MUTATION, label=ADAPTIVE_LABEL,
        )

        generate_all_comparison_plots(baseline, adaptive, hw, true_pf, hw_dir)

        save_comparison_report(
            baseline["aggregate"], adaptive["aggregate"],
            params, ADAPTIVE_MUTATION, hw, hw_dir,
        )
        hw_csv = save_comparison_csv(
            baseline["aggregate"], adaptive["aggregate"], hw, hw_dir,
        )
        print(f"    Saved comparison to {hw_dir}/")

        with open(hw_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                overall_rows.append(row)

    if overall_rows:
        fieldnames = list(overall_rows[0].keys())
        with open(overall_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(overall_rows)
        print(f"\n  Dataset CSV: {overall_csv_path}")

    _print_overall_summary(overall_rows, title=f"SUMMARY FOR {dataset}")
    return overall_rows


def main():
    parser = argparse.ArgumentParser(
        description="HW-NAS-201: Baseline NSGA-II vs Adaptive NSGA-II (SaMuNet)"
    )
    parser.add_argument("--hardware", type=str, nargs="*", default=None)
    parser.add_argument("--dataset", type=str, nargs="*", default=None,
                        help="Datasets to run (default: all)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    datasets = args.dataset or DATASETS
    n_runs = args.runs or ALGORITHM["n_runs"]
    generations = args.gens or ALGORITHM["generations"]
    pop_size = args.pop_size or ALGORITHM["pop_size"]
    seed_base = ALGORITHM["seed_base"]
    output_dir = args.output_dir or os.path.join(OUTPUT["base_dir"], "comparative")

    all_rows = {}
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'#'*60}")

        rows = _run_comparison_for_dataset(
            dataset=dataset,
            hardwares=args.hardware,
            n_runs=n_runs,
            generations=generations,
            pop_size=pop_size,
            seed_base=seed_base,
            output_dir=output_dir,
        )
        all_rows[dataset] = rows

    if len(datasets) > 1:
        combined = []
        for rows in all_rows.values():
            combined.extend(rows)
        if combined:
            _print_overall_summary(combined, title="OVERALL CROSS-DATASET SUMMARY")

    return all_rows

def _print_overall_summary(rows, title="OVERALL COMPARISON SUMMARY"):
    """Print a concise winner tally across all hardware targets."""
    if not rows:
        return

    hw_groups = {}
    for row in rows:
        hw = row["hardware"]
        hw_groups.setdefault(hw, []).append(row)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Hardware':<12} {'Adaptive':>10} {'Baseline':>10} {'Tie':>6}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*6}")

    tally = {"Adaptive": 0, "Baseline": 0, "Tie": 0}

    for hw, hw_rows in hw_groups.items():
        hw_wins = {"Adaptive": 0, "Baseline": 0, "Tie": 0}
        for row in hw_rows:
            w = row.get("winner", "Tie")
            hw_wins[w] = hw_wins.get(w, 0) + 1
        tally["Adaptive"] += hw_wins["Adaptive"]
        tally["Baseline"] += hw_wins["Baseline"]
        tally["Tie"] += hw_wins["Tie"]
        print(f"  {hw:<12} {hw_wins['Adaptive']:>10} {hw_wins['Baseline']:>10} {hw_wins['Tie']:>6}")

    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*6}")
    print(f"  {'TOTAL':<12} {tally['Adaptive']:>10} {tally['Baseline']:>10} {tally['Tie']:>6}")


if __name__ == "__main__":
    main()