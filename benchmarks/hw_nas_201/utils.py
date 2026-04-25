from __future__ import annotations
import csv
import os
from datetime import datetime
import numpy as np

from nsga2.core.nondominated_sort import fast_non_dominated_sort

from benchmarks.hw_nas_201.config import ALGORITHM, OUTPUT, PLOT

from nsga2.metrics.gd import generational_distance
from nsga2.metrics.igd import inverted_generational_distance
from nsga2.metrics.igdp import inverted_generational_distance_plus
from nsga2.metrics.hv import hypervolume
from nsga2.metrics.spread import spread

from nsga2.visualization.pareto_front import plot_pareto_front
from nsga2.visualization.gd import plot_gd
from nsga2.visualization.igd import plot_igd
from nsga2.visualization.igdp import plot_igdp
from nsga2.visualization.hypervolume import plot_hypervolume_indicator
from nsga2.visualization.spread import plot_spread

# ── Pareto Fronts ───────────────────────────────────────────────────────────
def extract_front(population):
    """Extract first Pareto front from population."""
    fronts = fast_non_dominated_sort(population)
    return np.array([ind.objectives for ind in fronts[0]])


def to_plot_space(front):
    """[-accuracy, latency] → [accuracy, latency]"""
    plot_front = front.copy()
    plot_front[:, 0] = -plot_front[:, 0]
    return plot_front


def _normalize_fronts(obtained, true_pf):
    """Normalize both fronts jointly to [0, 1] per objective."""
    combined = np.vstack([obtained, true_pf])
    mins = np.min(combined, axis=0)
    ranges = np.max(combined, axis=0) - mins
    ranges[ranges < 1e-12] = 1.0
    return (obtained - mins) / ranges, (true_pf - mins) / ranges

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(front, true_pf, ref):
    """Compute all metrics on normalized fronts."""
    norm_front, norm_true = _normalize_fronts(front, true_pf)
    norm_ref = np.ones(front.shape[1]) + ALGORITHM["ref_offset"]
    return {
        "GD": generational_distance(norm_front, norm_true),
        "IGD": inverted_generational_distance(norm_front, norm_true),
        "IGD+": inverted_generational_distance_plus(norm_front, norm_true),
        "HV": hypervolume(norm_front, norm_ref),
        "Spread": spread(norm_front, norm_true),
    }


def aggregate_metrics(run_metrics):
    """Aggregate metrics across runs into mean ± std."""
    agg = {}
    for key in run_metrics[0]:
        values = [r[key] for r in run_metrics]
        agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return agg


def print_metrics(results):
    """Print metrics for all hardware targets."""
    for hw, metrics in results.items():
        print(f"\n  {hw}:")
        for name, stats in metrics.items():
            if isinstance(stats, dict):
                print(f"    {name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            else:
                print(f"    {name}: {stats}")

# ── I/O ────────────────────────────────────────────────────────────────────────
def create_hw_dirs(hw_dir):
    """Create subdirectories for a hardware's output."""
    dirs = {}
    for key, subdir in OUTPUT["subdirs"].items():
        d = os.path.join(hw_dir, subdir)
        os.makedirs(d, exist_ok=True)
        dirs[key] = d
    return dirs


def save_results_txt(results, params, save_dir):
    """Save results to a text file."""
    lines = [
        "HW-NAS-201 Benchmark Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Parameters ===",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("=== Results ===")
    for hw, metrics in results.items():
        lines.append(f"\n  {hw}")
        for name, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(f"    {name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            else:
                lines.append(f"    {name}: {stats}")
    path = os.path.join(save_dir, "results.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def save_architectures_csv(problem, best_set, best_front, save_dir):
    """Save the best architectures and their objectives to CSV."""
    plot_front = to_plot_space(best_front)
    path = os.path.join(save_dir, "architectures.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "accuracy", "latency_ms"])
        for i in range(len(best_set)):
            idx = int(best_set[i, 0])
            writer.writerow([
                problem.arch_str(idx),
                f"{plot_front[i, 0]:.4f}",
                f"{plot_front[i, 1]:.4f}",
            ])
    return path


def save_history_csv(history, save_dir):
    """Save convergence history to CSV."""
    path = os.path.join(save_dir, "convergence_history.csv")
    cols = ["generation", "gd", "igd", "igd_plus", "hv", "spread"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for gen in range(len(history["gd"])):
            writer.writerow([
                gen + 1,
                f"{history['gd'][gen]:.6f}",
                f"{history['igd'][gen]:.6f}",
                f"{history['igd_plus'][gen]:.6f}",
                f"{history['hv'][gen]:.6f}",
                f"{history['spread'][gen]:.6f}",
            ])
    return path

# ── Plotting ───────────────────────────────────────────────────────────────────
def generate_plots(hw, best_front, true_pf, history, dirs, metrics=None):
    """Generate all plots for a hardware target."""
    best_plot = to_plot_space(best_front)
    true_plot = to_plot_space(true_pf)
    offset = ALGORITHM["ref_offset"]

    # Reference point in plot space
    ref = np.array([
        np.min(true_plot[:, 0]) - offset,
        np.max(true_plot[:, 1]) + offset,
    ])

     # Pareto fronts
    plot_pareto_front(
        best_plot, true_front=true_plot,
        title=f"HW-NAS-201 - {hw}",
        xlabel=PLOT["xlabel"], ylabel=PLOT["ylabel"],
        save_path=os.path.join(dirs["pareto"], f"{hw}_pareto.png"),
        show=False,
    )

    hv_val = metrics.get("hv") if metrics else None
    plot_hypervolume_indicator(
        best_plot, ref_point=ref, true_front=true_plot,
        hv_value=hv_val,
        title=f"HW-NAS-201 - {hw} (Hypervolume)",
        xlabel=PLOT["xlabel"], ylabel=PLOT["ylabel"],
        save_path=os.path.join(dirs["pareto"], f"{hw}_hv_indicator.png"),
        show=False,
    )

    # Convergence & Diversity
    plot_gd(history["gd"], title=f"{hw} GD",
            save_path=os.path.join(dirs["convergence"], f"{hw}_gd.png"), show=False)
    plot_igd(history["igd"], title=f"{hw} IGD",
             save_path=os.path.join(dirs["convergence"], f"{hw}_igd.png"), show=False)
    plot_igdp(history["igd_plus"], title=f"{hw} IGD+",
              save_path=os.path.join(dirs["convergence"], f"{hw}_igd_plus.png"), show=False)

    plot_spread(history["spread"], title=f"{hw} Spread",
                save_path=os.path.join(dirs["diversity"], f"{hw}_spread.png"), show=False)