from __future__ import annotations
import csv
import os
from datetime import datetime
import numpy as np

from nsga2.core.nondominated_sort import fast_non_dominated_sort

from benchmarks.zdt.config import ALGORITHM, OUTPUT, PLOT

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


def _normalize_fronts(obtained, true_pf):
    """Normalize both fronts jointly to [0, 1] per objective."""
    combined = np.vstack([obtained, true_pf])
    mins = np.min(combined, axis=0)
    ranges = np.max(combined, axis=0) - mins
    ranges[ranges < 1e-12] = 1.0
    return (obtained - mins) / ranges, (true_pf - mins) / ranges

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(front, true_pf):
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
    """Print metrics for all problems."""
    for name, metrics in results.items():
        print(f"\n  {name}:")
        for mname, stats in metrics.items():
            if isinstance(stats, dict):
                print(f"    {mname}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            else:
                print(f"    {mname}: {stats}")


def create_problem_dirs(prob_dir):
    """Create subdirectories for a problem's output."""
    dirs = {}
    for key, subdir in OUTPUT["subdirs"].items():
        d = os.path.join(prob_dir, subdir)
        os.makedirs(d, exist_ok=True)
        dirs[key] = d
    return dirs


def save_results_txt(results, params, save_dir):
    """Save results to a text file."""
    lines = [
        "ZDT Benchmark Results (NSGA-II)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Parameters ===",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("=== Results ===")
    for name, metrics in results.items():
        lines.append(f"\n  {name}")
        for mname, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(f"    {mname}: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            else:
                lines.append(f"    {mname}: {stats}")
    path = os.path.join(save_dir, "results.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
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
def generate_plots(name, best_front, true_pf, history, dirs, metrics=None):
    """Generate all plots for a ZDT problem."""
    # Both objectives minimised 
    offset = ALGORITHM["ref_offset"]
    all_pts = np.vstack([best_front, true_pf])
    ref = np.array([
        np.max(all_pts[:, 0]) + offset,
        np.max(all_pts[:, 1]) + offset,
    ])

    # Pareto fronts
    plot_pareto_front(
        best_front, true_front=true_pf,
        title=f"{name} — NSGA-II",
        xlabel=PLOT["xlabel"], ylabel=PLOT["ylabel"],
        save_path=os.path.join(dirs["pareto"], f"{name.lower()}_pareto.png"),
        show=False,
    )

    hv_val = metrics.get("hv") if metrics else None
    plot_hypervolume_indicator(
        best_front, ref_point=ref, true_front=true_pf,
        hv_value=hv_val,
        title=f"{name} — Hypervolume Indicator",
        xlabel=PLOT["xlabel"], ylabel=PLOT["ylabel"],
        save_path=os.path.join(dirs["pareto"], f"{name.lower()}_hv_indicator.png"),
        show=False,
    )

    # Convergence & Diversity
    plot_gd(history["gd"], title=f"{name} — GD",
            save_path=os.path.join(dirs["convergence"], f"{name.lower()}_gd.png"), show=False)
    plot_igd(history["igd"], title=f"{name} — IGD",
             save_path=os.path.join(dirs["convergence"], f"{name.lower()}_igd.png"), show=False)
    plot_igdp(history["igd_plus"], title=f"{name} — IGD+",
              save_path=os.path.join(dirs["convergence"], f"{name.lower()}_igdp.png"), show=False)
    plot_spread(history["spread"], title=f"{name} — Spread",
                save_path=os.path.join(dirs["diversity"], f"{name.lower()}_spread.png"), show=False)