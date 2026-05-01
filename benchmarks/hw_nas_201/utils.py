from __future__ import annotations
import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from nsga2.core.nondominated_sort import fast_non_dominated_sort

from benchmarks.hw_nas_201.config import ALGORITHM, OUTPUT, PLOT
from nsga2.visualization.style import LABEL_SIZE, TITLE_SIZE, DPI

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

def count_on_true_front(obtained_front, true_pf, atol=1e-6):
    """Count how many unique true Pareto front points are matched
    by at least one obtained solution"""
    covered = np.zeros(len(true_pf), dtype=bool)
    for obj in obtained_front:
        for i, t in enumerate(true_pf):
            if not covered[i] and np.allclose(t, obj, atol=atol):
                covered[i] = True
    return int(np.sum(covered))

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
            writer.writerow([
                problem.arch_str(best_set[i]),
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
        best_plot[:, [1, 0]], true_front=true_plot[:, [1, 0]],
        title=f"HW-NAS-201 - {hw}",
        xlabel=PLOT["ylabel"], ylabel=PLOT["xlabel"],
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
    

# ── Comparison Utilities ───────────────────────────────────────────────────────
# Used by examples/run_comparison.py to produce side-by-side baseline vs adaptive
# plots, reports, and CSV files.

BASELINE_COLOR = "#3B82F6"       # blue
BASELINE_MARKER = "o"
ADAPTIVE_COLOR = "#EF4444"       # red
ADAPTIVE_MARKER = "s"

# For these metrics, lower values are better.
LOWER_IS_BETTER = {"GD", "IGD", "IGD+", "Spread"}


def _determine_winner(metric_name, baseline_mean, adaptive_mean):
    """Decide which variant wins for a given metric."""
    if metric_name in LOWER_IS_BETTER:
        if adaptive_mean < baseline_mean:
            return "Adaptive"
        elif baseline_mean < adaptive_mean:
            return "Baseline"
        return "Tie"
    # Higher is better (HV, n_on_true_pf)
    if adaptive_mean > baseline_mean:
        return "Adaptive"
    if baseline_mean > adaptive_mean:
        return "Baseline"
    return "Tie"


# ── Comparison Plots ──────────────────────────────────────────────────────────

def generate_comparison_convergence_plot(
    baseline_vals,
    adaptive_vals,
    metric_name,
    hw,
    ylabel,
    save_path,
):
    """Plot baseline and adaptive convergence curves on the same figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gens_b = np.arange(1, len(baseline_vals) + 1)
    gens_a = np.arange(1, len(adaptive_vals) + 1)

    ax.plot(gens_b, baseline_vals, color=BASELINE_COLOR, linewidth=2,
            marker=BASELINE_MARKER, markersize=3, label="Baseline NSGA-II",
            alpha=0.9)
    ax.plot(gens_a, adaptive_vals, color=ADAPTIVE_COLOR, linewidth=2,
            marker=ADAPTIVE_MARKER, markersize=3, label="Adaptive NSGA-II",
            alpha=0.9)

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(f"{hw} — {metric_name} Convergence Comparison",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Comparison {metric_name} plot saved to {save_path}")
    plt.close()
    return fig


def generate_comparison_pareto_plot(
    baseline_front,
    adaptive_front,
    true_pf,
    hw,
    save_path,
):
    """Plot both obtained fronts + true front.  x = latency, y = accuracy."""
    baseline_plot = to_plot_space(baseline_front)[:, [1, 0]]
    adaptive_plot = to_plot_space(adaptive_front)[:, [1, 0]]
    true_plot = to_plot_space(true_pf)[:, [1, 0]]

    fig, ax = plt.subplots(figsize=(10, 7))

    # True Pareto front
    idx = np.argsort(true_plot[:, 0])
    ax.plot(true_plot[idx, 0], true_plot[idx, 1], "k--", linewidth=1.5,
            label="True Pareto Front", zorder=1)

    # Baseline solutions
    ax.scatter(baseline_plot[:, 0], baseline_plot[:, 1],
               c=BASELINE_COLOR, s=40, alpha=0.7, edgecolors="white",
               linewidth=0.5, marker=BASELINE_MARKER,
               label="Baseline NSGA-II", zorder=2)

    # Adaptive solutions
    ax.scatter(adaptive_plot[:, 0], adaptive_plot[:, 1],
               c=ADAPTIVE_COLOR, s=40, alpha=0.7, edgecolors="white",
               linewidth=0.5, marker=ADAPTIVE_MARKER,
               label="Adaptive NSGA-II", zorder=3)

    ax.set_xlabel("Latency (ms)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
    ax.set_title(f"HW-NAS-201 — {hw} Pareto Front Comparison",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Comparison Pareto front saved to {save_path}")
    plt.close()
    return fig


def generate_adaptive_probs_plot(prob_history, hw, save_path):
    """Plot the evolution of adaptive strategy selection probabilities."""
    if not prob_history:
        return None

    generations = [h["generation"] for h in prob_history]
    probs = np.array([h["probs"] for h in prob_history])

    strategy_names = ["conservative", "uniform", "aggressive"]
    colors = ["#2563EB", "#10B981", "#EF4444"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(probs.shape[1]):
        ax.plot(generations, probs[:, i], color=colors[i], linewidth=2,
                marker=markers[i], markersize=6, label=strategy_names[i])

    ax.set_xlabel("Generation", fontsize=LABEL_SIZE)
    ax.set_ylabel("Selection Probability", fontsize=LABEL_SIZE)
    ax.set_title(f"{hw} — Adaptive Mutation Strategy Probabilities",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  Adaptive probs plot saved to {save_path}")
    plt.close()
    return fig


def generate_all_comparison_plots(baseline_data, adaptive_data, hw, true_pf, save_dir):
    """Generate every comparison plot for one hardware target."""
    pareto_dir = os.path.join(save_dir, "pareto_fronts")
    conv_dir = os.path.join(save_dir, "convergence")
    adaptive_dir = os.path.join(save_dir, "adaptive_probs")
    for d in (pareto_dir, conv_dir, adaptive_dir):
        os.makedirs(d, exist_ok=True)

    # Pareto front
    generate_comparison_pareto_plot(
        baseline_data["best_front"], adaptive_data["best_front"],
        true_pf, hw,
        os.path.join(pareto_dir, f"{hw}_comparison_pareto.png"),
    )

    # Convergence curves
    metric_info = [
        ("gd",       "GD",    "GD"),
        ("igd",      "IGD",   "IGD"),
        ("igd_plus", "IGD+",  "IGD+"),
        ("hv",       "HV",    "HV"),
        ("spread",   "Spread", r"Spread ($\Delta$)"),
    ]
    for key, name, ylabel in metric_info:
        generate_comparison_convergence_plot(
            baseline_data["history"][key],
            adaptive_data["history"][key],
            name, hw, ylabel,
            os.path.join(conv_dir, f"{hw}_comparison_{key}.png"),
        )

    # Adaptive probability evolution
    pool = adaptive_data.get("adaptive_pool")
    if pool and getattr(pool, "prob_history", None):
        generate_adaptive_probs_plot(
            pool.prob_history, hw,
            os.path.join(adaptive_dir, f"{hw}_adaptive_probs.png"),
        )


# ── Comparison Reports ────────────────────────────────────────────────────────

def save_comparison_report(baseline_agg, adaptive_agg, params,
                          adaptive_config, hw, save_dir):
    """Save a detailed text comparison report for one hardware target."""
    os.makedirs(save_dir, exist_ok=True)

    lines = [
        "=" * 72,
        "  HW-NAS-201 Comparison: Baseline NSGA-II vs Adaptive NSGA-II (SaMuNet)",
        f"  Hardware: {hw}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "=== Parameters ===",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=== Adaptive Mutation Config ===")
    if adaptive_config:
        for k, v in adaptive_config.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  (disabled)")

    # ── Table ───────────────────────────────────────────────────────────
    lines += [
        "",
        "=" * 72,
        f"  {hw} — Metric Comparison",
        "=" * 72,
        "",
        f"  {'Metric':<14} | {'Baseline (mean +/- std)':^26} | {'Adaptive (mean +/- std)':^26} | {'Delta mean':>10} | {'Delta %':>8} | {'Winner':<10}",
        f"  {'-'*14}-+-{'-'*26}-+-{'-'*26}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}",
    ]

    metric_order = [
        "IGD+", "HV", "Spread",
        "n_on_true_pf",
    ]
    wins = {"Adaptive": 0, "Baseline": 0, "Tie": 0}

    for metric in metric_order:
        b = baseline_agg.get(metric)
        a = adaptive_agg.get(metric)
        if b is None or a is None:
            continue

        if isinstance(b, dict) and isinstance(a, dict):
            b_m, b_s = b["mean"], b["std"]
            a_m, a_s = a["mean"], a["std"]
            delta = a_m - b_m
            pct = (delta / abs(b_m) * 100) if abs(b_m) > 1e-12 else 0.0
            winner = _determine_winner(metric, b_m, a_m)

            b_str = f"{b_m:.6f} +/- {b_s:.6f}"
            a_str = f"{a_m:.6f} +/- {a_s:.6f}"
            d_str = f"{delta:+.6f}"
            p_str = f"{pct:+.1f}%"

            wins[winner] += 1
        else:
            # Scalar (n_pareto, n_true_pareto, n_on_true_pf)
            b_v, a_v = float(b), float(a)
            delta = int(a_v - b_v)
            pct = (delta / abs(b_v) * 100) if abs(b_v) > 1e-12 else 0.0
            winner = _determine_winner(metric, b_v, a_v)   

            b_str = str(b)
            a_str = str(a)
            d_str = f"{delta:+d}"
            p_str = f"{pct:+.1f}%"

            wins[winner] += 1                                

        lines.append(
            f"  {metric:<14} | {b_str:^26} | {a_str:^26} "
            f"| {d_str:>10} | {p_str:>8} | {winner:<10}"
        )

    lines += [
        "",
        f"  Summary: Adaptive wins {wins['Adaptive']}, "
        f"Baseline wins {wins['Baseline']}, Tie {wins['Tie']}",
        "",
    ]

    path = os.path.join(save_dir, "comparison_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Comparison report saved to {path}")
    return path


def save_comparison_csv(baseline_agg, adaptive_agg, hw, save_dir):
    """Save per-hardware comparison metrics to CSV."""
    os.makedirs(save_dir, exist_ok=True)

    metric_order = [
        "IGD+", "HV", "Spread",
        "n_on_true_pf",
    ]
    path = os.path.join(save_dir, "comparison_metrics.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hardware", "metric",
            "baseline_mean", "baseline_std",
            "adaptive_mean", "adaptive_std",
            "delta_mean", "delta_pct",
            "lower_is_better", "winner",
        ])

        for metric in metric_order:
            b = baseline_agg.get(metric)
            a = adaptive_agg.get(metric)
            if b is None or a is None:
                continue

            if isinstance(b, dict) and isinstance(a, dict):
                b_m, b_s = b["mean"], b["std"]
                a_m, a_s = a["mean"], a["std"]
                delta = a_m - b_m
                pct = (delta / abs(b_m) * 100) if abs(b_m) > 1e-12 else 0.0
                lower = str(metric in LOWER_IS_BETTER or metric == "n_on_true_pf")
                winner = _determine_winner(metric, b_m, a_m)

                writer.writerow([
                    hw, metric,
                    f"{b_m:.6f}", f"{b_s:.6f}",
                    f"{a_m:.6f}", f"{a_s:.6f}",
                    f"{delta:+.6f}", f"{pct:+.2f}",
                    lower, winner,
                ])
            else:
                b_v, a_v = int(b), int(a)
                delta = a_v - b_v
                pct = (delta / abs(b_v) * 100) if abs(b_v) > 0 else 0.0
                writer.writerow([
                    hw, metric, b_v, "", a_v, "",
                    f"{delta:+d}", f"{pct:+.1f}", "", "Tie",
                ])

    return path