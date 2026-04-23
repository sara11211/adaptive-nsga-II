"""
Pareto front visualization.

Plots obtained solutions alongside the true Pareto-optimal front (if available).
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup — use DejaVu Sans as default (reliable across environments)
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except RuntimeError:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_pareto_front(
    obtained_front: np.ndarray,
    true_front: Optional[np.ndarray] = None,
    title: str = "Pareto Front",
    xlabel: str = "$f_1$",
    ylabel: str = "$f_2$",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot the obtained Pareto front with optional true Pareto front overlay.

    Args:
        obtained_front: (N, M) array of obtained objective values.
        true_front: Optional (P, M) array of true Pareto-optimal objectives.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: If provided, save figure to this path (e.g. "output/pareto.png").
        show: Whether to display the figure.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # True Pareto front (line)
    if true_front is not None:
        # Sort by first objective for a clean line
        sorted_idx = np.argsort(true_front[:, 0])
        ax.plot(
            true_front[sorted_idx, 0],
            true_front[sorted_idx, 1],
            "k--",
            linewidth=1.5,
            label="True Pareto Front",
            zorder=1,
        )

    # Obtained solutions (scatter)
    ax.scatter(
        obtained_front[:, 0],
        obtained_front[:, 1],
        c="#2563EB",
        s=40,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        label="NSGA-II",
        zorder=2,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Pareto front saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig
