"""
Convergence plot: shows metric values across generations.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except RuntimeError:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_convergence(
    metric_values: List[float],
    metric_name: str = "GD",
    title: str = "Convergence",
    xlabel: str = "Generation",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot a metric's value across generations.

    Args:
        metric_values: List of metric values, one per generation.
        metric_name: Name of the metric (used in ylabel if ylabel is None).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label (defaults to metric_name).
        save_path: If provided, save figure to this path.
        show: Whether to display the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    generations = np.arange(1, len(metric_values) + 1)
    ax.plot(generations, metric_values, color="#2563EB", linewidth=2, marker="o", markersize=3)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel or metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Convergence plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig
