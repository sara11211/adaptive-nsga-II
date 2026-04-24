"""Pareto front scatter plot with optional true front overlay."""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from nsga2.visualization.style import LABEL_SIZE, TITLE_SIZE, DPI


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
    
    """Scatter plot of obtained solutions, with optional true Pareto front."""
    fig, ax = plt.subplots(figsize=figsize)

    if true_front is not None:
        idx = np.argsort(true_front[:, 0])
        ax.plot(true_front[idx, 0], true_front[idx, 1], "k--", linewidth=1.5,
                label="True Pareto Front", zorder=1)

    ax.scatter(obtained_front[:, 0], obtained_front[:, 1], c="#2563EB", s=40,
               alpha=0.8, edgecolors="white", linewidth=0.5, label="NSGA-II", zorder=2)

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Pareto front saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig