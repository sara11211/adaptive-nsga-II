from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from nsga2.visualization.style import LABEL_SIZE, TITLE_SIZE, DPI

"""2D hypervolume indicator plot (dominated region with reference point)."""

def plot_hypervolume_indicator(
    front: np.ndarray,
    ref_point: np.ndarray,
    true_front: Optional[np.ndarray] = None,
    title: str = "Hypervolume Indicator",
    xlabel: str = "$f_1$",
    ylabel: str = "$f_2$",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
    color: str = "#3B82F6",
    shade_color: str = "#3B82F6",
    shade_alpha: float = 0.25,
) -> plt.Figure:
    
    """Plot the 2D dominated region (hypervolume) bounded by the front and ref point."""
    front = np.asarray(front, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)
    sorted_front = front[np.argsort(front[:, 0])]

    fig, ax = plt.subplots(figsize=figsize)

    # Staircase polygon
    poly_x, poly_y = [], []
    for i in range(len(sorted_front)):
        x_i, y_i = sorted_front[i]
        if i == 0:
            poly_x += [x_i, x_i]
            poly_y += [y_i, y_i]
        else:
            prev_y = min(sorted_front[:i, 1])
            poly_x += [x_i, x_i]
            poly_y += [prev_y, y_i]
    poly_x += [sorted_front[-1, 0], sorted_front[0, 0]]
    poly_y += [ref_point[1], ref_point[1]]

    ax.fill(poly_x, poly_y, color=shade_color, alpha=shade_alpha, label="Hypervolume")

    # Boundary
    bound_x, bound_y = list(poly_x), list(poly_y)
    bound_x.append(sorted_front[0, 0])
    bound_y.append(sorted_front[0, 1])
    ax.plot(bound_x, bound_y, color=shade_color, linewidth=1.2, linestyle="--", alpha=0.6)

    # True Pareto front
    if true_front is not None:
        idx = np.argsort(true_front[:, 0])
        ax.plot(true_front[idx, 0], true_front[idx, 1], "k--", linewidth=1.5,
                label="True Pareto Front", zorder=2)

    # Solutions
    ax.scatter(sorted_front[:, 0], sorted_front[:, 1], c=color, s=40, alpha=0.9,
               edgecolors="white", linewidth=0.8, label="NSGA-II solutions", zorder=3)

    # Reference point 
    ax.plot(ref_point[0], ref_point[1], marker="X", color="#EF4444",
            markersize=10, markeredgewidth=1, linestyle="None", zorder=4)
    ax.plot([ref_point[0], ref_point[0]], [0, ref_point[1]],
            color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.plot([0, ref_point[0]], [ref_point[1], ref_point[1]],
            color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.5)
    ref_handle = mlines.Line2D([], [], color="#EF4444", marker="X", linestyle="None",
                               markersize=10, markeredgewidth=2,
                               label=f"Ref ({ref_point[0]:.2f}, {ref_point[1]:.2f})")
    ax.add_artist(ref_handle)

    # HV value box
    from nsga2.metrics.hv import hypervolume as _hv
    hv_value = _hv(front, ref_point)
    ax.text(0.02, 0.95, f"HV = {hv_value:.4f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Hypervolume indicator saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig