"""
Spread (Delta) metric for diversity assessment.

Measures the relative extent of spread achieved among the obtained
solutions along the Pareto front. Defined in Section IV-B of the paper.

Delta = (df + dl + sum|d_i - d_bar|) / (df + dl + (N-1)*d_bar)

where:
  - df = distance to extreme Pareto solutions (farest)
  - dl = distance to the other extreme
  - d_i = Euclidean distance between consecutive solutions
  - d_bar = mean of all d_i
  - N = number of solutions

Spread = 1 - Delta.  Higher values are better (1 = ideal spread).
"""

from __future__ import annotations
import numpy as np


def spread(obtained_front: np.ndarray) -> float:
    """Compute the Spread (diversity) metric.

    Args:
        obtained_front: (N, M) array of objective values from the
                        nondominated solutions.

    Returns:
        Spread value in [0, 1]. Higher is better (1 = perfect uniform spread).
    """
    n = len(obtained_front)
    if n < 2:
        return 0.0

    # Sort by first objective
    sorted_front = obtained_front[np.argsort(obtained_front[:, 0])]

    # Euclidean distances between consecutive solutions
    diffs = np.diff(sorted_front, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    d_bar = np.mean(distances)

    # d_f = distance from first solution to ideal extreme (use first solution as reference)
    # d_l = distance from last solution to ideal extreme (use last solution as reference)
    # When true Pareto extremes are not known, we approximate df = dl = d_bar
    df = d_bar
    dl = d_bar

    sum_dev = np.sum(np.abs(distances - d_bar))

    delta = (df + dl + sum_dev) / (df + dl + (n - 1) * d_bar)

    return float(1.0 - delta)
