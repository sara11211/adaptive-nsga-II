"""
Generational Distance (GD) metric.

Measures how far the obtained Pareto front is from the true Pareto front.
Lower values are better (0 = perfect convergence).

GD = sqrt( (1/n) * sum_{i=1}^{n} d_i^2 )

where d_i is the minimum Euclidean distance from solution i to the
nearest point on the true Pareto front.
"""

from __future__ import annotations
import numpy as np


def generational_distance(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """Compute the Generational Distance metric.

    Args:
        obtained_front: (N, M) array — objectives of solutions found by the algorithm.
        true_front: (P, M) array — objectives of the true Pareto-optimal set.

    Returns:
        GD value (non-negative, 0 = perfect).
    """
    if len(obtained_front) == 0:
        return float("inf")

    # Pairwise distances: (N, P)
    diffs = obtained_front[:, np.newaxis, :] - true_front[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Minimum distance from each obtained solution to true front
    min_distances = np.min(distances, axis=1)

    return float(np.sqrt(np.mean(min_distances ** 2)))
