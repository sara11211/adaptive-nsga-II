"""
Inverted Generational Distance (IGD) metric.

Measures both convergence and diversity. Unlike GD, IGD is computed
from the perspective of the true Pareto front: how far is each true
Pareto point from the nearest obtained solution?

IGD = (1/|P*|) * sum_{j=1}^{|P*|} d_j

where d_j is the minimum Euclidean distance from true Pareto point j
to the nearest obtained solution.

Lower values are better (0 = perfect).
"""

from __future__ import annotations
import numpy as np


def inverted_generational_distance(
    obtained_front: np.ndarray, true_front: np.ndarray
) -> float:
    """Compute the Inverted Generational Distance metric.

    Args:
        obtained_front: (N, M) array — objectives of solutions found.
        true_front: (P, M) array — objectives of the true Pareto-optimal set.

    Returns:
        IGD value (non-negative, 0 = perfect).
    """
    if len(true_front) == 0:
        return float("inf")

    # Pairwise distances from each true point to all obtained points: (P, N)
    diffs = true_front[:, np.newaxis, :] - obtained_front[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Minimum distance from each true Pareto point to obtained set
    min_distances = np.min(distances, axis=1)

    return float(np.mean(min_distances))
