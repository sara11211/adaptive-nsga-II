from __future__ import annotations
import numpy as np

"""
Inverted Generational Distance Plus (IGD+) metric.

A weakly Pareto-compliant variant of IGD.
Unlike standard IGD, IGD+ does not penalize obtained solutions that dominate
points on the true Pareto front. The distance is only counted in dimensions
where the obtained solution is worse than the true Pareto point.

IGD+ = (1/|P*|) * sum_{j=1}^{|P*|} d_j^+

where d_j^+ is the modified distance from true Pareto point j to the nearest
obtained solution, calculated only over dominated objective values.

Lower values are better (0 = perfect).
"""

def inverted_generational_distance_plus(
    obtained_front: np.ndarray, true_front: np.ndarray
) -> float:
    """Compute the Inverted Generational Distance Plus (IGD+) metric.

    Args:
        obtained_front: (N, M) array — objectives of solutions found.
        true_front: (P, M) array — objectives of the true Pareto-optimal set.

    Returns:
        IGD+ value (non-negative, 0 = perfect).
    """
    if len(true_front) == 0:
        return float("inf")
    
    if len(obtained_front) == 0:
        return float("inf")

    # Pairwise differences from each true point to all obtained points: (P, N, M)
    diffs = true_front[:, np.newaxis, :] - obtained_front[np.newaxis, :, :]
    
    # IGD+ modification: Only consider positive differences (where obtained is worse than true)
    dominated_diffs = np.maximum(diffs, 0.0)
    
    # Calculate modified Euclidean distances: (P, N)
    distances = np.sqrt(np.sum(dominated_diffs ** 2, axis=2))

    # Minimum IGD+ distance from each true Pareto point to the obtained set
    min_distances = np.min(distances, axis=1)

    return float(np.mean(min_distances))
