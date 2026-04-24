from __future__ import annotations
import numpy as np

"""
Spread (Delta) metric for diversity assessment.
Measures both the uniformity of the obtained solutions and their convergence
to the boundaries of the true Pareto front.

Delta = (df + dl + sum|d_i - d_bar|) / (df + dl + (N-1)*d_bar)

Lower values are better (0 = perfect spread and boundary convergence).
"""

def spread(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """Compute the Spread (diversity) metric for 2 objectives.

    Args:
        obtained_front: (N, 2) array of objective values from the
                        nondominated solutions.
        true_front:     (P, 2) array of objective values from the
                        true Pareto front.

    Returns:
        Spread value. Lower is better (0 = perfect).
    """
    n, m = obtained_front.shape
    
    if n < 2:
        return float('inf') # Cannot calculate spread with less than 2 points

    # Calculate df and dl using the True Pareto front extremes
    # True extreme points (minimization)
    true_extreme_f1 = true_front[np.argmin(true_front[:, 0])] # Minimum of objective 1
    true_extreme_f2 = true_front[np.argmin(true_front[:, 1])] # Minimum of objective 2

    # Obtained extreme points (boundary solutions of the obtained front)
    obtained_extreme_f1 = obtained_front[np.argmin(obtained_front[:, 0])]
    obtained_extreme_f2 = obtained_front[np.argmin(obtained_front[:, 1])]

    # Distance between true extremes and obtained extremes
    df = np.linalg.norm(obtained_extreme_f1 - true_extreme_f1)
    dl = np.linalg.norm(obtained_extreme_f2 - true_extreme_f2)

    # Sort obtained front by first objective
    sorted_front = obtained_front[np.argsort(obtained_front[:, 0])]

    # Euclidean distances between consecutive solutions
    diffs = np.diff(sorted_front, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    if len(distances) == 0:
        return float('inf')

    d_bar = np.mean(distances)

    # Sum of absolute deviations from the mean distance
    sum_dev = np.sum(np.abs(distances - d_bar))

    # Compute Delta 
    denominator = df + dl + (n - 1) * d_bar
    
    if denominator == 0:
        return float('inf')

    delta = (df + dl + sum_dev) / denominator

    
    return float(delta)
