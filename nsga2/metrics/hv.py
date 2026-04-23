"""
Hypervolume (HV) indicator.

The hypervolume is the Lebesgue measure (area for 2D, volume for 3D)
of the objective space dominated by the obtained solution set and
bounded by a reference point.

Larger values are better.
"""

from __future__ import annotations
import numpy as np


def hypervolume(
    front: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Compute the hypervolume indicator.

    For 2-objective problems an exact O(N log N) sweep-line algorithm is used.
    For higher dimensions a naive Monte-Carlo approximation is computed.

    Args:
        front: (N, M) array of objective values (all to be minimised).
        ref_point: (M,) array — the reference (anti-optimal) point.
                   Must dominate (be worse than) all solutions in the front,
                   i.e. ref_point[i] >= max(front[:, i]) for all i.

    Returns:
        Hypervolume value (non-negative).
    """
    if len(front) == 0:
        return 0.0

    front = np.asarray(front, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)

    if front.shape[1] == 2:
        return _hv_2d(front, ref_point)
    else:
        return _hv_monte_carlo(front, ref_point)


def _hv_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume via sweep-line algorithm.

    1. Sort solutions by first objective.
    2. Accumulate rectangular strips between consecutive solutions.
    """
    # Sort by first objective ascending
    sorted_idx = np.argsort(front[:, 0])
    sorted_front = front[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for i in range(len(sorted_front)):
        x_i = sorted_front[i, 0]
        y_i = sorted_front[i, 1]

        # Width of the strip (from previous x to this x, or from ref)
        if i == 0:
            width = ref_point[0] - x_i
        else:
            width = sorted_front[i - 1, 0] - x_i
            width = ref_point[0] - x_i  # Actually use distance from ref

        # Height contribution: from this y to the previous highest y
        height = prev_y - y_i
        # Clip: contribution is always positive
        height = max(height, 0.0)
        width = max(ref_point[0] - x_i, 0.0)

        hv += width * height
        prev_y = min(prev_y, y_i)

    return hv


def _hv_monte_carlo(front: np.ndarray, ref_point: np.ndarray, n_samples: int = 100000) -> float:
    """Approximate hypervolume for M > 2 objectives using Monte-Carlo.

    Samples random points in the hyper-rectangle defined by the worst
    solution and the reference point, then counts what fraction is
    dominated by the front.
    """
    n_obj = front.shape[1]

    # Bounding box: lower corner = ideal point of the front
    lower = np.min(front, axis=0)
    upper = ref_point

    # Box volume
    box_vol = float(np.prod(upper - lower))

    # Random sampling
    rng = np.random.default_rng(42)
    points = rng.uniform(lower, upper, size=(n_samples, n_obj))

    # Check which points are dominated by at least one solution
    # A point p is dominated if there exists a solution s with s <= p component-wise
    # (at least one strict)
    dominated = np.zeros(n_samples, dtype=bool)
    for s in front:
        dominated |= np.all(s <= points, axis=1)

    return box_vol * float(np.mean(dominated))
