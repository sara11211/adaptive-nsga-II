from __future__ import annotations
import numpy as np

"""
Hypervolume (HV) indicator.

The hypervolume is the Lebesgue measure (area for 2D, volume for 3D)
of the objective space dominated by the obtained solution set and
bounded by a reference point.

Larger values are better.
"""

def hypervolume(
    front: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Compute the hypervolume indicator.

    For 2-objective problems an exact O(N log N) sweep-line algorithm is used.

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

    # Filter out points that are worse than the reference point in any dimension.
    mask = np.all(front <= ref_point, axis=1)
    front = front[mask]

    # If all points were filtered out, hypervolume is 0
    if len(front) == 0:
        return 0.0
    
    return _hv_2d(front, ref_point)


def _hv_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume via sweep-line algorithm.

    1. Sort solutions by first objective ascending.
    2. For each point, accumulate a horizontal strip that stretches from
       the point's x-coordinate to the reference x-coordinate.
    3. The height of the strip is the difference between the current y
       and the previous best (lowest) y
    """
    # Sort by first objective ascending
    sorted_idx = np.argsort(front[:, 0])
    sorted_front = front[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for i in range(len(sorted_front)):
        x_i = sorted_front[i, 0]
        y_i = sorted_front[i, 1]

        # Width of the strip (from x_i to the reference point)
        width = max(ref_point[0] - x_i, 0.0)

        # Height of the strip (from y_i to the previous lowest y)
        height = max(prev_y - y_i, 0.0)

        hv += width * height
        prev_y = min(prev_y, y_i)

    return hv
