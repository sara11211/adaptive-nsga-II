"""
Simulated Binary Crossover (SBX) for real-coded NSGA-II.

Reference: Deb & Agrawal (1995), "Simulated Binary Crossover for
Continuous Search Space", Complex Systems, Vol. 9, pp. 115-148.

SBX simulates the behaviour of single-point crossover in binary strings
but works directly with real-valued decision variables. A spread factor
(beta_q) is sampled from a distribution whose shape is controlled by
the distribution index (eta_c).

Higher eta_c -> children closer to parents (narrower spread).
Lower eta_c  -> children farther from parents (wider spread).
"""

from __future__ import annotations
import numpy as np


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    prob_crossover: float = 0.9,
    eta_c: float = 20.0,
    bounds: np.ndarray | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Simulated Binary Crossover on two parent vectors.

    With probability *prob_crossover* the two parents produce two children
    via the SBX operator. Otherwise they are returned unchanged.

    Args:
        parent1: Decision variables of parent 1.
        parent2: Decision variables of parent 2.
        prob_crossover: Crossover probability (default 0.9).
        eta_c: Distribution index for crossover (default 20.0).
               Larger values produce offspring more similar to parents.
        bounds: Optional (N, 2) array of [lower, upper] bounds per variable.
        rng: Optional RandomState for reproducibility.

    Returns:
        Tuple (child1, child2) with the same shape as the parents.
    """
    if rng is None:
        rng = np.random
    child1 = parent1.copy()
    child2 = parent2.copy()

    if rng.rand() > prob_crossover:
        return child1, child2

    for i in range(len(parent1)):
        if abs(parent1[i] - parent2[i]) < 1e-14:
            continue

        # Ensure p1 <= p2 for symmetric calculation
        if parent1[i] < parent2[i]:
            p1, p2 = parent1[i], parent2[i]
        else:
            p1, p2 = parent2[i], parent1[i]

        u = rng.rand()

        # Compute beta_q from the distribution
        beta_q = _beta_q(u, eta_c)

        c1 = 0.5 * ((1 + beta_q) * p1 + (1 - beta_q) * p2)
        c2 = 0.5 * ((1 - beta_q) * p1 + (1 + beta_q) * p2)

        # Respect variable bounds
        if bounds is not None:
            lo, hi = bounds[i]
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)

        child1[i] = c1
        child2[i] = c2

    return child1, child2


def _beta_q(u: float, eta: float) -> float:
    """Sample the spread factor beta_q given random value u in (0, 1).

    The distribution is:
      - If u <= 0.5:  beta = (2u)^(1/(eta+1))
      - If u >  0.5:  beta = (1/(2(1-u)))^(1/(eta+1))
    """
    if u <= 0.5:
        return (2.0 * u) ** (1.0 / (eta + 1.0))
    else:
        return (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
