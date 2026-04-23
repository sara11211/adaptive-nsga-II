"""
Polynomial mutation for real-coded NSGA-II.

Reference: Deb & Goyal (1996), "A Combined Genetic Adaptive Search
(GenAS) for Engineering Design", Computer Science and Informatics, Vol. 26.

A polynomial probability distribution is used to perturb decision variables.
The shape of the distribution is controlled by the distribution index (eta_m).

Higher eta_m -> smaller mutations (more exploitation).
Lower eta_m  -> larger mutations (more exploration).
"""

from __future__ import annotations
import numpy as np


def polynomial_mutation(
    individual: np.ndarray,
    prob_mutation: float,
    eta_m: float = 20.0,
    bounds: np.ndarray | None = None,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply polynomial mutation to a decision-variable vector **in-place**.

    Each variable is mutated independently with probability *prob_mutation*.

    Args:
        individual: Decision variables (modified in-place, also returned).
        prob_mutation: Per-variable mutation probability.
        eta_m: Distribution index for mutation (default 20.0).
               Larger values produce smaller perturbations.
        bounds: Optional (N, 2) array of [lower, upper] bounds per variable.
        rng: Optional RandomState for reproducibility.

    Returns:
        The mutated decision-variable vector (same object as *individual*).
    """
    if rng is None:
        rng = np.random
    for i in range(len(individual)):
        if rng.rand() >= prob_mutation:
            continue

        lo = bounds[i, 0] if bounds is not None else 0.0
        hi = bounds[i, 1] if bounds is not None else 1.0

        delta = _delta_q(rng.rand(), eta_m)
        delta = min(delta, hi - individual[i])
        delta = max(delta, lo - individual[i])

        individual[i] += delta

    return individual


def _delta_q(u: float, eta: float) -> float:
    """Sample the perturbation delta_q given random value u in (0, 1).

    The polynomial distribution:
      - If u < 0.5: delta = (2u)^(1/(eta+1)) - 1
      - If u >= 0.5: delta = 1 - (2(1-u))^(1/(eta+1))
    """
    if u < 0.5:
        return (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
    else:
        return 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
