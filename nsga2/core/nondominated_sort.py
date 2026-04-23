"""
Fast nondominated sorting for NSGA-II.

Implements the O(MN^2) algorithm from Section III-A of the paper:
1. For each solution, compute domination count (n_p) and dominated set (S_p).
2. All solutions with n_p = 0 belong to front 0.
3. For each front, iterate through members' dominated sets and decrement counts.
4. Solutions whose count reaches zero form the next front.
"""

from __future__ import annotations
from typing import List
import numpy as np
from nsga2.core.individual import Individual


def _constrained_dominates(a: Individual, b: Individual) -> bool:
    """Check if individual *a* constrained-dominates individual *b*.

    Following Definition 1 in Section VI of the paper:
      1. If both feasible: a dominates b in the usual Pareto sense.
      2. If a feasible, b infeasible: a dominates b.
      3. If both infeasible: a has smaller total constraint violation.
    """
    a_feas = a.is_feasible
    b_feas = b.is_feasible

    if a_feas and b_feas:
        # Case 1: both feasible — standard Pareto dominance
        return _dominates(a.objectives, b.objectives)
    if a_feas and not b_feas:
        # Case 2: feasible always beats infeasible
        return True
    if not a_feas and b_feas:
        # Case 2 (reverse)
        return False
    # Case 3: both infeasible — smaller constraint violation wins
    return a.total_constraint_violation < b.total_constraint_violation


def _dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Standard Pareto dominance: a dominates b iff a is no worse on all
    objectives and strictly better on at least one."""
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Partition *population* into nondomination fronts.

    Uses the fast algorithm from the paper with O(MN^2) complexity
    where M = number of objectives, N = population size.

    Args:
        population: List of evaluated Individual objects.

    Returns:
        A list of fronts, where each front is a list of Individuals.
        fronts[0] is the first (best) Pareto front.
    """
    n = len(population)
    if n == 0:
        return []

    # Determine whether any individual has constraints
    has_constraints = any(p.constraints is not None and len(p.constraints) > 0 for p in population)
    dom_func = _constrained_dominates if has_constraints else (
        lambda a, b: _dominates(a.objectives, b.objectives)
    )

    # Domination count for each individual
    domination_count = np.zeros(n, dtype=int)
    # Set of indices each individual dominates
    dominated_sets: List[List[int]] = [[] for _ in range(n)]

    # Count pairwise dominations — O(MN^2)
    for i in range(n):
        for j in range(i + 1, n):
            if dom_func(population[i], population[j]):
                domination_count[j] += 1
                dominated_sets[i].append(j)
            elif dom_func(population[j], population[i]):
                domination_count[i] += 1
                dominated_sets[j].append(i)

    fronts: List[List[Individual]] = []

    # First front: all individuals with domination_count == 0
    front_indices = [i for i in range(n) if domination_count[i] == 0]
    front = [population[i] for i in front_indices]
    for ind in front:
        ind.rank = 0
    fronts.append(front)

    # Build subsequent fronts
    while front_indices:
        next_front_indices: List[int] = []
        for i in front_indices:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)
        if not next_front_indices:
            break
        front_indices = next_front_indices
        front = [population[i] for i in front_indices]
        for ind in front:
            ind.rank = len(fronts)
        fronts.append(front)

    return fronts
