"""
Individual representation for NSGA-II.

Each individual carries:
- decision variables (genome)
- objective function values
- constraint violation values
- rank from nondominated sorting
- crowding distance
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class Individual:
    """A single solution in the NSGA-II population.

    Attributes:
        decision_vars: numpy array of decision variable values.
        objectives: numpy array of objective function values (after evaluation).
        constraints: numpy array of constraint violations (g_i(x) >= 0 form).
                     Positive values mean the constraint is satisfied.
        rank: nondomination rank (0 = Pareto front 1).
        crowding_distance: crowding distance within its front.
    """

    __slots__ = ("decision_vars", "objectives", "constraints", "rank", "crowding_distance")

    def __init__(
        self,
        decision_vars: np.ndarray,
        n_objectives: int,
        n_constraints: int = 0,
    ) -> None:
        self.decision_vars = np.asarray(decision_vars, dtype=np.float64)
        self.objectives: Optional[np.ndarray] = None
        self.constraints: Optional[np.ndarray] = None
        self.rank: int = 0
        self.crowding_distance: float = 0.0

        # Pre-allocate arrays for objectives and constraints
        if n_objectives > 0:
            self.objectives = np.full(n_objectives, np.inf)
        if n_constraints > 0:
            self.constraints = np.full(n_constraints, np.inf)

    def evaluate(self, problem) -> None:
        """Evaluate objectives and constraints using the given problem.

        Args:
            problem: A callable or object with ``evaluate(x)`` method that
                     returns (objectives, constraints) tuple.
        """
        obj, con = problem.evaluate(self.decision_vars)
        self.objectives = np.asarray(obj, dtype=np.float64)
        if con is not None and len(con) > 0:
            self.constraints = np.asarray(con, dtype=np.float64)
        else:
            self.constraints = None

    @property
    def total_constraint_violation(self) -> float:
        """Sum of negative constraint violations (0 if feasible)."""
        if self.constraints is None:
            return 0.0
        # Constraints in the form g(x) <= 0 => violation = max(0, g(x))
        # Here we store violations directly: positive = satisfied, negative = violated
        return float(-np.sum(np.minimum(self.constraints, 0.0)))

    @property
    def is_feasible(self) -> bool:
        """True if all constraints are satisfied (or no constraints)."""
        if self.constraints is None:
            return True
        return bool(np.all(self.constraints >= -1e-12))

    def __repr__(self) -> str:
        obj_str = (
            f"{self.objectives}" if self.objectives is not None else "unevaluated"
        )
        return (
            f"Individual(rank={self.rank}, cd={self.crowding_distance:.4f}, "
            f"obj={obj_str})"
        )

    def copy(self) -> "Individual":
        """Return a deep copy of this individual."""
        new = Individual.__new__(Individual)
        new.decision_vars = self.decision_vars.copy()
        new.objectives = self.objectives.copy() if self.objectives is not None else None
        new.constraints = (
            self.constraints.copy() if self.constraints is not None else None
        )
        new.rank = self.rank
        new.crowding_distance = self.crowding_distance
        return new
