"""
Constrained multi-objective test problems.

Implements CONSTR, SRN, and TNK from Table V of the NSGA-II paper.
Constraints are in the form g(x) <= 0 (returned as negative values when violated).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class CONSTR:
    """CONSTR problem: part of the unconstrained Pareto front is infeasible.

    Two objectives, two constraints, two variables.
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.n_constr = 2
        self.bounds = np.array([[0.1, 1.0], [0.0, 5.0]])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        f2 = (1.0 + x[1]) / x[0]

        g1 = -x[1] - 9.0 * x[0] + 6.0
        g2 = x[1] - 9.0 * x[0] + 1.0

        # Store as "satisfied when <= 0" for the constraint handler
        return np.array([f1, f2]), np.array([-g1, -g2])


class SRN:
    """SRN problem (Srinivas & Deb original NSGA test).

    Two objectives, two constraints, two variables.
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.n_constr = 2
        self.bounds = np.array([[-20.0, 20.0], [-20.0, 20.0]])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = 2.0 + (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
        f2 = 9.0 * x[0] - (x[1] - 1.0) ** 2

        g1 = -(x[0] ** 2 + x[1] ** 2) + 225.0
        g2 = x[0] - 3.0 * x[1] + 10.0

        return np.array([f1, f2]), np.array([-g1, -g2])


class TNK:
    """TNK problem (Tanaka et al.).

    Two objectives, two constraints, two variables.
    The Pareto front is discontinuous.
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.n_constr = 2
        self.bounds = np.tile([0.0, np.pi], (2, 1))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        f2 = x[1]

        g1 = -(x[0] ** 2 + x[1] ** 2) + 1.0 + 0.1 * np.cos(
            16.0 * np.arctan(x[0] / x[1])
        )
        g2 = -0.5 * (x[0] - 0.5) ** 2 - (x[1] - 0.5) ** 2 + 0.5

        return np.array([f1, f2]), np.array([-g1, -g2])
