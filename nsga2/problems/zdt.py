"""
ZDT benchmark problems (Zitzler, Deb, Thiele 2000).

All problems have two objectives to be minimised and no constraints.
Defined in Table I of the NSGA-II paper.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


class _ZDTBase:
    """Common base for ZDT problems."""

    n_var: int = 30
    n_obj: int = 2
    n_constr: int = 0
    bounds: np.ndarray = np.zeros((30, 2))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        """Return n_points uniformly spaced points on the true Pareto front."""
        raise NotImplementedError


class ZDT1(_ZDTBase):
    """ZDT1: Convex Pareto front.

    f1 = x1
    f2 = g * (1 - sqrt(f1/g))
    g  = 1 + 9*sum(x[1:]) / (n-1)

    Pareto front: f2 = 1 - sqrt(f1), for f1 in [0, 1].
    """

    def __init__(self) -> None:
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT2(_ZDTBase):
    """ZDT2: Non-convex Pareto front.

    f2 = g * (1 - (f1/g)^2)

    Pareto front: f2 = 1 - f1^2, for f1 in [0, 1].
    """

    def __init__(self) -> None:
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])


class ZDT3(_ZDTBase):
    """ZDT3: Disconnected Pareto front.

    f2 = g * (1 - sqrt(f1/g) - f1*sin(10*pi*f1)/g)

    The Pareto front has several disconnected segments.
    """

    def __init__(self) -> None:
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - f1 * np.sin(10 * np.pi * f1) / g)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        """Generate true Pareto front points (disconnected)."""
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        # Keep only nondominated points
        points = np.column_stack([f1, f2])
        # Filter: f1 in the disconnected regions
        # Approximate by checking f2 < 1 (remove dominated segments)
        mask = f2 <= 1.0 + 1e-6
        return points[mask]


class ZDT4(_ZDTBase):
    """ZDT4: Many local Pareto fronts (21^10 local fronts).

    f1 = x1
    g  = 1 + 10*(n-1) + sum(x[i]^2 - 10*cos(4*pi*x[i]))
    f2 = g * (1 - sqrt(f1/g))

    Global Pareto front: f2 = 1 - sqrt(f1), x[1:] = 0.
    """

    def __init__(self) -> None:
        self.n_var = 10
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.zeros((self.n_var, 2))
        self.bounds[0] = [0.0, 1.0]
        self.bounds[1:] = [-5.0, 5.0]

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(
            x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])
        )
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT6(_ZDTBase):
    """ZDT6: Non-uniformly distributed Pareto front.

    The density of solutions is higher near f1 = 1 and lower near f1 = 0.

    f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
    g  = 1 + 9*(sum(x[1:])/(n-1))^0.25
    f2 = g * (1 - (f1/g)^2)
    """

    def __init__(self) -> None:
        self.n_var = 10
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        f1 = np.linspace(0.280775, 1.0, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])
