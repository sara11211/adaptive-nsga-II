"""ZDT test problems - Two objectives, both minimised, no constraints.

Defined in Table I of the NSGA-II paper. (for test purposes)
"""

from __future__ import annotations
import numpy as np

# ── Base Class ──────────────────────────────────────────────────────────────────
class _ZDTBase:
    """Common base: 2 objectives, no constraints."""

    n_var: int = 30
    n_obj: int = 2
    n_constr: int = 0
    bounds: np.ndarray = np.zeros((30, 2))

    def evaluate(self, x: np.ndarray):
        """Evaluate objectives and constraints for a solution vector.

        Args:
            x: Decision variable vector.
        Returns: 
            Tuple of (objectives array, constraints array).
        """
        raise NotImplementedError

    def pareto_front(self, n_points: int = 1000) -> np.ndarray:
        """Generate the true Pareto-optimal front.
        
        Args:
            n_points: Number of points to sample.
        Returns: 
            Array of shape (n_points, n_obj) of objective values.
        """
        raise NotImplementedError

# ── ZDT Implementations ─────────────────────────────────────────────────────────
class ZDT1(_ZDTBase):
    """Convex Pareto front. PF: f2 = 1 - sqrt(f1)."""

    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points=1000):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT2(_ZDTBase):
    """Non-convex Pareto front. PF: f2 = 1 - f1^2."""

    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points=1000):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - f1 ** 2])


class ZDT3(_ZDTBase):
    """Disconnected Pareto front (multiple segments)."""

    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - f1 * np.sin(10 * np.pi * f1) / g)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points=1000):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        return np.column_stack([f1, f2])[f2 <= 1.0 + 1e-6]


class ZDT4(_ZDTBase):
    """Many local Pareto fronts. Global PF: f2 = 1 - sqrt(f1)."""

    def __init__(self):
        self.n_var = 10
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.zeros((self.n_var, 2))
        self.bounds[0] = [0.0, 1.0]
        self.bounds[1:] = [-5.0, 5.0]

    def evaluate(self, x):
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(
            x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])
        )
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points=1000):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT6(_ZDTBase):
    """Non-uniform density. PF: f2 = 1 - f1^2, f1 in [0.2808, 1]."""

    def __init__(self):
        self.n_var = 10
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.tile([0.0, 1.0], (self.n_var, 1))

    def evaluate(self, x):
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), np.array([])

    def pareto_front(self, n_points=1000):
        f1 = np.linspace(0.280775, 1.0, n_points)
        return np.column_stack([f1, 1.0 - f1 ** 2])