"""
HW-NAS-201: Hardware-aware Neural Architecture Search Benchmark.

This module provides a multi-objective formulation of the HW-NAS-Bench-201
benchmark. The two objectives are:

  f1: Top-1 accuracy on the target hardware (to maximise -> negated for minimisation)
  f2: Latency on the target hardware (to minimise)

A lookup-based approach is used where architectures are encoded as integer
indices (0 to 15624) mapping to pre-computed accuracy and latency values.
Since the actual dataset is large, synthetic surrogate data is generated
for demonstration purposes. Replace ``_load_data()`` with actual data loading
to use real HW-NAS-201 results.

References:
  - Dong et al., "HW-NAS-Bench: Hardware-Aware Neural Architecture Search
    Benchmark", NeurIPS 2021.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# Hardware targets available in HW-NAS-201
AVAILABLE_HARDWARES: List[str] = [
    "edgegpu",
    "edgetpu",
    "raspberry",
    "jetson_nano",
    "phone",
]

# Number of architectures in NAS-Bench-201
N_ARCHITECTURES = 15625


class HWNAS201:
    """HW-NAS-201 multi-objective problem.

    The decision variable is a single integer index in [0, 15624]
    representing a neural architecture. The two objectives are:
      - Negative top-1 accuracy (higher accuracy = lower objective).
      - Normalised inference latency (lower is better).

    Args:
        hardware: Target hardware name (one of AVAILABLE_HARDWARES).
    """

    def __init__(self, hardware: str = "edgegpu") -> None:
        if hardware not in AVAILABLE_HARDWARES:
            raise ValueError(
                f"Unknown hardware '{hardware}'. "
                f"Choose from: {AVAILABLE_HARDWARES}"
            )
        self.hardware = hardware
        self.n_var = 1
        self.n_obj = 2
        self.n_constr = 0
        self.bounds = np.array([[0.0, float(N_ARCHITECTURES - 1)]])

        # Load (or generate surrogate) accuracy and latency data
        self._accuracies, self._latencies = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load accuracy and latency data for the target hardware.

        Returns a synthetic surrogate for demonstration. Replace this
        method with actual data loading from the HW-NAS-201 dataset files.
        """
        rng = np.random.RandomState(hash(self.hardware) % (2**31))

        # Synthetic accuracies: distributed between 50% and 96%
        accuracies = 0.50 + 0.46 * rng.rand(N_ARCHITECTURES)

        # Synthetic latencies: correlated with lower accuracy (faster = less accurate)
        # Add noise to make the Pareto front interesting
        base_latency = 10.0 + (1.0 - (accuracies - 0.50) / 0.46) * 90.0
        latencies = base_latency * (0.7 + 0.6 * rng.rand(N_ARCHITECTURES))

        # Sort by accuracy for deterministic behaviour
        sort_idx = np.argsort(accuracies)
        return accuracies[sort_idx], latencies[sort_idx]

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single architecture.

        Args:
            x: Array of length 1 containing the architecture index.

        Returns:
            (objectives, constraints) — objectives = [-accuracy, latency].
        """
        idx = int(np.clip(x[0], 0, N_ARCHITECTURES - 1))
        acc = self._accuracies[idx]
        lat = self._latencies[idx]

        # Minimise: f1 = -accuracy, f2 = latency
        objectives = np.array([-acc, lat])
        return objectives, np.array([])

    def pareto_front(self, n_points: int = 500) -> np.ndarray:
        """Compute the approximate Pareto front from the dataset.

        Returns the best-known nondominated solutions across all architectures.
        """
        all_objs = np.column_stack([-self._accuracies, self._latencies])

        # Extract nondominated solutions
        n = len(all_objs)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            for j in range(n):
                if i == j or not is_nd[j]:
                    continue
                if np.all(all_objs[j] <= all_objs[i]) and np.any(all_objs[j] < all_objs[i]):
                    is_nd[i] = False
                    break

        return all_objs[is_nd]
