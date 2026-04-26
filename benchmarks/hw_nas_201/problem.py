"""HW-NAS-201 multi-objective problem.

f1 = -accuracy (negated for minimisation)
f2 = latency
"""

from __future__ import annotations
import csv
import os
import numpy as np

from benchmarks.hw_nas_201.config import DATA, DATASETS

_data_cache = {}


def discover_hardware(dataset, lookup_dir=None):
    """Get available hardware targets from lookup CSV header.

    Args:
        dataset: Dataset name.
        lookup_dir: Directory containing CSV files.
    Returns: 
        List of hardware names.
    """
    if lookup_dir is None:
        lookup_dir = DATA["lookup_dir"]
    csv_path = os.path.join(lookup_dir, f"nas201_{dataset}.csv")
    with open(csv_path) as f:
        header = next(csv.reader(f))
    return [c.replace("_latency_ms", "") for c in header if c.endswith("_latency_ms")]


def _read_header(csv_path):
    with open(csv_path) as f:
        return next(csv.reader(f))

# ── Problem Class ──────────────────────────────────────────────────────────────
class HWNAS201:
    """HW-NAS-201 search problem mapping architecture index to accuracy and latency."""

    def __init__(self, hardware, dataset, lookup_dir=None):
        """""
        Args:
            hardware: Target hardware name.
            dataset: Dataset name.
            lookup_dir: Directory containing lookup tables.
        """
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")
        if lookup_dir is None:
            lookup_dir = DATA["lookup_dir"]

        available_hw = discover_hardware(dataset, lookup_dir)
        if hardware not in available_hw:
            raise ValueError(f"Unknown hardware '{hardware}'. Available: {available_hw}")

        self.hardware = hardware
        self.dataset = dataset
        self.lookup_dir = lookup_dir
        self._load_data()

        self.n_var = 1
        self.n_obj = 2

    def _load_data(self):
        """Load and cache architecture data from CSV."""

        cache_key = (self.dataset, self.hardware)
        if cache_key in _data_cache:
            self._accuracies, self._latencies, self._arch_strings = _data_cache[cache_key]
            return

        csv_path = os.path.join(self.lookup_dir, f"nas201_{self.dataset}.csv")
        header = _read_header(csv_path)
        hw_idx = header.index(f"{self.hardware}_latency_ms")

        accs, lats, strs = [], [], []
        with open(csv_path) as f:
            next(f)
            for row in csv.reader(f):
                accs.append(float(row[2]))
                lats.append(float(row[hw_idx]))
                strs.append(row[1])

        self._accuracies = np.array(accs)
        self._latencies = np.array(lats)
        self._arch_strings = strs
        self._n = len(accs)

        _data_cache[cache_key] = (self._accuracies, self._latencies, self._arch_strings)
        print(f"[HWNAS201] Loaded {self._n} archs for {self.dataset}/{self.hardware}")

    def evaluate(self, x):
        """Evaluate objectives for an architecture index.
        
        Args:
            x: Decision variable array (contains index).
        Returns: 
            Numpy array of objective values.
        """
        idx = int(np.clip(x[0], 0, self._n - 1))
        return np.array([-self._accuracies[idx], self._latencies[idx]])

    def arch_str(self, index):
        """Get the architecture string for a given index."""
        return self._arch_strings[int(index)]

    def pareto_front(self):
        """Brute-force nondominated sort over all architectures (internal space)."""
        all_objs = np.column_stack([-self._accuracies, self._latencies])
        valid = self._latencies > 0
        all_objs = all_objs[valid]
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