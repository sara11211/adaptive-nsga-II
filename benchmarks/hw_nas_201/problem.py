"""HW-NAS-201 multi-objective problem.

f1 = -accuracy (negated for minimisation)
f2 = latency

Encoding: 6 genes, each with 5 choices (NAS-Bench-201 cell operations).
  Gene 0: operation on edge input(0)       -> intermediate node 0
  Gene 1: operation on edge input(0)       -> intermediate node 1
  Gene 2: operation on edge intermediate_0 -> intermediate node 1
  Gene 3: operation on edge input(0)       -> intermediate node 2
  Gene 4: operation on edge intermediate_0 -> intermediate node 2
  Gene 5: operation on edge intermediate_1 -> intermediate node 2
"""

from __future__ import annotations
import csv
import os
import numpy as np

from benchmarks.hw_nas_201.config import DATA, DATASETS

_data_cache = {}

# NAS-Bench-201 operations
OPS = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
OP_TO_IDX = {op: i for i, op in enumerate(OPS)}
N_OPS = len(OPS)  # 5


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


def _parse_arch_string(arch_str):
    """Parse NAS-Bench-201 architecture string into 6 operation indices.

    Format: |op~src|+|op~src|op~src|+|op~src|op~src|op~src|
    3 intermediate nodes with 1 + 2 + 3 = 6 edges total.

    Returns:
        Tuple of 6 integers (0-4), one per operation.
    """
    nodes = arch_str.strip('|').split('|+|')
    ops = []
    for node_str in nodes:
        parts = node_str.split('|')
        for part in parts:
            if '~' in part:
                op_name = part.split('~')[0]
                ops.append(OP_TO_IDX[op_name])
    return tuple(ops)


# ── Problem Class ──────────────────────────────────────────────────────────────
class HWNAS201:
    """HW-NAS-201 search problem with cell-level encoding.

    Each individual is a 6-gene vector where each gene selects one of 5
    NAS-Bench-201 operations for a specific edge in the cell DAG.
    """

    def __init__(self, hardware, dataset, lookup_dir=None):
        """Args:
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

        self.n_var = 6
        self.n_obj = 2
        self.num_choices = np.array([N_OPS] * self.n_var)

    def _load_data(self):
        """Load and cache architecture data from CSV.

        Builds a mapping from 6-gene encoding tuples to CSV row indices
        by parsing each architecture string.
        """
        cache_key = (self.dataset, self.hardware)
        if cache_key in _data_cache:
            (self._accuracies, self._latencies, self._arch_strings,
             self._encoding_to_idx) = _data_cache[cache_key]
            return

        csv_path = os.path.join(self.lookup_dir, f"nas201_{self.dataset}.csv")
        header = _read_header(csv_path)
        hw_idx = header.index(f"{self.hardware}_latency_ms")

        accs, lats, strs = [], [], []
        encoding_to_idx = {}

        with open(csv_path) as f:
            next(f)
            for idx, row in enumerate(csv.reader(f)):
                accs.append(float(row[2]))
                lats.append(float(row[hw_idx]))
                strs.append(row[1])
                encoding = _parse_arch_string(row[1])
                encoding_to_idx[encoding] = idx

        self._accuracies = np.array(accs)
        self._latencies = np.array(lats)
        self._arch_strings = strs
        self._encoding_to_idx = encoding_to_idx
        self._n = len(accs)

        _data_cache[cache_key] = (
            self._accuracies, self._latencies, self._arch_strings,
            self._encoding_to_idx,
        )
        print(f"[HWNAS201] Loaded {self._n} archs for {self.dataset}/{self.hardware}")
        print(f"[HWNAS201] Encoding map: {len(self._encoding_to_idx)} entries "
              f"(expected {N_OPS ** 6})")

    def evaluate(self, x):
        """Evaluate objectives for a 6-gene cell encoding.

        Args:
            x: Decision variable array of length 6 (integers 0-4).
        Returns:
            Numpy array [-accuracy, latency]. Returns [inf, inf] for invalid
            encodings or architectures with non-positive latency.
        """
        key = tuple(int(v) for v in x)
        if key not in self._encoding_to_idx:
            return np.array([np.inf, np.inf])

        idx = self._encoding_to_idx[key]
        lat = self._latencies[idx]
        if lat <= 0:
            return np.array([np.inf, np.inf])

        return np.array([-self._accuracies[idx], lat])

    def arch_str(self, x):
        """Get the architecture string for a given 6-gene encoding.

        Args:
            x: Decision variable array of length 6.
        Returns:
            Architecture string, or None if encoding is invalid.
        """
        key = tuple(int(v) for v in x)
        if key not in self._encoding_to_idx:
            return None
        idx = self._encoding_to_idx[key]
        return self._arch_strings[idx]

    def pareto_front(self):
        """Brute-force nondominated sort over all architectures."""
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