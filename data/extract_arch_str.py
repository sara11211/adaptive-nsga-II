"""Extract NAS-Bench-201 architecture strings from the HW-NAS-Bench API.

Saves a lookup array where index i gives the arch string for architecture i.
"""

import sys, os
import numpy as np
from config import HW_BENCH_REPO, PICKLE_PATH, ARCH_STRINGS_PATH

# Allow local imports
sys.path.insert(0, HW_BENCH_REPO)


def main():
    """Load HW-NAS-Bench API, extract arch strings, and save to .npy."""
    from hw_nas_bench_api import HWNASBenchAPI as HWAPI

    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(f"Pickle not found at {PICKLE_PATH}")

    # ── Loading ─────────────────────────────────────────────────────────────────
    print("Loading HW-NAS-Bench API...")
    hw_api = HWAPI(PICKLE_PATH, search_space="nasbench201")

    # Determine number of architectures from API 
    nb = hw_api.HW_metrics[hw_api.search_space]
    first_dataset = list(nb.keys())[0]
    first_metric = [k for k in nb[first_dataset].keys() if k.endswith("_latency")][0]
    n_architectures = len(nb[first_dataset][first_metric])
    print(f"Architectures found: {n_architectures}")

    # ── Extraction ──────────────────────────────────────────────────────────────
    arch_strings = []
    for idx in range(n_architectures):
        config = hw_api.get_net_config(idx, first_dataset)
        arch_strings.append(config["arch_str"])
        if (idx + 1) % 5000 == 0:
            print(f"  {idx + 1}/{n_architectures}...")

    # ── Saving ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ARCH_STRINGS_PATH), exist_ok=True)
    np.save(ARCH_STRINGS_PATH, arch_strings)
    print(f"\nSaved to {ARCH_STRINGS_PATH}")
    print(f"Example — index 0:   {arch_strings[0]}")
    print(f"Example — index 100: {arch_strings[100]}")


if __name__ == "__main__":
    main()