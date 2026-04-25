"""Build NAS-Bench-201 lookup tables using the HW-NAS-Bench API.

One CSV per dataset containing index, arch_str, accuracy, and latency
for every hardware target.
"""

import sys, os, csv
import numpy as np
from config import HW_BENCH_REPO, PICKLE_PATH, ACC_PATH, ARCH_STRINGS_PATH, OUTPUT_DIR

# Allow local imports
sys.path.insert(0, HW_BENCH_REPO)

def main():
    """Load API data, merge with accuracies, and save as CSV lookup tables."""
    from hw_nas_bench_api import HWNASBenchAPI as HWAPI

    for path in [PICKLE_PATH, ACC_PATH, ARCH_STRINGS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run extract_accuracies.py and extract_arch_strings.py first."
            )
        
    # ── Loading ─────────────────────────────────────────────────────────────────
    print("Loading HW-NAS-Bench API...")
    hw_api = HWAPI(PICKLE_PATH, search_space="nasbench201")

    nb = hw_api.HW_metrics[hw_api.search_space]
    available_datasets = list(nb.keys())
    first_metric = [k for k in nb[available_datasets[0]].keys() if k.endswith("_latency")][0]
    n_architectures = len(nb[available_datasets[0]][first_metric])
    hardware_names = [
        k.replace("_latency", "")
        for k in nb[available_datasets[0]].keys()
        if k.endswith("_latency")
    ]

    print(f"Architectures: {n_architectures}")
    print(f"Datasets:      {available_datasets}")
    print(f"Hardware:      {hardware_names}")

    # Load pre-extracted data
    arch_strings = np.load(ARCH_STRINGS_PATH, allow_pickle=True)
    acc_data = np.load(ACC_PATH)
    datasets_to_process = [d for d in available_datasets if d in acc_data]
    print(f"Processing:    {datasets_to_process}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Processing & Saving ─────────────────────────────────────────────────────
    for dataset in datasets_to_process:
        print(f"\nProcessing {dataset}...")

        # Query API for latencies
        latencies = {hw: [] for hw in hardware_names}
        for idx in range(n_architectures):
            hw_info = hw_api.query_by_index(idx, dataset)
            for hw in hardware_names:
                latencies[hw].append(hw_info[f"{hw}_latency"])
            if (idx + 1) % 5000 == 0:
                print(f"  {idx + 1}/{n_architectures}...")

        output_csv = os.path.join(OUTPUT_DIR, f"nas201_{dataset}.csv")
        header = ["index", "arch_str", "accuracy_test"] + [
            f"{hw}_latency_ms" for hw in hardware_names
        ]

        # Write to CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in range(n_architectures):
                row = [idx, arch_strings[idx], f"{acc_data[dataset][idx]:.4f}"]
                row += [f"{latencies[hw][idx]:.6f}" for hw in hardware_names]
                writer.writerow(row)

        print(f"  Saved {output_csv} ({os.path.getsize(output_csv)/1024:.1f} KB)")

    print(f"\nAll lookup tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()