"""Run NSGA-II on HW-NAS-201 across all datasets and hardware targets.

Usage:
    python examples/run_hw_nas201.py
    python examples/run_hw_nas201.py --dataset cifar100
    python examples/run_hw_nas201.py --hardware edgegpu edgetpu --gens 50
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.hw_nas_201.config import DATASETS
from benchmarks.hw_nas_201.run_hw_benchmark import run_hw_benchmark


def main():
    """Parse arguments and execute HW-NAS-201 benchmark across datasets.
    
    Returns: 
        Dictionary of aggregated metric results per dataset.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(description="HW-NAS-201 example — all datasets & devices")
    parser.add_argument("--dataset", type=str, nargs="*", default=None,
                        help="Datasets to run (default: all)")
    parser.add_argument("--hardware", type=str, nargs="*", default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    args = parser.parse_args()

    datasets = args.dataset or DATASETS
    all_results = {}

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'#'*60}")

        results = run_hw_benchmark(
            dataset=dataset,
            hardwares=args.hardware,
            n_runs=args.runs,
            generations=args.gens,
            pop_size=args.pop_size,
        )
        all_results[dataset] = results

    return all_results


if __name__ == "__main__":
    main()