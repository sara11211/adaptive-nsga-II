"""Run NSGA-II on all ZDT benchmark problems.

Usage:
    python examples/run_zdt.py
    python examples/run_zdt.py --problems ZDT1 ZDT3
    python examples/run_zdt.py --gens 50 --runs 3
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.zdt.run_zdt_benchmark import run_zdt_benchmark


def main():
    """Parse arguments and execute ZDT benchmark for all selected problems.
    
    Returns: 
        Dictionary of aggregated metric results.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(description="ZDT example — all problems")
    parser.add_argument("--problems", type=str, nargs="*", default=None,
                        help="Problems to run (default: all)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--gens", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    args = parser.parse_args()

    results = run_zdt_benchmark(
        problems=args.problems,
        n_runs=args.runs,
        generations=args.gens,
        pop_size=args.pop_size,
    )
    return results


if __name__ == "__main__":
    main()