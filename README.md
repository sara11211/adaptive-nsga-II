# NSGA-II Implementation

**A Fast and Elitist Multiobjective Genetic Algorithm**

Implementation of the NSGA-II algorithm from the paper:

> K. Deb, A. Pratap, S. Agarwal, T. Meyarivan,
> "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
> IEEE Transactions on Evolutionary Computation, Vol. 6, No. 2, April 2002.

---

## Project Structure

```
nsga2/
├── core/                       # Core NSGA-II components
│   ├── individual.py           # Individual (solution) representation
│   ├── nondominated_sort.py    # Fast nondominated sorting  O(MN²)
│   ├── crowding_distance.py    # Crowding distance assignment O(MN log N)
│   ├── selection.py            # Binary tournament selection (crowded-comparison)
│   ├── crossover.py            # Simulated Binary Crossover (SBX)
│   ├── mutation.py             # Polynomial mutation
│   └── nsga2.py                # Main NSGA-II algorithm loop
│
├── problems/                   # Benchmark test problems
│   ├── zdt.py                  # ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
│   └── constrained.py          # CONSTR, SRN, TNK (constrained)
│
├── metrics/                    # Performance metrics
│   ├── gd.py                   # Generational Distance
│   ├── igd.py                  # Inverted Generational Distance
│   ├── hv.py                   # Hypervolume indicator
│   └── spread.py               # Spread (diversity) metric
│
├── visualization/              # Plotting utilities
│   ├── pareto_front.py         # Pareto front plots
│   └── convergence.py          # Convergence metric plots
│
└── utils/                      # Helpers
    └── helpers.py              # Nondominated filter

benchmarks/                     # Benchmark runners
├── run_zdt.py                  # Run all ZDT problems with metric collection
└── hw_nas_201/                 # HW-NAS-201 benchmark
    ├── problem.py              # HW-NAS-201 problem definition (5 hardware targets)
    └── run_hw_benchmark.py     # Run across all hardware platforms

tests/                          # Unit and integration tests
├── test_nondominated_sort.py
├── test_crowding_distance.py
├── test_operators.py           # Tests for selection, crossover, mutation
├── test_nsga2.py               # End-to-end algorithm tests
├── test_problems.py            # Benchmark problem tests
└── test_metrics.py             # Metric calculation tests

examples/                       # Usage examples
├── run_zdt1.py                 # Basic ZDT1 run with plots
├── run_constrained.py          # Constrained problem example
└── run_hw_nas201.py            # HW-NAS-201 across hardware targets

output/                         # Generated images and results
```

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Quick Start

```python
from nsga2.core.nsga2 import NSGA2
from nsga2.problems.zdt import ZDT1
from nsga2.visualization.pareto_front import plot_pareto_front

problem = ZDT1()

optimizer = NSGA2(
    problem=problem,
    pop_size=100,
    n_var=30,
    n_obj=2,
    bounds=problem.bounds,
    seed=42,
)

pareto_set, pareto_front = optimizer.run(generations=250)

plot_pareto_front(
    pareto_front,
    true_front=problem.pareto_front(),
    title="ZDT1 — NSGA-II",
)
```

## Running Benchmarks

```bash
# Run all ZDT problems (10 independent runs each)
python -m benchmarks.run_zdt --gens 250 --pop-size 100

# Run HW-NAS-201 across all hardware targets
python -m benchmarks.hw_nas_201.run_hw_benchmark --gens 100

# Run examples
python examples/run_zdt1.py
python examples/run_constrained.py
python examples/run_hw_nas201.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Algorithm Details

The implementation follows the paper closely:

| Component | Complexity | Description |
|-----------|-----------|-------------|
| Nondominated sorting | O(MN²) | Fast approach using domination counts |
| Crowding distance | O(MN log N) | Density estimation via nearest-neighbour cuboid |
| Tournament selection | O(1) per selection | Crowded-comparison operator |
| SBX crossover | O(N) per pair | Simulated Binary Crossover with eta_c |
| Polynomial mutation | O(N) | Per-variable mutation with eta_m |

## Supported Metrics

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| GD | Convergence to true Pareto front | Lower is better |
| IGD | Convergence + diversity | Lower is better |
| HV | Volume of dominated space | Higher is better |
| Spread | Uniformity of solution distribution | Higher is better |

## HW-NAS-201 Benchmark

Multi-objective neural architecture search across 5 hardware targets:

- **edgegpu** — Edge GPU
- **edgetpu** — Edge TPU
- **raspberry** — Raspberry Pi
- **jetson_nano** — NVIDIA Jetson Nano
- **phone** — Mobile phone

Two objectives: maximise accuracy, minimise latency.

The `problem.py` module uses synthetic surrogate data for demonstration.
Replace `_load_data()` with actual HW-NAS-201 dataset files for real experiments.
