# Dynamic NSGA-II with Self-Adaptive Mutation for NAS

An NSGA-II implementation with a self-adaptive mutation pool. Three complementary strategies (conservative, uniform, aggressive): their selection probabilities evolve each generation based on Pareto-dominance feedback, letting the algorithm automatically balance exploration and exploitation. Evaluated on **HW-NAS-201**, optimizing accuracy vs. inference latency across multiple edge devices.

## Adaptive Mutation Mechanism

Instead of a fixed mutation operator, the algorithm maintains three strategies whose selection probabilities update periodically based on their observed effectiveness. Each generation follows this cycle:

<p align="center">
  <img src="https://i.imgur.com/RjCTZpV.png" alt="Dynamic Mutation Flowchart" width="85%"/>
</p>

### Mutation Strategies

| Strategy | Mutates | Behavior |
|---|---|---|
| **Conservative** | 1 gene to neighbor (+/-1 mod n) | Exploitation |
| **Uniform** | 1 gene to any other value | Balanced |
| **Aggressive** | 2-3 genes simultaneously | Exploration |

Default initial probabilities: `[0.3, 0.4, 0.3]`, updated every 5 generations.

## Project Structure

```
├── nsga2/                          # Core framework
│   ├── core/
│   │   ├── nsga2.py                # NSGA-II optimizer with adaptive mutation
│   │   ├── adaptive_mutation.py    # Self-adaptive mutation pool
│   │   ├── crossover.py            # Uniform crossover
│   │   ├── mutation.py             # Standard discrete mutation (baseline)
│   │   ├── nondominated_sort.py    # Fast non-dominated sorting
│   │   ├── crowding_distance.py    # Crowding distance assignment
│   │   ├── selection.py            # Binary tournament selection
│   │   └── individual.py           # Individual (vars, objectives, rank, CD)
│   ├── metrics/                    # GD, IGD, IGD+, HV, Spread
│   ├── visualization/              # Plotting utilities
│   ├── problems/                   # OneZeroMax test problem
│   └── utils/                      # Helper utilities
├── benchmarks/hw_nas_201/          # HW-NAS-201 benchmark
├── data/lookup/                    # Pre-computed CSV lookup tables
├── examples/                       # Usage scripts
└── tests/                          # Unit tests
```

## Installation

```bash
git clone https://github.com/sara11211/dynamic-nsga-II.git
cd dynamic-nsga-II
pip install -r requirements.txt
```

Verify with `pytest tests/ -v`.

## Benchmark: HW-NAS-201

```bash
# Run on specific dataset and hardware
python examples/run_hw_nas201.py --dataset cifar10 --hardware edgegpu edgetpu --gens 100

# Run all datasets and hardware targets
python examples/run_hw_nas201.py
```

### Baseline vs Adaptive Comparison

```bash
# Full comparison (all datasets, all hardware)
python examples/run_comparison.py

# Specific targets
python examples/run_comparison.py --dataset cifar10 cifar100 --hardware edgegpu raspi4 --gens 50
```

Generates per-hardware convergence plots, Pareto front overlays, strategy probability evolution, comparison reports (TXT/CSV), and an overall summary.

## Configuration

Parameters are in `benchmarks/hw_nas_201/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 50 | Population size |
| `generations` | 100 | Max generations |
| `n_runs` | 5 | Independent runs for statistics |
| `seed_base` | 42 | Random seed base |
| `prob_crossover` | 0.9 | Crossover probability |
| `initial_probs` | [0.3, 0.4, 0.3] | Strategy selection probabilities |
| `update_period` | 5 | Generations between probability updates |

All overridable via CLI arguments.

## Performance Metrics

| Metric | Direction | What it measures |
|---|---|---|
| **GD** | Lower is better | Convergence to the true Pareto front |
| **IGD** | Lower is better | Convergence + coverage of the true front |
| **IGD+** | Lower is better | Strict convergence (penalizes only worse dimensions) |
| **HV** | Higher is better | Volume dominated by the obtained front |
| **Spread** | Lower is better | Uniformity of solution distribution along the front |

## Results

18 dataset-hardware combinations, 5 independent runs each:

| Metric | Improved | Worse | No Change |
|---|---|---|---|
| **IGD+** (Convergence) | 12/18 | 4/18 | 2/18 |
| **Spread** (Diversity) | 15/18 | 3/18 | 0/18 |

## References

1. K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE TEVC*, 2002.
2. B. Xue et al., "A Self-Adaptive Mutation Neural Architecture Search Algorithm Based on Blocks," *IEEE CIM*, 2021.

---
_**Contributors:** ABAZIZ Sarah & SEBBAH Sarah Farah_
