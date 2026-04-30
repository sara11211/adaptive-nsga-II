import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Algorithm Parameters ───────────────────────────────────────────────────────
ALGORITHM = {
    "pop_size": 50,
    "generations": 100,
    "n_runs": 5,
    "seed_base": 42,
    "prob_crossover": 0.9,
    "ref_offset": 0.01,
}

# ── Adaptive Mutation (SaMuNet) ───────────────────────────────────────────────
# Set to None to disable.  When enabled, replaces standard discrete_mutation
# with a self-adaptive 3-strategy pool (conservative / uniform / aggressive).
ADAPTIVE_MUTATION = {
    "initial_probs": [0.3, 0.4, 0.3],   # conservative, uniform, aggressive
    "update_period": 5,                    # recompute every N generations
}

# ── Dataset Settings ───────────────────────────────────────────────────────────
DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
DEFAULT_DATASET = "cifar10"

DATA = {
    "lookup_dir": os.path.join(PROJECT_ROOT, "data", "lookup"),
}

# ── Output Paths ───────────────────────────────────────────────────────────────
OUTPUT = {
    "base_dir": os.path.join(PROJECT_ROOT, "output", "hw_nas_201"),
    "subdirs": {
        "pareto": "pareto_fronts",
        "convergence": "convergence",
        "diversity": "diversity",
    },
}

# ── Metrics & Plot Labels ──────────────────────────────────────────────────────
METRICS = ["GD", "IGD", "IGD+", "HV", "Spread"]

PLOT = {
    "xlabel": "Accuracy (%)",
    "ylabel": "Latency (ms)",
}