# ── Algorithm Parameters ───────────────────────────────────────────────────────
ALGORITHM = {
    "pop_size": 50,
    "generations": 100,
    "n_runs": 5,
    "seed_base": 42,
    "prob_crossover": 0.9,
    "eta_c": 20.0,
    "eta_m": 20.0,
    "ref_offset": 0.1,
}

# ── Problem Selection ──────────────────────────────────────────────────────────
PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]

# ── Output Paths ───────────────────────────────────────────────────────────────
OUTPUT = {
    "base_dir": "output",
    "subfolder": "zdt",
    "subdirs": {
        "pareto": "pareto",
        "convergence": "convergence",
        "diversity": "diversity",
    },
}

# ── Plot Labels ────────────────────────────────────────────────────────────────
PLOT = {
    "xlabel": "$f_1$",
    "ylabel": "$f_2$",
}