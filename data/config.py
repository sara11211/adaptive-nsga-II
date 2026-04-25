import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_BENCH_REPO = os.path.join(PROJECT_ROOT, "HW-NAS-Bench")

PICKLE_PATH = os.path.join(HW_BENCH_REPO, "HW-NAS-Bench-v1_0.pickle")
PTH_PATH = os.path.join(PROJECT_ROOT, "data", "NAS-Bench-201-v1_1-096897.pth")
ACC_PATH = os.path.join(PROJECT_ROOT, "data", "nas201_accuracies.npz")
ARCH_STRINGS_PATH = os.path.join(PROJECT_ROOT, "data", "nas201_arch_strings.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "lookup")