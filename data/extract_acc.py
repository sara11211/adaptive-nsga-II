"""Extract NAS-Bench-201 accuracies from the .pth file."""

import sys, gc, os
import numpy as np
import psutil
from config import HW_BENCH_REPO, PTH_PATH, ACC_PATH

# Allow local imports
sys.path.insert(0, HW_BENCH_REPO)

def main():
    """Load the .pth data, extract test accuracies per dataset, and save to .npz."""
    import torch
    from nas_201_api.api_utils import ArchResults

    # Patch torch.load for compatibility
    _orig = torch.load
    torch.load = lambda f, **kw: _orig(f, weights_only=False, map_location="cpu")

    if not os.path.exists(PTH_PATH):
        raise FileNotFoundError(
            f".pth not found at {PTH_PATH}\n"
            f"Download: https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_"
        )

    # ── Loading ─────────────────────────────────────────────────────────────────
    ram = psutil.virtual_memory()
    print(
        f"Loading {PTH_PATH} .....\n"
        f"RAM usage: {ram.used / 1024**3:.1f} GB used / {ram.total / 1024**3:.1f} GB total "
        f"({ram.available / 1024**3:.1f} GB available)"
    )
    data = torch.load(PTH_PATH)
    print("Loaded. Extracting accuracies...")

    # ── Extraction ──────────────────────────────────────────────────────────────
    results = {}
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        split = "ori-test" if dataset == "cifar10" else "x-test"
        accs = []
        for idx in sorted(data["arch2infos"].keys()):
            try:
                ar = ArchResults.create_from_state_dict(data["arch2infos"][idx]["full"])
                info = ar.get_metrics(dataset, split, None, False)
                accs.append(info["accuracy"])
            except Exception:
                accs.append(0.0)
        results[dataset] = np.array(accs)
        print(f"  {dataset}: {len(accs)} architectures")

    del data
    gc.collect()

    # ── Saving ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ACC_PATH), exist_ok=True)
    np.savez_compressed(ACC_PATH, **results)
    print(f"\nSaved to {ACC_PATH} ({os.path.getsize(ACC_PATH):,} bytes)")


if __name__ == "__main__":
    main()