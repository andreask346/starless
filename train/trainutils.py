# Shared robustness helpers for unattended training (both train.py and
# train_sharp.py). Fixes from the pre-run adversarial review:
#  - atomic checkpoint save (crash/STOP mid-write can't corrupt the only ckpt)
#  - NaN/inf handling that EXITS NONZERO so the queue's retry+resume fires
#  - best-metric persistence across resume

import os
import sys

import torch


def atomic_save(obj, path):
    """torch.save to a temp file then os.replace — never leaves a partial
    checkpoint if the process is killed mid-write."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path, device):
    """Load a checkpoint, tolerating a corrupt current file by falling back to
    the rotated previous one. Returns (state_dict_bundle | None)."""
    for p in (path, path + ".prev"):
        if os.path.isfile(p):
            try:
                return torch.load(p, map_location=device, weights_only=True)
            except Exception as e:
                print(f"checkpoint {p} unreadable ({e}); trying fallback",
                      flush=True)
    return None


class NaNGuard:
    """Skip non-finite steps; abort (exit 1) after too many consecutive ones
    so the queue retries from the last good checkpoint instead of training on
    NaN for days."""

    def __init__(self, max_consecutive=25):
        self.max = max_consecutive
        self.streak = 0

    def check(self, loss):
        import math
        v = loss.item()
        if math.isfinite(v):
            self.streak = 0
            return True
        self.streak += 1
        print(f"WARNING: non-finite loss ({v}), skipping step "
              f"({self.streak}/{self.max})", flush=True)
        if self.streak >= self.max:
            print("ABORT: too many non-finite steps — exiting 1 so the queue "
                  "resumes from the last good checkpoint", flush=True)
            sys.exit(1)
        return False
