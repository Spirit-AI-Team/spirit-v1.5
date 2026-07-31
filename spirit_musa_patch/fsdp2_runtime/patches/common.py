"""Shared helpers for categorized runtime patches."""

import os
from contextlib import nullcontext


def profile_range(name: str):
    """Return an opt-in profiler range with zero normal-run behavior."""
    if int(os.environ.get("SPIRIT_PROFILE_RANGES", "0")) != 1:
        return nullcontext()
    import torch

    return torch.profiler.record_function(name)
