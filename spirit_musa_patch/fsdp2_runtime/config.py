"""Integer configuration contract for the FSDP2 runtime.

Launch scripts intentionally use numeric values so a run can be reproduced
from one compact configuration.  The helpers below are the single place where
those values are decoded.  Legacy words are accepted only for older launch
files and are never emitted by the current launchers.

Boolean switches use 0=off and 1=on.

Enumerations:
  FSDP2_MODE: 0=auto, 1=fsdp, 2=hsdp
  FSDP2_WRAP_LEVEL: 0=coarse, 1=block
  OPTIMIZER: 0=AdamW, 1=fused AdamW
  GA_SYNC_MODE: 0=no_sync, 1=all_reduce
  GRAD_CLIP_IMPL: 0=reference, 1=HSDP foreach
  QWEN *_IMPL: 0=reference, 1=MUSA fused
  VISION_ATTN_LAYOUT: 0=varlen, 1=dense fixed-grid
  VISION_QKV_LAYOUT: 0=views, 1=packed contiguous
  PATCH_EMBED_IMPL: 0=conv3d, 1=GEMM
"""

from __future__ import annotations

import os


def int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    value = default if raw is None or raw.strip() == "" else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def flag(name: str, default: int = 0) -> bool:
    value = int_env(name, default)
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {value}")
    return bool(value)


def choice(
    name: str,
    default: int,
    labels: tuple[str, ...],
    legacy: dict[str, int] | None = None,
) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        token = raw.strip().lower()
        try:
            value = int(token)
        except ValueError:
            if legacy is None or token not in legacy:
                raise ValueError(
                    f"{name} must be an integer in 0..{len(labels) - 1}; "
                    f"got {raw!r}"
                ) from None
            value = legacy[token]
        if value < 0 or value >= len(labels):
            raise ValueError(
                f"{name} must be an integer in 0..{len(labels) - 1}; "
                f"got {value}"
            )
    return value


def label(
    name: str,
    default: int,
    labels: tuple[str, ...],
    legacy: dict[str, int] | None = None,
) -> str:
    return labels[choice(name, default, labels, legacy)]

