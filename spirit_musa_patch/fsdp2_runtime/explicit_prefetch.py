"""Explicit FSDP2 layer prefetch for Spirit's repeated transformer stacks."""

from __future__ import annotations

import os
from collections.abc import Sequence

from .config import int_env


def _configure_stack(name: str, modules: Sequence[object], depth: int) -> tuple[int, int]:
    """Install forward-next and backward-previous prefetch edges for one stack.

    ``fully_shard`` attaches these methods to the wrapped child modules.  The
    edges must stay within one homogeneous sequential stack: prefetching a
    Vision block from Text (or from DiT) would not match the actual execution
    order and can retain unrelated parameters unnecessarily.
    """
    if not modules:
        return 0, 0

    required_methods = (
        "set_modules_to_forward_prefetch",
        "set_modules_to_backward_prefetch",
    )
    missing = [
        index
        for index, module in enumerate(modules)
        if not all(hasattr(module, method) for method in required_methods)
    ]
    if missing:
        raise RuntimeError(
            "Explicit FSDP prefetch requires every wrapped "
            f"{name} block to expose {required_methods}; missing indexes="
            f"{missing[:8]}"
        )

    forward_edges = 0
    backward_edges = 0
    for index, module in enumerate(modules):
        forward_targets = list(modules[index + 1:index + 1 + depth])
        if forward_targets:
            module.set_modules_to_forward_prefetch(forward_targets)
            forward_edges += len(forward_targets)

        backward_targets = list(modules[max(0, index - depth):index])
        if backward_targets:
            # Reverse the preceding range so the nearest block is requested
            # first, matching backward's reverse execution order.
            backward_targets.reverse()
            module.set_modules_to_backward_prefetch(backward_targets)
            backward_edges += len(backward_targets)

    return forward_edges, backward_edges


def install_explicit_prefetch(
    *,
    text_blocks: Sequence[object],
    vision_blocks: Sequence[object],
    dit_blocks: Sequence[object],
) -> None:
    """Apply ``FSDP_EXPLICIT_PREFETCH_LAYER_NUM`` after child FSDP wrapping.

    The setting is deliberately shared by the three sequential transformer
    stacks.  A zero value is an exact no-op, making it the rollback switch.
    """
    depth = int_env("FSDP_EXPLICIT_PREFETCH_LAYER_NUM", 0, minimum=0)
    if depth == 0:
        return

    stack_counts = (
        ("text", text_blocks),
        ("vision", vision_blocks),
        ("dit", dit_blocks),
    )
    edge_counts = {
        name: _configure_stack(name, modules, depth)
        for name, modules in stack_counts
    }

    if int(os.environ.get("RANK", "0")) == 0:
        summary = " ".join(
            f"{name}=blocks:{len(modules)},forward_edges:{edge_counts[name][0]},"
            f"backward_edges:{edge_counts[name][1]}"
            for name, modules in stack_counts
        )
        print(
            "[FSDP2_EXPLICIT_PREFETCH] "
            f"depth={depth} {summary}",
            flush=True,
        )
