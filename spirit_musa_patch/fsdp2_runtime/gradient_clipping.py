"""HSDP-aware gradient clipping with a conservative reference fallback.

FSDP2 exposes gradients as DTensors.  Calling the generic PyTorch helper on
those tensors can dispatch one norm/reduction per DTensor and make clipping a
visible synchronization point.  This path computes norms on each local shard,
then reduces one scalar per mesh shard group.  It deliberately rejects
placements whose global norm semantics are ambiguous and falls back to the
PyTorch implementation.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from .config import choice


_REFERENCE_CLIP = torch.nn.utils.clip_grad_norm_
_LOGGED = False
_FALLBACK_LOGGED = False
_ACTIVE_LOGGED = False


def _mode() -> str:
    return ("reference", "hsdp_foreach")[choice(
        "SPIRIT_GRAD_CLIP_IMPL", 1,
        ("reference", "hsdp_foreach"),
        {"reference": 0, "torch": 0, "default": 0,
         "hsdp_foreach": 1, "hsdp": 1},
    )]


class _UnsupportedPlacement(Exception):
    pass


def _reference_fallback(
    params, max_norm, norm_type, error_if_nonfinite, foreach, reason
):
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED and int(os.environ.get("RANK", "0")) == 0:
        print(f"[GRAD_CLIP] fallback=reference reason={reason}", flush=True)
        _FALLBACK_LOGGED = True
    return _REFERENCE_CLIP(params, max_norm, norm_type, error_if_nonfinite, foreach)


def _mesh_group_and_key(grad: DTensor):
    placements = tuple(grad.placements)
    if any(isinstance(p, Partial) for p in placements):
        raise _UnsupportedPlacement("Partial gradient placement")
    shard_dims = [i for i, p in enumerate(placements) if isinstance(p, Shard)]
    if len(shard_dims) > 1:
        raise _UnsupportedPlacement("multiple Shard placements")
    mesh = grad.device_mesh
    if not shard_dims:
        return None, None
    dim = shard_dims[0]
    if mesh.ndim == 1:
        group = mesh.get_group()
    else:
        # HSDP's parameter layout is Replicate x Shard.  Reduce only over the
        # shard axis; replica ranks already hold the same shard gradient.
        if any(i != dim and not isinstance(placements[i], Replicate) for i in range(mesh.ndim)):
            raise _UnsupportedPlacement("non-replicated HSDP axis")
        group = mesh[mesh.mesh_dim_names[dim]].get_group()
    return group, (id(mesh), dim)


def _sum_squares(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise RuntimeError("empty tensor list")
    buckets = defaultdict(list)
    for tensor in tensors:
        buckets[(tensor.device, tensor.dtype)].append(tensor)
    total = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for bucket in buckets.values():
        try:
            norms = torch._foreach_norm(bucket, 2.0)
            total += torch.stack([n.float() for n in norms]).square().sum()
        except (AttributeError, RuntimeError):
            total += torch.stack(
                [torch.linalg.vector_norm(t.float()) for t in bucket]
            ).square().sum()
    return total


def _local_grad(param):
    grad = getattr(param, "grad", None)
    if grad is None:
        return None
    if isinstance(grad, DTensor):
        return grad.to_local(), grad
    if not isinstance(grad, torch.Tensor):
        raise _UnsupportedPlacement(f"unsupported gradient type {type(grad)!r}")
    return grad, None


@torch.no_grad()
def hsdp_foreach_clip_grad_norm_(
    parameters,
    max_norm,
    norm_type=2.0,
    error_if_nonfinite=False,
    foreach=None,
):
    """Clip DTensor/local gradients while reducing only the shard axis.

    The function has the same public contract as ``torch.nn.utils``.  Any
    unsupported layout or norm type is handled by the reference helper before
    this function mutates gradients.
    """
    if norm_type not in (2, 2.0, "2", float("2")):
        return _reference_fallback(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            f"norm_type={norm_type}",
        )
    params = list(parameters)
    entries = []
    try:
        for param in params:
            grad_entry = _local_grad(param)
            if grad_entry is None:
                continue
            local, dt = grad_entry
            group, key = (
                _mesh_group_and_key(dt)
                if dt is not None
                else (None, None)
            )
            entries.append((local, dt, group, key))
    except _UnsupportedPlacement as error:
        return _reference_fallback(
            params,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            str(error),
        )
    if not entries:
        return torch.tensor(0.0)
    devices = {local.device for local, _, _, _ in entries}
    if len(devices) != 1:
        return _reference_fallback(
            params,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            f"mixed_gradient_devices={sorted(map(str, devices))}",
        )

    # Aggregate local squared norms by process group, avoiding one collective
    # per parameter.  Local (non-DTensor) gradients are already replicated.
    contributions = defaultdict(list)
    local_entries = []
    first_device = entries[0][0].device
    for local, dt, group, key in entries:
        if dt is None:
            local_entries.append(local)
        else:
            contributions[(key, group)].append(local)
    global _ACTIVE_LOGGED
    if not _ACTIVE_LOGGED and int(os.environ.get("RANK", "0")) == 0:
        print(
            "[GRAD_CLIP] active=hsdp_foreach "
            f"grads={len(entries)} dtensor_grads="
            f"{sum(dt is not None for _, dt, _, _ in entries)} "
            f"shard_groups={len(contributions)}",
            flush=True,
        )
        _ACTIVE_LOGGED = True
    total_sq = torch.zeros((), device=first_device, dtype=torch.float32)
    if local_entries:
        total_sq += _sum_squares(local_entries).to(device=first_device)
    for (key, group), tensors in contributions.items():
        part = _sum_squares(tensors).to(device=first_device)
        if group is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(part, op=dist.ReduceOp.SUM, group=group)
        total_sq += part

    total_norm = total_sq.sqrt()
    if error_if_nonfinite and not bool(torch.isfinite(total_norm).item()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite"
        )
    clip_coef = torch.clamp(
        torch.as_tensor(max_norm, device=first_device, dtype=torch.float32)
        / (total_norm + 1.0e-6), max=1.0,
    )
    # Always multiply by the clamped coefficient, as PyTorch does.  Branching
    # on ``clip_coef.item()`` would add a device-to-host synchronization.
    scale_buckets = defaultdict(list)
    for local, _, _, _ in entries:
        scale_buckets[(local.device, local.dtype)].append(local)
    for bucket in scale_buckets.values():
        try:
            torch._foreach_mul_(bucket, clip_coef)
        except (AttributeError, RuntimeError):
            for tensor in bucket:
                tensor.mul_(clip_coef.to(dtype=tensor.dtype))
    return total_norm.to(dtype=entries[0][0].dtype)


def install_gradient_clipping() -> str:
    """Install the selected implementation once and return its mode."""
    global _LOGGED
    if getattr(install_gradient_clipping, "_installed", False):
        return install_gradient_clipping._mode
    mode = _mode()
    selected = hsdp_foreach_clip_grad_norm_ if mode == "hsdp_foreach" else _REFERENCE_CLIP
    torch.nn.utils.clip_grad_norm_ = selected
    install_gradient_clipping._installed = True
    install_gradient_clipping._mode = mode
    if not _LOGGED and int(os.environ.get("RANK", "0")) == 0:
        print(f"[GRAD_CLIP] implementation={mode}", flush=True)
        _LOGGED = True
    return mode
