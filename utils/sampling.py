# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

from typing import Optional

import torch


def _resolve_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """If caller passes None, follow the active autocast dtype (if any),
    else fall back to float32 to preserve existing behaviour."""
    if dtype is not None:
        return dtype
    if torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    return torch.float32


def sample_beta(alpha: float, beta: float, bsize: int, device) -> torch.Tensor:
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    return m.sample((bsize,)).to(device).reshape((bsize,))


def sample_noise(
    shape, device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Draw standard-normal noise.

    Args:
        shape:  noise tensor shape
        device: target device
        dtype:  desired dtype. If ``None`` and we're inside an autocast
            context, follows the autocast dtype to avoid mismatches
            downstream (e.g. bf16 DiT ``action_in_proj``). Otherwise
            defaults to ``torch.float32``, preserving the original behaviour.
    """
    return torch.normal(
        mean=0.0, std=1.0, size=shape,
        dtype=_resolve_dtype(dtype), device=device,
    )


def sample_time(
    bsize: int, device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Draw flow-matching timesteps in (0, 1).

    Args:
        bsize:  batch size
        device: target device
        dtype:  desired dtype, see :func:`sample_noise`.
    """
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=_resolve_dtype(dtype), device=device)
