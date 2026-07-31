"""Compatibility facade for categorized FSDP2 model patches.

The implementation lives under ``fsdp2_runtime.patches``:
``performance`` contains retained throughput optimizations, ``profiling``
contains attribution-only ranges, ``vision`` contains the fixed-grid Vision
path, and ``adaptation`` contains FSDP2 dtype compatibility behavior.
"""

from .patches.adaptation import install_dtype_adaptation
from .patches.performance import (
    _install_qwen_fused_rms_norm,
    _install_text_fused_rope,
    install_legacy_model_config_compat,
    pack_qwen_text_mlps,
)
from .patches.dataset import install_dataset_repeat, install_video_capture_cache
from .patches.profiling import _install_attention_profile_ranges
from .patches.vision import (
    _install_fixed_grid_vision_attention,
    _install_flat_vision_inputs,
)
from .qwen_dataloader_preprocess import install_qwen_dataloader_preprocess
from .prefetch import install_async_h2d_prefetch
from .patches import vision_grid as _vision_grid  # noqa: F401


def apply_model_patches():
    """Install all selected performance, profiling, and compatibility patches."""
    import model.modeling_spirit_vla as spirit_model

    install_legacy_model_config_compat()
    install_dataset_repeat()
    install_video_capture_cache()
    install_qwen_dataloader_preprocess()
    install_async_h2d_prefetch()
    _install_qwen_fused_rms_norm()
    _install_text_fused_rope()
    _install_attention_profile_ranges()
    _install_fixed_grid_vision_attention()
    _install_flat_vision_inputs(spirit_model)
    install_dtype_adaptation()


__all__ = ["apply_model_patches", "pack_qwen_text_mlps"]
