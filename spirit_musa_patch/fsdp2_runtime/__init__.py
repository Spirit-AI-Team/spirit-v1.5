"""Opt-in FSDP2 runtime for the existing Spirit training entrypoints."""

from .checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_step,
    save_model,
)
from .compile_compat import install_torch_compile_compat
from .distributed import apply_fsdp, cleanup, setup_distributed
from .model_patches import apply_model_patches, pack_qwen_text_mlps

__all__ = [
    "apply_fsdp",
    "apply_model_patches",
    "cleanup",
    "install_torch_compile_compat",
    "load_training_checkpoint",
    "read_training_checkpoint_step",
    "pack_qwen_text_mlps",
    "save_model",
    "setup_distributed",
]
