"""Runtime controls for Qwen Text/Vision activation recomputation."""

from __future__ import annotations

import os


MAX_TEXT_RECOMPUTE_LAYERS = 36
MAX_VISION_RECOMPUTE_LAYERS = 24


def _env_layer_count(name: str, default: int, maximum: int) -> int:
    value = os.environ.get(name)
    try:
        count = default if value is None else int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    if not 0 <= count <= maximum:
        raise ValueError(
            f"{name} must be between 0 and {maximum}, got {count}"
        )
    return count


def _configure_qwen_recompute(
    qwen,
    *,
    text_recompute_layers: int,
    vision_recompute_layers: int,
) -> tuple[list[int], list[int], int, int]:
    """Enable recompute on the first N Text and Vision blocks."""
    modules = list(qwen.modules())
    text_blocks = [
        module
        for module in modules
        if type(module).__name__ == "Qwen3VLTextDecoderLayer"
    ]
    vision_blocks = [
        module
        for module in modules
        if type(module).__name__ == "Qwen3VLVisionBlock"
    ]

    if not text_blocks or not vision_blocks:
        raise RuntimeError(
            "Could not find Qwen checkpointable layers: "
            f"text_layers={len(text_blocks)} "
            f"vision_layers={len(vision_blocks)}"
        )
    if text_recompute_layers > len(text_blocks):
        raise ValueError(
            "Requested more Text recompute layers than the model contains: "
            f"requested={text_recompute_layers} total={len(text_blocks)}"
        )
    if vision_recompute_layers > len(vision_blocks):
        raise ValueError(
            "Requested more Vision recompute layers than the model contains: "
            f"requested={vision_recompute_layers} total={len(vision_blocks)}"
        )

    text_indexes = list(range(text_recompute_layers))
    vision_indexes = list(range(vision_recompute_layers))
    for index, module in enumerate(text_blocks):
        module.gradient_checkpointing = index < text_recompute_layers
    for index, module in enumerate(vision_blocks):
        module.gradient_checkpointing = index < vision_recompute_layers

    # These parent flags do not checkpoint the parent forward themselves, but
    # keeping them aligned makes model.is_gradient_checkpointing truthful.
    for module in modules:
        module_type = type(module).__name__
        if module_type == "Qwen3VLTextModel":
            module.gradient_checkpointing = text_recompute_layers > 0
        elif module_type == "Qwen3VLVisionModel":
            module.gradient_checkpointing = vision_recompute_layers > 0

    return (
        text_indexes,
        vision_indexes,
        len(text_blocks),
        len(vision_blocks),
    )


def install_selective_qwen_recompute() -> None:
    """Monkey-patch Qwen's enable call with independent Text/Vision controls."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    if getattr(
        Qwen3VLForConditionalGeneration,
        "_spirit_selective_recompute_installed",
        False,
    ):
        return

    text_recompute_layers = _env_layer_count(
        "SPIRIT_QWEN_TEXT_RECOMPUTE_LAYERS",
        MAX_TEXT_RECOMPUTE_LAYERS,
        MAX_TEXT_RECOMPUTE_LAYERS,
    )
    vision_recompute_layers = _env_layer_count(
        "SPIRIT_QWEN_VISION_RECOMPUTE_LAYERS",
        MAX_VISION_RECOMPUTE_LAYERS,
        MAX_VISION_RECOMPUTE_LAYERS,
    )
    original_enable = (
        Qwen3VLForConditionalGeneration.gradient_checkpointing_enable
    )

    def selective_gradient_checkpointing_enable(self, *args, **kwargs):
        result = original_enable(self, *args, **kwargs)
        (
            text_indexes,
            vision_indexes,
            text_total,
            vision_total,
        ) = _configure_qwen_recompute(
            self,
            text_recompute_layers=text_recompute_layers,
            vision_recompute_layers=vision_recompute_layers,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[QWEN_SELECTIVE_RECOMPUTE] "
                f"text_recompute={len(text_indexes)}/{text_total} "
                f"text_indexes={text_indexes} "
                f"vision_recompute={len(vision_indexes)}/{vision_total} "
                f"vision_indexes={vision_indexes} "
                "selection=first_n "
                "checkpoint_impl=non_reentrant",
                flush=True,
            )
        return result

    Qwen3VLForConditionalGeneration.gradient_checkpointing_enable = (
        selective_gradient_checkpointing_enable
    )
    Qwen3VLForConditionalGeneration._spirit_selective_recompute_installed = (
        True
    )

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[QWEN_SELECTIVE_RECOMPUTE_CONFIG] "
            f"text_recompute_layers={text_recompute_layers} "
            f"text_max={MAX_TEXT_RECOMPUTE_LAYERS} "
            f"vision_recompute_layers={vision_recompute_layers} "
            f"vision_max={MAX_VISION_RECOMPUTE_LAYERS}",
            flush=True,
        )
