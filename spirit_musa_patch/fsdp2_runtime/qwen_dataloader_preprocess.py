"""CPU DataLoader preprocessing for Spirit's fixed-layout Qwen inputs."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer

from utils import get_rope_index_3, get_user_prompt, preprocess_qwen_visual


QWEN_PIXEL_VALUES = "__spirit_qwen_pixel_values"
QWEN_IMAGE_GRID_THW = "__spirit_qwen_image_grid_thw"
QWEN_INPUT_IDS = "__spirit_qwen_input_ids"
QWEN_POSITION_IDS = "__spirit_qwen_position_ids"
QWEN_ATTENTION_MASK = "__spirit_qwen_attention_mask"


def _backbone_from_training_arguments() -> str:
    """Read the Spirit checkpoint config without changing the train entrypoint."""
    override = os.environ.get("SPIRIT_QWEN_BACKBONE")
    if override:
        return override
    try:
        checkpoint_dir = sys.argv[sys.argv.index("--pretrained_path") + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(
            "Qwen DataLoader preprocessing needs --pretrained_path or "
            "SPIRIT_QWEN_BACKBONE"
        ) from error
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    backbone = config.get("backbone")
    if not isinstance(backbone, str) or not backbone:
        raise RuntimeError(f"Missing non-empty backbone in {config_path}")
    return backbone


class QwenTensorCollator:
    """Build a fully tensorized, pin-memory-compatible Qwen batch in workers.

    The tokenizer and processor are intentionally created lazily in the worker
    process.  The collator can therefore be constructed before the MUSA model
    without serializing model state to DataLoader workers.
    """

    def __init__(self, backbone: str):
        self.backbone = backbone
        self._tokenizer = None
        self._image_processor = None
        self._text_pos_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure_components(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.backbone,
                add_eos_token=False,
                trust_remote_code=True,
                use_fast=False,
            )
            self._image_processor = AutoProcessor.from_pretrained(
                self.backbone
            ).image_processor
        return self._tokenizer, self._image_processor

    @staticmethod
    def _collate_tensor_fields(batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([sample[key] for sample in batch])
            for key in batch[0]
            if isinstance(batch[0][key], torch.Tensor)
        }

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        tokenizer, image_processor = self._ensure_components()
        result = self._collate_tensor_fields(batch)
        image_keys = sorted(
            key for key in result if key.startswith("observation.images.")
        )
        if not image_keys:
            raise ValueError("No image fields found in batch")

        first_image = result[image_keys[0]]
        batch_size = first_image.shape[0]
        expected_shape = (batch_size, 3, 240, 320)
        if any(tuple(result[key].shape) != expected_shape for key in image_keys):
            found_shapes = {key: tuple(result[key].shape) for key in image_keys}
            raise ValueError(
                "QwenTensorCollator requires fixed [B, 3, 240, 320] images; "
                f"got {found_shapes}"
            )

        num_images = len(image_keys)
        stacked_images = torch.stack([result[key] for key in image_keys], dim=1)
        flat_images = stacked_images.reshape(batch_size * num_images, *stacked_images.shape[2:])
        # Preserve the old model-side preprocessing's quantization point.
        flat_images_uint8 = (flat_images.float() * 255).to(torch.uint8)
        processed = image_processor.preprocess(flat_images_uint8, return_tensors="pt")
        pixel_values = processed["pixel_values"].contiguous()
        image_grid_thw = processed["image_grid_thw"].to("cpu").contiguous()

        if image_grid_thw.shape[0] != batch_size * num_images:
            raise RuntimeError(
                "Unexpected image_grid_thw rows: "
                f"actual={image_grid_thw.shape[0]} expected={batch_size * num_images}"
            )

        patch_counts = image_grid_thw.prod(dim=1).tolist()
        if pixel_values.shape[0] != sum(int(value) for value in patch_counts):
            raise RuntimeError(
                "Unexpected pixel_values rows: "
                f"actual={pixel_values.shape[0]} expected={sum(int(value) for value in patch_counts)}"
            )

        merge_size = int(image_processor.merge_size)
        input_ids_list: list[torch.Tensor] = []
        position_ids_list: list[torch.Tensor] = []
        for sample_index in range(batch_size):
            image_begin = sample_index * num_images
            sample_grid = image_grid_thw[image_begin:image_begin + num_images]
            sample_patch_counts = patch_counts[image_begin:image_begin + num_images]
            grid_thw_merged = [
                int(patch_count) // (merge_size**2)
                for patch_count in sample_patch_counts
            ]
            task = str(batch[sample_index]["task"])
            robot_type = str(batch[sample_index]["robot_type"])
            grid_key = tuple(tuple(int(value) for value in row) for row in sample_grid.tolist())
            cache_key = (task, robot_type, num_images, merge_size, grid_key)
            cached = self._text_pos_cache.get(cache_key)
            if cached is None:
                prompt = get_user_prompt(" ".join(["<image>"] * num_images), robot_type)
                input_ids = preprocess_qwen_visual(
                    [[{"from": "human", "value": prompt}, {"from": "gpt", "value": task}]],
                    tokenizer,
                    grid_thw_image=grid_thw_merged,
                )["input_ids"].cpu().contiguous()
                position_ids, _ = get_rope_index_3(
                    merge_size,
                    input_ids,
                    image_grid_thw=sample_grid,
                )
                cached = (input_ids, position_ids.cpu().contiguous())
                self._text_pos_cache[cache_key] = cached
            input_ids_list.append(cached[0].squeeze(0))
            position_ids_list.append(cached[1])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).contiguous()
        max_length = int(tokenizer.model_max_length)
        input_ids = input_ids[:, :max_length]
        position_ids = torch.cat(
            [
                F.pad(position_ids, (0, input_ids.shape[1] - position_ids.shape[2]), value=1)
                for position_ids in position_ids_list
            ],
            dim=1,
        ).contiguous()[:, :, :max_length]

        result[QWEN_PIXEL_VALUES] = pixel_values
        result[QWEN_IMAGE_GRID_THW] = image_grid_thw
        result[QWEN_INPUT_IDS] = input_ids
        result[QWEN_POSITION_IDS] = position_ids
        result[QWEN_ATTENTION_MASK] = input_ids.ne(tokenizer.pad_token_id)
        # The model consumes only the processed Qwen tensors.  Do not also
        # transfer the original camera tensors to MUSA.
        for key in image_keys:
            del result[key]
        return result


def install_qwen_dataloader_preprocess() -> None:
    """Install the tensorizing collate function before ``training.main``.

    ``train_fsdp2.py`` constructs its DataLoader with
    ``collate_fn=dataset.collate_fn`` and ``pin_memory=True``. Replacing this
    class-level static method here keeps that file untouched while ensuring
    workers emit only CPU tensors that the existing pin-memory thread can pin.
    """
    from .config import flag

    if not flag("SPIRIT_DATALOADER_QWEN_PREPROCESS", 1):
        return

    from dataset import RoboChallengeDataset

    if getattr(RoboChallengeDataset, "_spirit_qwen_tensor_collator_installed", False):
        return

    backbone = _backbone_from_training_arguments()
    RoboChallengeDataset.collate_fn = staticmethod(QwenTensorCollator(backbone))
    RoboChallengeDataset._spirit_qwen_tensor_collator_installed = True
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[QWEN_DATALOADER_PREPROCESS] "
            "enabled=True pin_memory=training_default "
            "collate=QwenTensorCollator "
            f"backbone={backbone}",
            flush=True,
        )
