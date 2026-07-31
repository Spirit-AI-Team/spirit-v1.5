"""One-shot runtime model-FLOPs estimator using real tensor shapes."""

from __future__ import annotations

from collections import defaultdict

import torch


class RuntimeModelFlopEstimator:
    """Measure one forward and estimate useful training FLOPs.

    Linear/Conv FLOPs use exact runtime tensor shapes. Attention QK/AV FLOPs
    are added analytically for Qwen Text, Qwen Vision, and Diffusers Attention.
    Hooks are removed after one forward, so steady-state timing is unaffected.
    """

    def __init__(self, model, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.forward_flops = defaultdict(float)
        self.handles = []

    @staticmethod
    def _component(module) -> str:
        name = type(module).__name__
        if "Vision" in name:
            return "vision"
        if "Qwen3VLText" in name:
            return "text"
        if name == "Attention":
            return "dit_attention"
        return "matrix"

    def _linear_hook(self, module, inputs, output):
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        value = inputs[0]
        # Each output element performs in_features multiply-adds (2 FLOPs).
        flops = 2 * value.numel() * module.out_features
        self.forward_flops["linear"] += float(flops)

    def _conv_hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            return
        kernel_ops = module.weight[0].numel()
        flops = 2 * output.numel() * kernel_ops
        self.forward_flops["conv"] += float(flops)

    def _attention_hook(self, module, args, kwargs, output):
        if not args or not torch.is_tensor(args[0]):
            return
        hidden = args[0]
        name = type(module).__name__

        if name == "Qwen3VLTextAttention":
            batch, query_len, hidden_size = hidden.shape
            # Decoder causal attention computes approximately half of the
            # full QK and AV matrices: 2 * B * L^2 * D forward FLOPs.
            flops = 2 * batch * query_len * query_len * hidden_size
            self.forward_flops["text_attention_qkav"] += float(flops)
            return

        if name == "Qwen3VLVisionAttention":
            cu_seqlens = kwargs.get("cu_seqlens")
            if cu_seqlens is None and len(args) > 1:
                cu_seqlens = args[1]
            if cu_seqlens is None:
                return
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            sum_length_squared = int((lengths * lengths).sum().item())
            hidden_size = hidden.shape[-1]
            # Bidirectional QK plus AV: 4 * sum(segment_len^2) * D.
            flops = 4 * sum_length_squared * hidden_size
            self.forward_flops["vision_attention_qkav"] += float(flops)
            return

        if name == "Attention" and hidden.ndim == 3:
            encoder = kwargs.get("encoder_hidden_states")
            if encoder is None and len(args) > 1:
                encoder = args[1]
            key_value = encoder if torch.is_tensor(encoder) else hidden
            batch, query_len = hidden.shape[:2]
            key_len = key_value.shape[1]
            inner_dim = getattr(module, "inner_dim", hidden.shape[-1])
            flops = 4 * batch * query_len * key_len * inner_dim
            self.forward_flops["dit_attention_qkav"] += float(flops)

    def start(self) -> None:
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self.handles.append(
                    module.register_forward_hook(self._linear_hook)
                )
            elif isinstance(
                module,
                (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
            ):
                self.handles.append(
                    module.register_forward_hook(self._conv_hook)
                )

            if type(module).__name__ in {
                "Qwen3VLTextAttention",
                "Qwen3VLVisionAttention",
                "Attention",
            }:
                self.handles.append(
                    module.register_forward_hook(
                        self._attention_hook,
                        with_kwargs=True,
                    )
                )

    def finish(self) -> dict[str, float]:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        total_forward = sum(self.forward_flops.values())
        if total_forward <= 0:
            raise RuntimeError("Runtime FLOPs estimator captured no operations")
        return {
            "forward_flops_per_rank": total_forward,
            "forward_flops_per_sample": total_forward / self.batch_size,
            **dict(self.forward_flops),
        }

