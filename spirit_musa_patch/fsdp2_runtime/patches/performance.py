"""Retained throughput optimizations for Qwen RMSNorm/RoPE/SwiGLU."""
import os
from .common import profile_range as _profile_range
from ..config import choice


def install_legacy_model_config_compat() -> None:
    """Install source-version performance parity before model construction.

    The optimized same-lineage model consumes ``SPIRIT_PATCH_EMBED_IMPL`` and
    loads the Qwen backbone directly in BF16.  The current official model does
    neither: it passes ``dtype=torch.float32`` explicitly and always executes
    the Conv3d patch embed.  Reproduce those two behaviors here so the stock
    model file remains untouched.
    """
    value = os.environ.get("SPIRIT_PATCH_EMBED_IMPL")
    if value in {"0", "1"}:
        implementation = ("conv3d", "gemm")[int(value)]
        os.environ["SPIRIT_PATCH_EMBED_IMPL"] = implementation
    else:
        implementation = value or "conv3d"
    if implementation not in {"conv3d", "gemm"}:
        raise ValueError(
            "SPIRIT_PATCH_EMBED_IMPL must be 0/1 or conv3d/gemm; "
            f"got {implementation!r}"
        )

    import torch
    import torch.nn.functional as functional
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLVisionPatchEmbed,
    )

    if implementation == "gemm" and not getattr(
        Qwen3VLVisionPatchEmbed,
        "_spirit_gemm_patch_installed",
        False,
    ):
        def gemm_forward(self, hidden_states):
            target_dtype = self.proj.weight.dtype
            hidden_states = hidden_states.reshape(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ).to(dtype=target_dtype)
            return functional.linear(
                hidden_states.flatten(1),
                self.proj.weight.flatten(1),
                self.proj.bias,
            )

        Qwen3VLVisionPatchEmbed.forward = gemm_forward
        Qwen3VLVisionPatchEmbed._spirit_gemm_patch_installed = True

    if not getattr(
        Qwen3VLForConditionalGeneration,
        "_spirit_bf16_load_patch_installed",
        False,
    ):
        original_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained

        def bf16_from_pretrained(cls, *args, **kwargs):
            stock_requested_dtype = kwargs.get("dtype")
            kwargs["dtype"] = torch.bfloat16
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[QWEN_BACKBONE_LOAD_DTYPE] "
                    f"stock_requested={stock_requested_dtype} "
                    "selected=torch.bfloat16 source_parity=True",
                    flush=True,
                )
            return original_from_pretrained(*args, **kwargs)

        Qwen3VLForConditionalGeneration.from_pretrained = classmethod(
            bf16_from_pretrained
        )
        Qwen3VLForConditionalGeneration._spirit_bf16_load_patch_installed = True

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[SPIRIT_PATCH_EMBED_IMPL] "
            f"runtime_value={value} selected={implementation} "
            f"class_patch={implementation == 'gemm'}",
            flush=True,
        )

def _qwen_rms_norm_impl() -> str:
    """Return the opt-in Qwen RMSNorm implementation.

    The reference path remains the default because the MUSA fused operator is
    version-dependent and changes the accumulation implementation.  Keeping
    the switch here also makes the operator A/B reversible without touching the
    transformers installation.
    """
    return ("reference", "musa_fused")[choice(
        "SPIRIT_QWEN_RMS_NORM_IMPL", 1,
        ("reference", "musa_fused"),
        {"reference": 0, "musa_fused": 1},
    )]


def _install_qwen_fused_rms_norm() -> None:
    """Opt in to the backend RMSNorm while preserving a reference fallback.

    Qwen's stock implementation expands RMSNorm into cast/pow/mean/rsqrt/mul.
    torch-musa provides a fused RMSNorm dispatcher.  We only select it when
    input and weight dtypes match; this preserves the stock behavior for the
    mixed-dtype case, where the fused implementation may promote differently.
    """
    implementation = _qwen_rms_norm_impl()
    if implementation == "reference":
        return

    import torch

    required_ops = (
        "aten::_fused_rms_norm",
        "aten::_fused_rms_norm_backward",
    )
    missing_ops = [
        operator
        for operator in required_ops
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            operator, "PrivateUse1"
        )
    ]
    if missing_ops:
        raise RuntimeError(
            "SPIRIT_QWEN_RMS_NORM_IMPL=musa_fused requires torch-musa "
            f"PrivateUse1 kernels; missing={missing_ops}"
        )

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRMSNorm,
        )
    except Exception as error:
        raise RuntimeError(
            "SPIRIT_QWEN_RMS_NORM_IMPL=musa_fused requires "
            "transformers.models.qwen3_vl"
        ) from error

    if getattr(Qwen3VLTextRMSNorm, "_spirit_rms_norm_patched", False):
        return

    original_forward = Qwen3VLTextRMSNorm.forward

    def fused_forward(self, hidden_states):
        # Do not silently alter the reference result on unsupported devices,
        # mixed dtypes, or unexpected normalized shapes.  Those cases are
        # intentionally kept on the original implementation.
        device_type = hidden_states.device.type
        is_musa = device_type in {"musa", "privateuseone"}
        dtypes_match = hidden_states.dtype == self.weight.dtype
        shape_matches = (
            hidden_states.ndim > 0
            and hidden_states.shape[-1] == self.weight.shape[0]
        )
        if not is_musa or not dtypes_match or not shape_matches:
            if (
                is_musa
                and not getattr(
                    Qwen3VLTextRMSNorm,
                    "_spirit_rms_norm_fallback_logged",
                    False,
                )
                and int(os.environ.get("RANK", "0")) == 0
            ):
                print(
                    "[QWEN_RMS_NORM] implementation=reference_fallback "
                    f"input_dtype={hidden_states.dtype} "
                    f"weight_dtype={self.weight.dtype} "
                    f"input_last_dim="
                    f"{hidden_states.shape[-1] if hidden_states.ndim else 'none'} "
                    f"weight_dim={self.weight.shape[0]}",
                    flush=True,
                )
                Qwen3VLTextRMSNorm._spirit_rms_norm_fallback_logged = True
            return original_forward(self, hidden_states)

        import torch.nn.functional as functional

        if (
            not getattr(
                Qwen3VLTextRMSNorm, "_spirit_rms_norm_active_logged", False
            )
            and int(os.environ.get("RANK", "0")) == 0
        ):
            print(
                "[QWEN_RMS_NORM] active=musa_fused "
                f"dtype={hidden_states.dtype} "
                f"normalized_shape=({self.weight.shape[0]},)",
                flush=True,
            )
            Qwen3VLTextRMSNorm._spirit_rms_norm_active_logged = True

        return functional.rms_norm(
            hidden_states,
            (self.weight.shape[0],),
            self.weight,
            self.variance_epsilon,
        )

    Qwen3VLTextRMSNorm.forward = fused_forward
    Qwen3VLTextRMSNorm._spirit_rms_norm_patched = True

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[QWEN_RMS_NORM] implementation=musa_fused "
            "mixed_dtype=reference_fallback shape_mismatch=reference_fallback",
            flush=True,
        )


def _text_rope_impl() -> str:
    """Resolve the Qwen Text MRoPE implementation."""
    return ("reference", "musa_fused")[choice(
        "SPIRIT_QWEN_TEXT_ROPE_IMPL", 1,
        ("reference", "musa_fused"),
        {"reference": 0, "musa_fused": 1},
    )]


def _text_swiglu_impl() -> str:
    """Resolve the Qwen Text MLP SwiGLU implementation."""
    return ("reference", "musa_fused")[choice(
        "SPIRIT_QWEN_TEXT_SWIGLU_IMPL", 1,
        ("reference", "musa_fused"),
        {"reference": 0, "musa_fused": 1},
    )]


def pack_qwen_text_mlps(model) -> int:
    """Pack gate/up weights after checkpoint load and before FSDP/optimizer.

    Checkpoints are intentionally loaded into the stock Qwen modules first.
    This conversion then replaces two equally shaped parameters with one
    concatenated parameter and uses the official muDNN fused SwiGLU operator.
    The total parameter count and elementwise optimizer math are unchanged.
    Checkpoint export converts the packed parameter back to the original
    ``gate_proj.weight`` and ``up_proj.weight`` names.
    """
    implementation = _text_swiglu_impl()
    if implementation == "reference":
        return 0

    import types

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    converted = 0

    def packed_forward(self, hidden_states):
        with _profile_range("SPIRIT::TextMLP::GateUpProjection"):
            packed = self.gate_up_proj(hidden_states)
        with _profile_range("SPIRIT::TextMLP::SwiGLU"):
            if packed.device.type in {"musa", "privateuseone"}:
                activated = torch.ops.aten._fused_swiglu_forward.default(
                    packed
                )
            else:
                gate, up = packed.chunk(2, dim=-1)
                activated = F.silu(gate) * up
        with _profile_range("SPIRIT::TextMLP::DownProjection"):
            return self.down_proj(activated)

    for module in model.modules():
        if type(module).__name__ != "Qwen3VLTextMLP":
            continue
        if hasattr(module, "gate_up_proj"):
            continue
        if not hasattr(module, "gate_proj") or not hasattr(module, "up_proj"):
            raise RuntimeError(
                "Unexpected Qwen3VLTextMLP layout: gate_proj/up_proj missing"
            )
        gate_projection = module.gate_proj
        up_projection = module.up_proj
        if gate_projection.bias is not None or up_projection.bias is not None:
            raise RuntimeError("Packed Text SwiGLU requires bias-free gate/up")
        gate_weight = gate_projection.weight
        up_weight = up_projection.weight
        if (
            gate_weight.shape != up_weight.shape
            or gate_weight.dtype != up_weight.dtype
            or gate_weight.device != up_weight.device
            or gate_weight.requires_grad != up_weight.requires_grad
        ):
            raise RuntimeError(
                "Packed Text SwiGLU requires matching gate/up parameters"
            )

        packed_weight = torch.cat(
            (gate_weight.detach(), up_weight.detach()), dim=0
        )
        packed_projection = nn.Linear(
            gate_weight.shape[1],
            gate_weight.shape[0] * 2,
            bias=False,
            device="meta",
            dtype=gate_weight.dtype,
        )
        packed_projection.weight = nn.Parameter(
            packed_weight, requires_grad=gate_weight.requires_grad
        )
        del module.gate_proj
        del module.up_proj
        module.gate_up_proj = packed_projection
        module.forward = types.MethodType(packed_forward, module)
        module._spirit_text_swiglu_packed = True
        converted += 1

    if converted == 0:
        raise RuntimeError("No Qwen3VLTextMLP modules found for packing")
    if int(os.environ.get("RANK", "0")) == 0:
        first = next(
            module
            for module in model.modules()
            if getattr(module, "_spirit_text_swiglu_packed", False)
        )
        print(
            "[QWEN_TEXT_SWIGLU] implementation=musa_fused "
            f"converted_mlps={converted} "
            f"packed_weight_shape={tuple(first.gate_up_proj.weight.shape)} "
            "checkpoint_export=reference_gate_up_names "
            "backward=aten_fused_swiglu_backward",
            flush=True,
        )
    return converted


def _install_text_fused_rope() -> None:
    """Use the FlashAttention rotary kernel for Qwen Text MRoPE.

    Qwen supplies batch-specific frequencies as ``[B, S, D]`` with the
    half-width frequency table duplicated across the last dimension.  The
    installed rotary kernel accepts a two-dimensional table.  Flattening both
    tokens and their frequencies to ``[B*S, ...]`` is exact because rotary
    embedding is pointwise in the token dimension, and unlike selecting the
    first batch item it remains correct when samples have different MRoPE
    positions.
    """
    implementation = _text_rope_impl()
    if implementation == "reference":
        return

    import torch
    from flash_attn.layers.rotary import apply_rotary_emb
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen_modeling

    if getattr(qwen_modeling, "_spirit_text_rope_fused_installed", False):
        return

    original_rope = qwen_modeling.apply_rotary_pos_emb
    # Hold strong references to exactly one model-forward frequency set. This
    # lets all 36 layers and checkpoint recomputation reuse the materialized
    # half tables, prevents allocator pointer reuse from producing a false
    # cache hit, and avoids accumulating tables across training steps.
    frequency_cache: tuple[object, object, object, object] | None = None

    def fused_rope(
        query,
        key,
        cos,
        sin,
        position_ids=None,
        unsqueeze_dim=1,
    ):
        nonlocal frequency_cache
        del position_ids
        is_musa = query.device.type in {"musa", "privateuseone"}
        if not is_musa:
            return original_rope(
                query,
                key,
                cos,
                sin,
                unsqueeze_dim=unsqueeze_dim,
            )
        if unsqueeze_dim != 1:
            raise RuntimeError(
                "musa_fused Text MRoPE requires unsqueeze_dim=1; "
                f"got {unsqueeze_dim}"
            )
        if query.ndim != 4 or key.ndim != 4:
            raise RuntimeError(
                "musa_fused Text MRoPE requires Q/K shaped [B,H,S,D]; "
                f"query={tuple(query.shape)} key={tuple(key.shape)}"
            )
        batch, query_heads, sequence_length, head_dim = query.shape
        if (
            key.shape[0] != batch
            or key.shape[2] != sequence_length
            or key.shape[3] != head_dim
        ):
            raise RuntimeError(
                "musa_fused Text MRoPE received incompatible Q/K shapes: "
                f"query={tuple(query.shape)} key={tuple(key.shape)}"
            )
        if head_dim % 2 or head_dim > 256:
            raise RuntimeError(
                "musa_fused Text MRoPE requires an even head_dim <= 256; "
                f"got {head_dim}"
            )
        expected_frequency_shape = (batch, sequence_length, head_dim)
        if tuple(cos.shape) != expected_frequency_shape or tuple(
            sin.shape
        ) != expected_frequency_shape:
            raise RuntimeError(
                "musa_fused Text MRoPE received unexpected frequencies: "
                f"cos={tuple(cos.shape)} sin={tuple(sin.shape)} "
                f"expected={expected_frequency_shape}"
            )
        if not (
            query.dtype == key.dtype == cos.dtype == sin.dtype
            and query.device == key.device == cos.device == sin.device
        ):
            raise RuntimeError(
                "musa_fused Text MRoPE requires matching Q/K/frequency "
                "dtype and device"
            )

        half_dim = head_dim // 2
        cache_matches = (
            frequency_cache is not None
            and frequency_cache[0] is cos
            and frequency_cache[1] is sin
        )
        if not cache_matches:
            # Qwen constructs [freqs, freqs] along the head dimension.  Keep
            # only the first half and materialize it once for all 36 layers.
            cos_half = cos[..., :half_dim].contiguous().view(
                batch * sequence_length, half_dim
            )
            sin_half = sin[..., :half_dim].contiguous().view(
                batch * sequence_length, half_dim
            )
            frequency_cache = (cos, sin, cos_half, sin_half)
        else:
            cos_half = frequency_cache[2]
            sin_half = frequency_cache[3]

        query_bshd = query.transpose(1, 2).reshape(
            1, batch * sequence_length, query_heads, head_dim
        )
        key_heads = key.shape[1]
        key_bshd = key.transpose(1, 2).reshape(
            1, batch * sequence_length, key_heads, head_dim
        )
        query_rotated = apply_rotary_emb(
            query_bshd,
            cos_half,
            sin_half,
            interleaved=False,
            inplace=False,
        )
        key_rotated = apply_rotary_emb(
            key_bshd,
            cos_half,
            sin_half,
            interleaved=False,
            inplace=False,
        )
        query_rotated = query_rotated.view(
            batch, sequence_length, query_heads, head_dim
        ).transpose(1, 2)
        key_rotated = key_rotated.view(
            batch, sequence_length, key_heads, head_dim
        ).transpose(1, 2)

        if (
            not getattr(
                qwen_modeling, "_spirit_text_rope_fused_logged", False
            )
            and int(os.environ.get("RANK", "0")) == 0
        ):
            print(
                "[QWEN_TEXT_ROPE] implementation=musa_fused "
                f"query_shape={tuple(query.shape)} "
                f"key_shape={tuple(key.shape)} "
                f"frequency_shape={tuple(cos_half.shape)} "
                "batch_specific_frequencies=flattened "
                "backward=flash_attn_conjugate_kernel",
                flush=True,
            )
            qwen_modeling._spirit_text_rope_fused_logged = True
        return query_rotated, key_rotated

    qwen_modeling.apply_rotary_pos_emb = fused_rope
    qwen_modeling._spirit_text_rope_fused_installed = True
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[QWEN_TEXT_ROPE] configured=musa_fused "
            "unsupported_device=reference_fallback",
            flush=True,
        )
