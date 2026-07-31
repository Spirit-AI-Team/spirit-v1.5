"""Attribution-only profiler ranges; no training semantics are changed."""
import os
from .common import profile_range as _profile_range
from ..config import flag

def _install_attention_profile_ranges() -> None:
    """Annotate Text attention; Vision/DiT ranges are added at their patches.

    Keep this entirely profile-only.  The stock Qwen forward remains the
    implementation, while its child modules and selected runtime attention
    interface receive nested ranges.  This avoids copying version-sensitive
    model logic merely for attribution.
    """
    if not flag("SPIRIT_PROFILE_RANGES", 0):
        return
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextAttention,
        )
    except Exception as error:
        raise RuntimeError(
            "SPIRIT_PROFILE_RANGES=1 requires Qwen3VLTextAttention"
        ) from error
    if getattr(Qwen3VLTextAttention, "_spirit_profile_ranges_installed", False):
        return
    original_forward = Qwen3VLTextAttention.forward

    def wrap_instance_forward(module, range_name):
        if getattr(module, "_spirit_profile_range_installed", False):
            return
        child_forward = module.forward

        def ranged_child_forward(*args, **kwargs):
            with _profile_range(range_name):
                return child_forward(*args, **kwargs)

        module.forward = ranged_child_forward
        module._spirit_profile_range_installed = True

    def install_instance_ranges(module):
        for projection in (module.q_proj, module.k_proj, module.v_proj):
            wrap_instance_forward(
                projection, "SPIRIT::TextAttention::QKVProjection"
            )
        for norm in (module.q_norm, module.k_norm):
            wrap_instance_forward(norm, "SPIRIT::TextAttention::QKNorm")
        wrap_instance_forward(
            module.o_proj, "SPIRIT::TextAttention::OutputProjection"
        )

    def profiled_forward(self, *args, **kwargs):
        install_instance_ranges(self)
        with _profile_range("SPIRIT::TextAttention"):
            return original_forward(self, *args, **kwargs)

    Qwen3VLTextAttention.forward = profiled_forward
    Qwen3VLTextAttention._spirit_profile_ranges_installed = True

    # RoPE is a free function in the Qwen module rather than a child module.
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen_modeling

    if not getattr(qwen_modeling, "_spirit_text_rope_range_installed", False):
        original_rope = qwen_modeling.apply_rotary_pos_emb

        def profiled_rope(*args, **kwargs):
            with _profile_range("SPIRIT::TextAttention::RoPE"):
                return original_rope(*args, **kwargs)

        qwen_modeling.apply_rotary_pos_emb = profiled_rope
        qwen_modeling._spirit_text_rope_range_installed = True

    # Wrap whichever Transformers attention interface the loaded checkpoint
    # actually selects.  In this environment ``sdpa`` reaches a MUSA causal
    # flash-attention kernel; forcing the unrelated ``flash_attention_2``
    # integration would be both incorrect and version-dependent.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for implementation in tuple(ALL_ATTENTION_FUNCTIONS.keys()):
        original_interface = ALL_ATTENTION_FUNCTIONS[implementation]

        def profiled_interface(
            module,
            *args,
            _implementation=implementation,
            _original_interface=original_interface,
            **kwargs,
        ):
            if not isinstance(module, Qwen3VLTextAttention):
                return _original_interface(module, *args, **kwargs)
            if (
                not getattr(
                    Qwen3VLTextAttention,
                    "_spirit_attention_backend_logged",
                    False,
                )
                and int(os.environ.get("RANK", "0")) == 0
            ):
                query = args[0] if args else kwargs.get("query")
                key = args[1] if len(args) > 1 else kwargs.get("key")
                value = args[2] if len(args) > 2 else kwargs.get("value")
                attention_mask = (
                    args[3]
                    if len(args) > 3
                    else kwargs.get("attention_mask")
                )
                print(
                    "[QWEN_TEXT_ATTN_PROFILE] "
                    f"implementation={_implementation} "
                    f"query_shape={tuple(query.shape)} "
                    f"key_shape={tuple(key.shape)} "
                    f"value_shape={tuple(value.shape)} "
                    f"mask_shape="
                    f"{None if attention_mask is None else tuple(attention_mask.shape)} "
                    f"mask_dtype="
                    f"{None if attention_mask is None else attention_mask.dtype}",
                    flush=True,
                )
                Qwen3VLTextAttention._spirit_attention_backend_logged = True

            # For SDPA, isolate the actual dispatcher call without changing
            # its arguments or backend selection.  The surrounding interface
            # range then contains GQA expansion, mask slicing, transpose, and
            # the final contiguous materialization.
            if _implementation == "sdpa":
                import torch.nn.functional as functional

                original_sdpa = functional.scaled_dot_product_attention

                def profiled_sdpa(*sdpa_args, **sdpa_kwargs):
                    with _profile_range(
                        "SPIRIT::TextAttention::CausalSDPA"
                    ):
                        return original_sdpa(*sdpa_args, **sdpa_kwargs)

                functional.scaled_dot_product_attention = profiled_sdpa
                try:
                    with _profile_range(
                        "SPIRIT::TextAttention::AttentionInterface"
                    ):
                        return _original_interface(module, *args, **kwargs)
                finally:
                    functional.scaled_dot_product_attention = original_sdpa

            with _profile_range(
                "SPIRIT::TextAttention::AttentionInterface"
            ):
                return _original_interface(module, *args, **kwargs)

        ALL_ATTENTION_FUNCTIONS[implementation] = profiled_interface


