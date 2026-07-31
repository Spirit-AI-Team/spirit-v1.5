"""Fixed-grid Vision attention, RoPE, layout, and flat-input optimizations."""
import os
from .common import profile_range as _profile_range
from ..config import choice, flag

def _vision_attention_layout() -> str:
    """0=varlen, 1=dense fixed-grid."""
    layout = ("varlen", "dense")[choice(
        "SPIRIT_QWEN_VISION_ATTN_LAYOUT", 1,
        ("varlen", "dense"),
        {"varlen": 0, "dense": 1},
    )]

    return layout


def _vision_rope_impl() -> str:
    """Resolve the Qwen Vision RoPE implementation."""
    return ("reference", "musa_fused")[choice(
        "SPIRIT_QWEN_VISION_ROPE_IMPL", 1,
        ("reference", "musa_fused"),
        {"reference": 0, "musa_fused": 1},
    )]


def _vision_qkv_layout() -> str:
    """0=views, 1=packed contiguous."""
    return ("views", "packed_contiguous")[choice(
        "SPIRIT_QWEN_VISION_QKV_LAYOUT", 0,
        ("views", "packed_contiguous"),
        {"views": 0, "packed_contiguous": 1},
    )]


def _install_fixed_grid_vision_attention() -> None:
    """Batch Qwen Vision attention when its varlen segments are equal length.

    Qwen's FA2 path flattens all image tokens and identifies independent images
    with ``cu_seqlens``. The fixed 240x320 benchmark layout makes those
    segments equal length, so dense batched SDPA is mathematically equivalent
    and avoids per-image varlen work. The first call validates the assumption;
    unsupported layouts fall back to the original Qwen implementation.
    """
    layout = _vision_attention_layout()
    rope_impl = _vision_rope_impl()
    qkv_layout = _vision_qkv_layout()
    if rope_impl == "musa_fused" and layout != "dense":
        raise ValueError(
            "SPIRIT_QWEN_VISION_ROPE_IMPL=musa_fused currently requires "
            "SPIRIT_QWEN_VISION_ATTN_LAYOUT=dense"
        )
    if layout == "varlen":
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[QWEN_VISION_ATTN] "
                "requested=varlen selected=varlen "
                "backend=qwen_flash_attention_2",
                flush=True,
            )
        return

    import torch
    import torch.nn.functional as functional
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionAttention,
        apply_rotary_pos_emb_vision,
    )
    if rope_impl == "musa_fused":
        from flash_attn.layers.rotary import apply_rotary_emb_qkv_

    if getattr(Qwen3VLVisionAttention, "_spirit_dense_attn_installed", False):
        return

    original_forward = Qwen3VLVisionAttention.forward
    # Static training shapes are validated once. The key is deliberately
    # independent of an individual Vision block so a 24-layer stack does not
    # introduce 24 device-to-host checks on its first step.
    validated_shapes: dict[tuple[str, int, int], bool] = {}
    logged_shapes: set[tuple[str, int, int]] = set()
    fused_rope_freqs: dict[tuple[str, int, int, str], tuple[object, object]] = {}
    fused_rope_validated: set[tuple[str, int, int, str]] = set()

    def reference_rope(qkv, cos, sin):
        query, key, value = qkv.unbind(dim=2)
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        return query, key, value

    def fused_rope_frequencies(qkv, cos, sin, head_dim):
        sequence_count, tokens_per_sequence = qkv.shape[:2]
        cos = cos.reshape(sequence_count, tokens_per_sequence, -1)
        sin = sin.reshape(sequence_count, tokens_per_sequence, -1)
        if cos.shape[-1] != head_dim:
            raise RuntimeError(
                "Unexpected Qwen Vision RoPE width: "
                f"cos={cos.shape[-1]} head_dim={head_dim}"
            )

        cache_key = (
            str(qkv.device),
            tokens_per_sequence,
            head_dim,
            str(qkv.dtype),
        )
        cached = fused_rope_freqs.get(cache_key)
        if cached is not None:
            return cache_key, cached

        # The fused FlashAttention rotary kernel takes one frequency table for
        # the whole dense batch. Verify once that every fixed-grid image uses
        # the same 2D positions before caching that table across train steps.
        shared_frequencies = bool(
            torch.equal(cos, cos[:1].expand_as(cos))
            and torch.equal(sin, sin[:1].expand_as(sin))
        )
        if not shared_frequencies:
            raise RuntimeError(
                "musa_fused Vision RoPE requires identical positional "
                "frequencies for every dense image sequence"
            )

        half_dim = head_dim // 2
        cos_half = cos[0, :, :half_dim].to(dtype=qkv.dtype).contiguous()
        sin_half = sin[0, :, :half_dim].to(dtype=qkv.dtype).contiguous()
        if len(fused_rope_freqs) >= 8:
            fused_rope_freqs.clear()
            fused_rope_validated.clear()
        fused_rope_freqs[cache_key] = (cos_half, sin_half)
        return cache_key, (cos_half, sin_half)

    def validate_fused_rope(qkv, cos, sin, cache_key, cos_half, sin_half):
        if cache_key in fused_rope_validated:
            return

        # Validate one complete image sequence to cover every spatial RoPE
        # position without duplicating the full training batch. This must stay
        # under no_grad: Vision blocks run inside non-reentrant activation
        # checkpointing, where a nested autograd.grad() triggers checkpoint's
        # internal _StopRecomputationError during backward recomputation.
        with torch.no_grad():
            reference_input = qkv[:1].detach().clone()
            fused_input = qkv[:1].detach().clone()
            reference_q, reference_k, _ = reference_rope(
                reference_input,
                cos[:1],
                sin[:1],
            )
            fused_output = apply_rotary_emb_qkv_(
                fused_input,
                cos_half,
                sin_half,
                interleaved=False,
            )
            fused_q, fused_k, _ = fused_output.unbind(dim=2)

            if not (
                torch.isfinite(fused_q).all()
                and torch.isfinite(fused_k).all()
            ):
                raise RuntimeError("musa_fused Vision RoPE produced non-finite output")

            forward_max_abs = max(
                float((reference_q - fused_q).abs().max().item()),
                float((reference_k - fused_k).abs().max().item()),
            )
            atol = float(os.environ.get("SPIRIT_QWEN_VISION_ROPE_ATOL", "0.02"))
            rtol = float(os.environ.get("SPIRIT_QWEN_VISION_ROPE_RTOL", "0.02"))
            q_close = bool(torch.isclose(fused_q, reference_q, atol=atol, rtol=rtol).all())
            k_close = bool(torch.isclose(fused_k, reference_k, atol=atol, rtol=rtol).all())
            if not (q_close and k_close):
                raise RuntimeError(
                    "musa_fused Vision RoPE forward differs from reference: "
                    f"max_abs={forward_max_abs:.8g} atol={atol} rtol={rtol}"
                )

        fused_rope_validated.add(cache_key)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[QWEN_VISION_ROPE_VALIDATION] "
                "impl=musa_fused status=PASS "
                f"forward_max_abs={forward_max_abs:.8g} "
                f"atol={atol} rtol={rtol} "
                "checkpoint_safe=True backward=flash_attn_conjugate_kernel",
                flush=True,
            )

    def dense_forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        del rotary_pos_emb
        sequence_count = cu_seqlens.numel() - 1
        total_tokens = hidden_states.shape[0]
        if sequence_count <= 0 or total_tokens % sequence_count:
            return original_forward(
                self,
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        tokens_per_sequence = total_tokens // sequence_count
        key = (str(hidden_states.device), sequence_count, tokens_per_sequence)
        is_fixed_grid = validated_shapes.get(key)
        if is_fixed_grid is None:
            expected = torch.arange(
                sequence_count + 1,
                device=cu_seqlens.device,
                dtype=cu_seqlens.dtype,
            ) * tokens_per_sequence
            is_fixed_grid = bool(torch.equal(cu_seqlens, expected))
            validated_shapes[key] = is_fixed_grid

        if not is_fixed_grid:
            if key not in logged_shapes and int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[QWEN_VISION_ATTN] "
                    "requested=dense selected=varlen "
                    "reason=nonuniform_cu_seqlens "
                    f"sequences={sequence_count} "
                    f"tokens_per_sequence={tokens_per_sequence}",
                    flush=True,
                )
                logged_shapes.add(key)
            return original_forward(
                self,
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        with _profile_range("SPIRIT::Vision::QKVProjection"):
            qkv = self.qkv(hidden_states).reshape(
                sequence_count,
                tokens_per_sequence,
                3,
                self.num_heads,
                self.head_dim,
            )
        cos, sin = position_embeddings
        cos = cos.reshape(sequence_count, tokens_per_sequence, -1)
        sin = sin.reshape(sequence_count, tokens_per_sequence, -1)
        if rope_impl == "musa_fused":
            with _profile_range("SPIRIT::Vision::RoPE"):
                rope_key, (cos_half, sin_half) = fused_rope_frequencies(
                    qkv, cos, sin, self.head_dim
                )
                validate_fused_rope(qkv, cos, sin, rope_key, cos_half, sin_half)
                qkv = apply_rotary_emb_qkv_(
                    qkv,
                    cos_half,
                    sin_half,
                    interleaved=False,
                )
                query_states, key_states, value_states = qkv.unbind(dim=2)
        else:
            with _profile_range("SPIRIT::Vision::RoPE"):
                query_states, key_states, value_states = reference_rope(qkv, cos, sin)

        with _profile_range("SPIRIT::Vision::QKVLayout"):
            if qkv_layout == "packed_contiguous":
                if rope_impl == "musa_fused":
                    # One combined materialization produces three contiguous
                    # [B,H,L,D] tensors instead of leaving SDPA to repair three
                    # strided views independently.
                    packed_qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
                else:
                    packed_qkv = torch.stack(
                        (
                            query_states.transpose(1, 2),
                            key_states.transpose(1, 2),
                            value_states.transpose(1, 2),
                        ),
                        dim=0,
                    )
                query_for_attn, key_for_attn, value_for_attn = packed_qkv.unbind(dim=0)
            else:
                query_for_attn = query_states.transpose(1, 2)
                key_for_attn = key_states.transpose(1, 2)
                value_for_attn = value_states.transpose(1, 2)

        with _profile_range("SPIRIT::Vision::Attention"):
            attn_output = functional.scaled_dot_product_attention(
                query_for_attn,
                key_for_attn,
                value_for_attn,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                scale=self.scaling,
            )
        with _profile_range("SPIRIT::Vision::AttentionOutput"):
            attn_output = attn_output.transpose(1, 2).reshape(total_tokens, -1)
            attn_output = self.proj(attn_output)

        if key not in logged_shapes and int(os.environ.get("RANK", "0")) == 0:
            print(
                "[QWEN_VISION_ATTN] "
                "requested=dense selected=dense "
                f"sequences={sequence_count} "
                f"tokens_per_sequence={tokens_per_sequence} "
                f"heads={self.num_heads} rope_impl={rope_impl} "
                f"qkv_layout={qkv_layout} "
                "backend=scaled_dot_product_attention",
                flush=True,
            )
            logged_shapes.add(key)
        return attn_output

    Qwen3VLVisionAttention.forward = dense_forward
    Qwen3VLVisionAttention._spirit_dense_attn_installed = True


def _install_flat_vision_inputs(spirit_model) -> None:
    """Reuse adjacent preprocess views without editing the vendored model."""
    if not flag("SPIRIT_QWEN_VISION_FLAT_INPUTS", 1):
        return

    import torch
    from ..qwen_dataloader_preprocess import (
        QWEN_ATTENTION_MASK,
        QWEN_IMAGE_GRID_THW,
        QWEN_INPUT_IDS,
        QWEN_PIXEL_VALUES,
        QWEN_POSITION_IDS,
    )

    policy_class = spirit_model.SpiritVLAPolicy
    if getattr(policy_class, "_spirit_flat_inputs_installed", False):
        return

    def coalesce_adjacent_views(views):
        if not views:
            return None
        first = views[0]
        if first.ndim != 2 or not first.is_contiguous():
            return None
        storage_pointer = first.untyped_storage().data_ptr()
        expected_offset = first.storage_offset()
        total_rows = 0
        for view in views:
            if (
                view.ndim != 2
                or view.shape[1] != first.shape[1]
                or view.stride() != first.stride()
                or view.untyped_storage().data_ptr() != storage_pointer
                or view.storage_offset() != expected_offset
            ):
                return None
            total_rows += view.shape[0]
            expected_offset += view.shape[0] * view.stride(0)
        return first.as_strided(
            (total_rows, first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )

    def encode_vision(self, batch):
        device = batch[spirit_model.OBS_ROBOT].device
        prepared_keys = (
            QWEN_PIXEL_VALUES,
            QWEN_IMAGE_GRID_THW,
            QWEN_INPUT_IDS,
            QWEN_POSITION_IDS,
            QWEN_ATTENTION_MASK,
        )
        prepared = all(key in batch for key in prepared_keys)
        if prepared:
            pixel_values = batch[QWEN_PIXEL_VALUES]
            image_grid_thw = batch[QWEN_IMAGE_GRID_THW]
            input_ids = batch[QWEN_INPUT_IDS]
            position_ids = batch[QWEN_POSITION_IDS]
            attention_mask = batch[QWEN_ATTENTION_MASK]
            selected = True
        else:
            (
                pixel_values_list,
                image_grid_thw_list,
                input_ids_list,
                position_ids_list,
            ) = self.preprocess_rb_batch(batch)

            pixel_values = coalesce_adjacent_views(pixel_values_list)
            image_grid_thw = coalesce_adjacent_views(image_grid_thw_list)
            selected = pixel_values is not None and image_grid_thw is not None
            if not selected:
                pixel_values = torch.cat(pixel_values_list, dim=0)
                image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
            pixel_values = pixel_values.to(device).contiguous()
            image_grid_thw = image_grid_thw.to(device).contiguous()

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.language_tokenizer.pad_token_id,
            ).to(device).contiguous()
            model_max_length = self.language_tokenizer.model_max_length
            input_ids = input_ids[:, :model_max_length]
            position_ids = spirit_model.pad_and_cat(position_ids_list)[
                :, :, :model_max_length
            ].to(device).contiguous()
            attention_mask = input_ids.ne(
                self.language_tokenizer.pad_token_id
            ).to(device)

        if not getattr(self, "_vision_flat_inputs_logged", False):
            print(
                "[QWEN_VISION_FLAT_INPUTS] "
                f"rank={os.environ.get('RANK', '0')} requested=1 "
                f"selected={selected} "
                f"dataloader_preprocessed={prepared} "
                f"pixel_values_contiguous={pixel_values.is_contiguous()} "
                f"pixel_values_shape={tuple(pixel_values.shape)}",
                flush=True,
            )
            self._vision_flat_inputs_logged = True

        if not flag("SPIRIT_QWEN_SKIP_LM_HEAD", 1):
            raise RuntimeError(
                "SPIRIT_QWEN_SKIP_LM_HEAD must be 1 for this experiment"
            )
        vlm_outputs = self.qwen.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        if vlm_outputs.hidden_states is None:
            raise RuntimeError("Qwen base model did not return hidden_states")
        num_vlm_last_embd = min(
            self.num_vlm_last_embd,
            len(vlm_outputs.hidden_states),
        )
        vlm_last_embed = torch.cat(
            vlm_outputs.hidden_states[-num_vlm_last_embd:],
            dim=1,
        )
        if not getattr(self, "_qwen_skip_lm_head_validated", False):
            print(
                "[QWEN_SKIP_LM_HEAD] "
                f"rank={os.environ.get('RANK', '0')} validation=PASS "
                "logits_computed=False "
                f"hidden_states_count={len(vlm_outputs.hidden_states)} "
                f"selected_count={num_vlm_last_embd} "
                f"output_shape={tuple(vlm_last_embed.shape)} "
                f"dtype={vlm_last_embed.dtype} "
                f"device={vlm_last_embed.device}",
                flush=True,
            )
            self._qwen_skip_lm_head_validated = True
        return vlm_last_embed

    policy_class._encode_vision = encode_vision
    policy_class._spirit_flat_inputs_installed = True
