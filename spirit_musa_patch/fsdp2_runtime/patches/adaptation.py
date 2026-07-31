"""Compatibility patches for FSDP2 BF16 DiT execution."""
import os
from .common import profile_range as _profile_range

def install_dtype_adaptation():
    """Patch only this process; keep the checked-in model implementation intact."""
    import model.modeling_spirit_vla as spirit_model

    if getattr(spirit_model, "_fsdp2_dtype_patch_applied", False):
        return

    original_embed_suffix = spirit_model.SpiritVLAPolicy._embed_suffix
    original_compute_loss = spirit_model.SpiritVLAPolicy.compute_loss

    def embed_suffix(self, state, noisy_actions, mask_state=True):
        # The current official model keeps normalized inputs in FP32, while
        # FSDP2 mixed precision unshards these projections in BF16.  Keep this
        # source-version compatibility at the opt-in runtime boundary.
        state = state.to(dtype=self.state_proj.weight.dtype)
        noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        return original_embed_suffix(
            self,
            state,
            noisy_actions,
            mask_state=mask_state,
        )

    def compute_loss(self, batch):
        # action_out_proj has the same official-source dtype mismatch.  A
        # temporary pre-hook keeps the original loss implementation and its
        # masks/logging unchanged, while the output hook restores FP32 loss
        # arithmetic just like the newer same-lineage implementation.
        def cast_input(module, args):
            return (args[0].to(dtype=module.weight.dtype),)

        def cast_output(_module, _args, output):
            return output.float()

        pre_handle = self.action_out_proj.register_forward_pre_hook(cast_input)
        post_handle = self.action_out_proj.register_forward_hook(cast_output)
        try:
            return original_compute_loss(self, batch)
        finally:
            pre_handle.remove()
            post_handle.remove()

    def block_forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        compute_dtype = self.attn1.to_q.weight.dtype
        norm_hidden_states = norm_hidden_states.to(dtype=compute_dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=compute_dtype)

        with _profile_range("SPIRIT::DiT::Attention"):
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        attn_output = self.attn_final_dropout(attn_output)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = self.norm3(hidden_states).to(dtype=compute_dtype)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

    def dit_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
    ):
        temb = self.timestep_encoder(timestep)
        # In block-granularity FSDP2, a child block's registered Parameter is
        # still the original FP32 sharded DTensor until that block's pre-hook
        # runs. proj_out_1 belongs to the already-unsharded root group here, so
        # its dtype is the actual mixed-precision compute dtype (BF16).
        compute_dtype = self.proj_out_1.weight.dtype
        hidden_states = hidden_states.to(dtype=compute_dtype).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(dtype=compute_dtype).contiguous()
        temb = temb.to(dtype=compute_dtype)

        all_hidden_states = [hidden_states] if return_all_hidden_states else None
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(hidden_states, temb=temb)
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        shift, scale = self.proj_out_1(functional.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states

    # Keep the original F.silu lookup without importing or changing the model.
    import torch.nn.functional as functional

    spirit_model.BasicTransformerBlock.forward = block_forward
    spirit_model.BaseDiT.forward = dit_forward
    spirit_model.SpiritVLAPolicy._embed_suffix = embed_suffix
    spirit_model.SpiritVLAPolicy.compute_loss = compute_loss
    spirit_model._fsdp2_dtype_patch_applied = True
    print("[FSDP2_MODEL_PATCH] dtype_casts=enabled", flush=True)
