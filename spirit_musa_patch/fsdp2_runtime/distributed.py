"""Composable FSDP2 setup shared by single-node FSDP and multi-node HSDP."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from .config import choice
from .explicit_prefetch import install_explicit_prefetch


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _fsdp_wrap_granularity() -> str:
    """Decode SPIRIT_FSDP2_WRAP_LEVEL: 0=coarse, 1=block."""
    return ("coarse", "block")[choice(
        "SPIRIT_FSDP2_WRAP_LEVEL", 1,
        ("coarse", "block"),
        {"coarse": 0, "block": 1},
    )]


def setup_distributed():
    """Initialise either 1D FSDP or 2D HSDP from the torchrun environment.

    ``SPIRIT_FSDP2_MODE=0`` is auto: one node uses 1D FSDP and multiple nodes
    use 2D HSDP.  Values 1 and 2 force FSDP and HSDP respectively.
    """
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        print("[FSDP2_TOPOLOGY] mode=single-process", flush=True)
        return 0, 0, 1, None

    dist.init_process_group(backend="nccl")
    local_rank = _env_int("LOCAL_RANK", 0)
    global_rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", world_size)
    mode = ("auto", "fsdp", "hsdp")[choice(
        "SPIRIT_FSDP2_MODE", 0,
        ("auto", "fsdp", "hsdp"),
        {"auto": 0, "fsdp": 1, "hsdp": 2},
    )]
    if local_world_size < 1 or world_size < 1:
        raise RuntimeError(
            f"Invalid process sizes: world_size={world_size}, "
            f"local_world_size={local_world_size}"
        )

    torch.cuda.set_device(local_rank)
    use_hsdp = mode == "hsdp" or (mode == "auto" and world_size > local_world_size)

    if not use_hsdp:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
        print(
            "[FSDP2_TOPOLOGY] "
            f"mode=fsdp rank={global_rank}/{world_size} "
            f"mesh=({world_size},)",
            flush=True,
        )
        return local_rank, global_rank, world_size, mesh

    shard_size = _env_int("SPIRIT_FSDP2_SHARD_SIZE", local_world_size)
    if shard_size < 2 or world_size % shard_size != 0:
        raise RuntimeError(
            "HSDP requires WORLD_SIZE to be divisible by a shard size >= 2; "
            f"got world_size={world_size}, shard_size={shard_size}"
        )
    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    print(
        "[FSDP2_TOPOLOGY] "
        f"mode=hsdp rank={global_rank}/{world_size} "
        f"mesh=({replicate_size},{shard_size})",
        flush=True,
    )
    return local_rank, global_rank, world_size, mesh


def _find_exactly_one(model, class_name):
    matches = [module for module in model.modules() if type(module).__name__ == class_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {class_name}, found {len(matches)}"
        )
    return matches[0]


def _find_all(model, class_name):
    matches = [module for module in model.modules() if type(module).__name__ == class_name]
    if not matches:
        raise RuntimeError(f"Expected at least one {class_name}, found none")
    return matches


def _mesh_groups(mesh):
    if mesh.ndim == 1:
        shard_group = mesh.get_group()
        return shard_group, None
    return mesh["shard"].get_group(), mesh["replicate"].get_group()


def apply_fsdp(model, mesh):
    """Apply FSDP2 bottom-up and preserve the Qwen tied embedding/head weight."""
    if mesh is None:
        return model

    qwen_text = _find_exactly_one(model, "Qwen3VLTextModel")
    vision = _find_exactly_one(model, "Qwen3VLVisionModel")
    text_blocks = _find_all(qwen_text, "Qwen3VLTextDecoderLayer")
    vision_blocks = _find_all(vision, "Qwen3VLVisionBlock")
    dit_blocks = _find_all(model, "BasicTransformerBlock")
    vision_patch_embeds = _find_all(vision, "Qwen3VLVisionPatchEmbed")
    vision_mergers = _find_all(vision, "Qwen3VLVisionPatchMerger")
    try:
        embed_tokens = model.qwen.model.language_model.embed_tokens
        lm_head = model.qwen.lm_head
    except AttributeError as error:
        raise RuntimeError("Unexpected Qwen module layout for FSDP2") from error

    if type(embed_tokens).__name__ != "Embedding":
        raise RuntimeError(
            "Expected language_model.embed_tokens to be Embedding, got "
            f"{type(embed_tokens).__name__}"
        )
    if lm_head.weight is not embed_tokens.weight:
        raise RuntimeError("Expected Qwen lm_head and embed_tokens to share one weight")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    rank = _env_int("RANK", 0)
    granularity = _fsdp_wrap_granularity()
    if rank == 0:
        topology = "hsdp" if mesh.ndim == 2 else "fsdp"
        if granularity == "block":
            print(
                "[FSDP2_WRAP_PLAN] "
                f"topology={topology} granularity=block "
                f"text_blocks={len(text_blocks)} "
                f"vision_blocks={len(vision_blocks)} "
                f"dit_blocks={len(dit_blocks)} "
                f"vision_patch_embeds={len(vision_patch_embeds)} "
                f"vision_mergers={len(vision_mergers)} "
                "child_reshard_after_forward=True "
                "vision_parent=remaining qwen_parent=remaining root=remaining",
                flush=True,
            )
        else:
            print(
                "[FSDP2_WRAP_PLAN] "
                f"topology={topology} granularity=coarse "
                "vision=whole vision_reshard_after_forward=False "
                "qwen_text=whole qwen_reshard_after_forward=False "
                "embed=explicit dit=root",
                flush=True,
            )

    # FSDP2 assigns parameter groups bottom-up. In block mode, each repeated
    # transformer block gets its own all-gather/reduce-scatter group. The
    # parent calls below then manage only parameters that live outside those
    # children (for example final norms and positional embeddings).
    leaf_modules = [
        *vision_blocks,
        *vision_patch_embeds,
        *vision_mergers,
        *text_blocks,
        *dit_blocks,
    ]
    if granularity == "block":
        for child in leaf_modules:
            fully_shard(
                child,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=mp_policy,
            )

    fully_shard(
        embed_tokens,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )

    if granularity == "block":
        fully_shard(
            vision,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )
        fully_shard(
            qwen_text,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )
    else:
        fully_shard(
            vision,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=mp_policy,
        )
        fully_shard(
            qwen_text,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=mp_policy,
        )

    # fully_shard replaces the embedding Parameter with a DTensor Parameter.
    # Restore Qwen's tied lm_head alias, then keep that already-owned Parameter
    # out of the root parameter group.
    shared_weight = embed_tokens.weight
    lm_head.weight = shared_weight
    if lm_head.weight is not shared_weight:
        raise RuntimeError("Failed to restore the Qwen shared embedding/head weight")

    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=None,
        mp_policy=mp_policy,
        ignored_params={shared_weight},
    )

    plain_trainable = [
        (name, type(parameter).__name__, str(parameter.device), str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not isinstance(parameter, DTensor)
    ]
    if plain_trainable:
        raise RuntimeError(
            "Trainable parameters remain outside FSDP2: "
            f"count={len(plain_trainable)} preview={plain_trainable[:20]}"
        )

    fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    expected_fsdp_modules = 4 if granularity == "coarse" else len(leaf_modules) + 4
    if len(fsdp_modules) != expected_fsdp_modules:
        raise RuntimeError(
            "Unexpected FSDP2 unit count: "
            f"granularity={granularity} expected={expected_fsdp_modules} "
            f"found={len(fsdp_modules)}"
        )

    # ``fully_shard`` has now attached the FSDP2 prefetch APIs to every child
    # transformer block.  Configure each sequential stack independently so
    # the next all-gather is requested while its predecessor is executing.
    install_explicit_prefetch(
        text_blocks=text_blocks,
        vision_blocks=vision_blocks,
        dit_blocks=dit_blocks,
    )

    shard_group, replicate_group = _mesh_groups(mesh)
    shard_size = dist.get_world_size(shard_group)
    replicate_size = 1 if replicate_group is None else dist.get_world_size(replicate_group)
    print(
        "[FSDP2_APPLIED] "
        f"rank={rank} granularity={granularity} "
        f"wrap_level={choice('SPIRIT_FSDP2_WRAP_LEVEL', 1, ('coarse', 'block'), {'coarse': 0, 'block': 1})} "
        f"fsdp_modules={len(fsdp_modules)} "
        "plain_trainable=0 mp_dtype=bf16 "
        f"shard_size={shard_size} replicate_size={replicate_size}",
        flush=True,
    )
    return model


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
