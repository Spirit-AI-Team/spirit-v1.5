import os

import torch
from ..config import flag
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
)


_original_fast_pos = (
    Qwen3VLVisionModel.fast_pos_embed_interpolate
)
_original_rot_pos = Qwen3VLVisionModel.rot_pos_emb


def _cache_enabled():
    return flag("SPIRIT_VISION_GEOM_CACHE", 1)


def _cpu_grid_and_key(module, grid_thw):
    # fast_pos和rot_pos在一次forward中收到同一个tensor，
    # 避免重复执行第二次DtoH。
    if (
        getattr(
            module,
            "_spirit_last_grid_source",
            None,
        )
        is grid_thw
    ):
        return (
            module._spirit_last_cpu_grid,
            module._spirit_last_grid_key,
        )

    if grid_thw.device.type == "cpu":
        cpu_grid = grid_thw.contiguous()
    else:
        cpu_grid = (
            grid_thw.detach()
            .to(
                device="cpu",
                non_blocking=False,
            )
            .contiguous()
        )

    grid_key = tuple(
        tuple(int(value) for value in row)
        for row in cpu_grid.tolist()
    )

    module._spirit_last_grid_source = grid_thw
    module._spirit_last_cpu_grid = cpu_grid
    module._spirit_last_grid_key = grid_key

    return cpu_grid, grid_key


def _build_fast_geometry(module, grid_key):
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    split_sizes = []
    layouts = []

    for temporal, height, width in grid_key:
        h_idxs = torch.linspace(
            0,
            module.num_grid_per_side - 1,
            height,
        )
        w_idxs = torch.linspace(
            0,
            module.num_grid_per_side - 1,
            width,
        )

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (
            h_idxs.int() + 1
        ).clip(
            max=module.num_grid_per_side - 1
        )
        w_ceil = (
            w_idxs.int() + 1
        ).clip(
            max=module.num_grid_per_side - 1
        )

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        base_h = (
            h_floor * module.num_grid_per_side
        )
        base_h_ceil = (
            h_ceil * module.num_grid_per_side
        )

        indices = [
            (
                base_h[None].T
                + w_floor[None]
            ).flatten(),
            (
                base_h[None].T
                + w_ceil[None]
            ).flatten(),
            (
                base_h_ceil[None].T
                + w_floor[None]
            ).flatten(),
            (
                base_h_ceil[None].T
                + w_ceil[None]
            ).flatten(),
        ]

        weights = [
            (
                (1 - dh)[None].T
                * (1 - dw)[None]
            ).flatten(),
            (
                (1 - dh)[None].T
                * dw[None]
            ).flatten(),
            (
                dh[None].T
                * (1 - dw)[None]
            ).flatten(),
            (
                dh[None].T
                * dw[None]
            ).flatten(),
        ]

        for index in range(4):
            idx_list[index].extend(
                indices[index].tolist()
            )
            weight_list[index].extend(
                weights[index].tolist()
            )

        split_sizes.append(height * width)
        layouts.append(
            (temporal, height, width)
        )

    idx_tensor = torch.tensor(
        idx_list,
        dtype=torch.long,
        device=module.pos_embed.weight.device,
    )
    weight_tensor = torch.tensor(
        weight_list,
        dtype=module.pos_embed.weight.dtype,
        device=module.pos_embed.weight.device,
    )

    return (
        idx_tensor,
        weight_tensor,
        tuple(split_sizes),
        tuple(layouts),
    )


def _run_fast_geometry(module, geometry):
    (
        idx_tensor,
        weight_tensor,
        split_sizes,
        layouts,
    ) = geometry

    # 这里必须每步执行，保证pos_embed.weight正常训练。
    pos_embeds = (
        module.pos_embed(idx_tensor)
        * weight_tensor[:, :, None]
    )
    patch_pos_embeds = (
        pos_embeds[0]
        + pos_embeds[1]
        + pos_embeds[2]
        + pos_embeds[3]
    )

    pieces = patch_pos_embeds.split(
        list(split_sizes)
    )
    merge_size = (
        module.config.spatial_merge_size
    )
    outputs = []

    for (
        pos_embed,
        (
            temporal,
            height,
            width,
        ),
    ) in zip(pieces, layouts):
        pos_embed = pos_embed.repeat(
            temporal,
            1,
        )
        pos_embed = (
            pos_embed.view(
                temporal,
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        outputs.append(pos_embed)

    return torch.cat(outputs)


def _fast_pos_cached(module, grid_thw):
    cpu_grid, grid_key = _cpu_grid_and_key(
        module,
        grid_thw,
    )

    if not _cache_enabled():
        return _original_fast_pos(
            module,
            cpu_grid,
        )

    cache = getattr(
        module,
        "_spirit_fast_geometry_cache",
        None,
    )
    if cache is None:
        cache = {}
        module._spirit_fast_geometry_cache = (
            cache
        )

    geometry = cache.get(grid_key)
    cache_hit = geometry is not None

    if geometry is None:
        if len(cache) >= 8:
            cache.clear()
        geometry = _build_fast_geometry(
            module,
            grid_key,
        )
        cache[grid_key] = geometry

    optimized = _run_fast_geometry(
        module,
        geometry,
    )

    if not getattr(
        module,
        "_spirit_fast_geometry_validated",
        False,
    ):
        reference = _original_fast_pos(
            module,
            cpu_grid,
        )

        exact = torch.equal(
            optimized.detach(),
            reference.detach(),
        )
        max_abs_diff = float(
            (
                optimized.detach().float()
                - reference.detach().float()
            )
            .abs()
            .max()
            .item()
        )

        if not exact:
            raise RuntimeError(
                "fast_pos geometry-cache "
                "validation failed: "
                f"exact={exact} "
                f"max_abs_diff={max_abs_diff}"
            )

        module._spirit_fast_geometry_validated = (
            True
        )
        print(
            "[VISION_GEOM_CACHE] "
            f"rank={os.environ.get('RANK', '0')} "
            "component=fast_pos "
            "validation=PASS "
            f"exact={exact} "
            f"max_abs_diff={max_abs_diff} "
            f"cache_hit={cache_hit} "
            f"entries={len(cache)} "
            f"grid_shape={tuple(grid_thw.shape)}",
            flush=True,
        )

    if (
        cache_hit
        and not getattr(
            module,
            "_spirit_fast_cache_hit_logged",
            False,
        )
    ):
        module._spirit_fast_cache_hit_logged = (
            True
        )
        print(
            "[VISION_GEOM_CACHE] "
            f"rank={os.environ.get('RANK', '0')} "
            "component=fast_pos "
            "cache_hit=PASS",
            flush=True,
        )

    return optimized


def _build_rot_pos_ids(
    module,
    grid_key,
    device,
):
    merge_size = module.spatial_merge_size
    total_tokens = sum(
        temporal * height * width
        for temporal, height, width
        in grid_key
    )

    pos_ids = torch.empty(
        (total_tokens, 2),
        dtype=torch.long,
        device=device,
    )

    offset = 0

    for temporal, height, width in grid_key:
        merged_h = height // merge_size
        merged_w = width // merge_size

        block_rows = torch.arange(
            merged_h,
            device=device,
        )
        block_cols = torch.arange(
            merged_w,
            device=device,
        )
        intra_rows = torch.arange(
            merge_size,
            device=device,
        )
        intra_cols = torch.arange(
            merge_size,
            device=device,
        )

        row_idx = (
            block_rows[:, None, None, None]
            * merge_size
            + intra_rows[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None]
            * merge_size
            + intra_cols[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h,
            merged_w,
            merge_size,
            merge_size,
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h,
            merged_w,
            merge_size,
            merge_size,
        ).reshape(-1)

        coords = torch.stack(
            (row_idx, col_idx),
            dim=-1,
        )

        if temporal > 1:
            coords = coords.repeat(
                temporal,
                1,
            )

        count = coords.shape[0]
        pos_ids[
            offset:offset + count
        ] = coords
        offset += count

    if offset != total_tokens:
        raise RuntimeError(
            "rot_pos geometry size mismatch: "
            f"offset={offset} "
            f"total_tokens={total_tokens}"
        )

    return pos_ids


def _rot_pos_cached(module, grid_thw):
    cpu_grid, grid_key = _cpu_grid_and_key(
        module,
        grid_thw,
    )

    if not _cache_enabled():
        return _original_rot_pos(
            module,
            cpu_grid,
        )

    max_hw = max(
        max(height, width)
        for _, height, width in grid_key
    )
    freq_table = module.rotary_pos_emb(
        max_hw
    )
    device = freq_table.device

    cache = getattr(
        module,
        "_spirit_rot_geometry_cache",
        None,
    )
    if cache is None:
        cache = {}
        module._spirit_rot_geometry_cache = cache

    cache_key = (
        grid_key,
        str(device),
    )
    pos_ids = cache.get(cache_key)
    cache_hit = pos_ids is not None

    if pos_ids is None:
        if len(cache) >= 8:
            cache.clear()
        pos_ids = _build_rot_pos_ids(
            module,
            grid_key,
            device,
        )
        cache[cache_key] = pos_ids

    optimized = freq_table[
        pos_ids
    ].flatten(1)

    if not getattr(
        module,
        "_spirit_rot_geometry_validated",
        False,
    ):
        reference = _original_rot_pos(
            module,
            cpu_grid,
        )

        exact = torch.equal(
            optimized.detach(),
            reference.detach(),
        )
        max_abs_diff = float(
            (
                optimized.detach().float()
                - reference.detach().float()
            )
            .abs()
            .max()
            .item()
        )

        if not exact:
            raise RuntimeError(
                "rot_pos geometry-cache "
                "validation failed: "
                f"exact={exact} "
                f"max_abs_diff={max_abs_diff}"
            )

        module._spirit_rot_geometry_validated = (
            True
        )
        print(
            "[VISION_GEOM_CACHE] "
            f"rank={os.environ.get('RANK', '0')} "
            "component=rot_pos "
            "validation=PASS "
            f"exact={exact} "
            f"max_abs_diff={max_abs_diff} "
            f"cache_hit={cache_hit} "
            f"entries={len(cache)}",
            flush=True,
        )

    if (
        cache_hit
        and not getattr(
            module,
            "_spirit_rot_cache_hit_logged",
            False,
        )
    ):
        module._spirit_rot_cache_hit_logged = True
        print(
            "[VISION_GEOM_CACHE] "
            f"rank={os.environ.get('RANK', '0')} "
            "component=rot_pos "
            "cache_hit=PASS",
            flush=True,
        )

    return optimized


Qwen3VLVisionModel.fast_pos_embed_interpolate = (
    _fast_pos_cached
)
Qwen3VLVisionModel.rot_pos_emb = (
    _rot_pos_cached
)
