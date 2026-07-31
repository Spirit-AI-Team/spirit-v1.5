"""Small MUSA compatibility shims needed by ``torch.compile``."""

import os
import sys


def _configure_musa_inductor() -> int:
    """Keep foreach/combo kernels below the MUSA compiler's argument limit."""
    from torch._inductor import config

    # This compatibility limit is fixed for the installed torch-musa build;
    # exposing it as a training knob only created unvalidated configurations.
    max_args = 64
    if not 2 <= max_args <= 250:
        raise ValueError(
            "fixed MUSA combo kernel limit must be between 2 and 250; "
            f"got {max_args}"
        )
    config.combo_kernel_max_num_args = max_args
    return max_args


def _install_triton_benchmark_compat() -> None:
    """Drop a newer Inductor hint unsupported by this MUSA Triton build."""
    from torch_musa._inductor.utils import TritonBenchmarker

    original = TritonBenchmarker.benchmark_gpu
    if getattr(original, "_spirit_compile_compat", False):
        return

    def benchmark_gpu_compat(self, callable_, **kwargs):
        kwargs.pop("is_vetted_benchmarking", None)
        return original(self, callable_, **kwargs)

    benchmark_gpu_compat._spirit_compile_compat = True
    TritonBenchmarker.benchmark_gpu = benchmark_gpu_compat


def _install_musagraph_tree_alias() -> None:
    """Expose the CUDA-named graph-tree module used by Inductor backward."""
    import torch
    from torch_musa._inductor import musagraph_trees

    torch._inductor.cudagraph_trees = musagraph_trees
    sys.modules["torch._inductor.cudagraph_trees"] = musagraph_trees


def _unique_paths(paths):
    return list(dict.fromkeys(path for path in paths if path))


def _install_cpp_extension_compat() -> None:
    """Adapt torchada's extension helpers to the current PyTorch signature."""
    import torch
    from torch.utils import cpp_extension

    current_include_paths = cpp_extension.include_paths
    if getattr(current_include_paths, "_spirit_compile_compat", False):
        return
    if current_include_paths.__module__ != "torchada.utils.cpp_extension":
        return

    current_library_paths = cpp_extension.library_paths
    torch_root = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_root, "include")
    torch_library = os.path.join(torch_root, "lib")

    def include_paths_compat(device_type="cpu", torch_include_dirs=True):
        paths = []
        if torch_include_dirs:
            paths.extend(
                [
                    torch_include,
                    os.path.join(torch_include, "torch", "csrc", "api", "include"),
                ]
            )
        if device_type in {"cuda", "musa"}:
            paths.extend(current_include_paths(device_type=device_type))
        return _unique_paths(paths)

    def library_paths_compat(
        device_type="cpu", torch_include_dirs=True, cross_target_platform=None
    ):
        del cross_target_platform
        paths = [torch_library] if torch_include_dirs else []
        if device_type in {"cuda", "musa"}:
            paths.extend(current_library_paths(device_type=device_type))
        return _unique_paths(paths)

    include_paths_compat._spirit_compile_compat = True
    library_paths_compat._spirit_compile_compat = True
    cpp_extension.include_paths = include_paths_compat
    cpp_extension.library_paths = library_paths_compat


def _install_geometry_cache_graph_breaks() -> None:
    """Keep Python geometry caches and their exact checks in eager mode."""
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    disable = getattr(torch.compiler, "disable", torch._dynamo.disable)
    for method_name in ("fast_pos_embed_interpolate", "rot_pos_emb"):
        method = getattr(Qwen3VLVisionModel, method_name)
        if getattr(method, "_spirit_compile_graph_break", False):
            continue
        disabled_method = disable(method)
        disabled_method._spirit_compile_graph_break = True
        setattr(Qwen3VLVisionModel, method_name, disabled_method)


def _install_text_rope_graph_break() -> None:
    """Avoid a zero-stride BMM lowering unsupported by current muDNN."""
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    method = Qwen3VLTextRotaryEmbedding.forward
    if getattr(method, "_spirit_compile_graph_break", False):
        return
    disable = getattr(torch.compiler, "disable", torch._dynamo.disable)
    disabled_method = disable(method)
    disabled_method._spirit_compile_graph_break = True
    Qwen3VLTextRotaryEmbedding.forward = disabled_method


def _install_dit_block_graph_break() -> None:
    """Run nested FSDP2 DiT blocks outside Dynamo's fake-tensor trace."""
    import torch
    from model.modeling_spirit_vla import BasicTransformerBlock

    method = BasicTransformerBlock.forward
    if getattr(method, "_spirit_compile_graph_break", False):
        return
    disable = getattr(torch.compiler, "disable", torch._dynamo.disable)
    disabled_method = disable(method)
    disabled_method._spirit_compile_graph_break = True
    BasicTransformerBlock.forward = disabled_method


def _install_batch_preprocess_graph_break() -> None:
    """Prevent large batches from becoming one oversized fused copy kernel."""
    import torch
    from model.modeling_spirit_vla import SpiritVLAPolicy

    method = SpiritVLAPolicy.preprocess_rb_batch
    if getattr(method, "_spirit_compile_graph_break", False):
        return
    disable = getattr(torch.compiler, "disable", torch._dynamo.disable)
    disabled_method = disable(method)
    disabled_method._spirit_compile_graph_break = True
    SpiritVLAPolicy.preprocess_rb_batch = disabled_method


def install_torch_compile_compat() -> None:
    """Expose the CUDA-style stream namespace expected by TorchDynamo.

    torchada redirects ``torch.cuda`` attribute access to ``torch_musa``.  The
    MUSA package exports Stream directly, but does not export the matching
    ``streams`` module attribute.  TorchDynamo's guard builder unconditionally
    reads ``torch.cuda.streams.Stream``, so provide that missing namespace from
    torch_musa's own stream implementation.
    """
    import torch
    import torch_musa
    from torch_musa.core import stream as musa_streams

    if not hasattr(torch_musa, "streams"):
        torch_musa.streams = musa_streams

    # Support both direct attribute access and ``import torch.cuda.streams``.
    sys.modules["torch.cuda.streams"] = musa_streams

    resolved_stream = torch.cuda.streams.Stream
    if resolved_stream is not torch_musa.Stream:
        raise RuntimeError(
            "torch.compile stream compatibility installation failed: "
            f"torch.cuda.streams.Stream={resolved_stream!r}, "
            f"torch_musa.Stream={torch_musa.Stream!r}"
        )

    _configure_musa_inductor()
    _install_triton_benchmark_compat()
    _install_musagraph_tree_alias()
    _install_cpp_extension_compat()
    _install_geometry_cache_graph_breaks()
    _install_text_rope_graph_break()
    _install_dit_block_graph_break()
    _install_batch_preprocess_graph_break()
