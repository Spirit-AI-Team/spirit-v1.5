"""FSDP2 runtime gradient-accumulation patch for the benchmark script.

The base training loop performs an optimizer update for every batch. This
module changes that behavior at runtime for ``train_fsdp2.py``.
"""

from __future__ import annotations

import builtins
import os
import re
import statistics
import sys
from dataclasses import dataclass, field

import torch

from fsdp2_runtime.flop_estimator import RuntimeModelFlopEstimator
from .config import choice


_optimizer_class_wrapper = None


def wrap_optimizer_for_gradient_accumulation(optimizer_class):
    """Return an optimizer class whose updates obey the active GA boundary.

    The standard AdamW path is patched during GA installation.  Optional
    optimizers, such as MUSA FusedAdamW, are selected later by the training
    entrypoint and must opt in through this helper.
    """
    if _optimizer_class_wrapper is None:
        return optimizer_class
    return _optimizer_class_wrapper(optimizer_class)


@dataclass
class _GAState:
    steps: int
    max_updates: int
    batch_size_per_rank: int
    world_size: int
    micro_in_update: int = 0
    current_accumulation_steps: int = 0
    last_update_micro_steps: int = 0
    completed_micro_steps: int = 0
    completed_updates: int = 0
    optimizer_stepped: bool = False
    just_completed: bool = False
    total_params: int | None = None
    losses: list[float] = field(default_factory=list)
    micro_step_seconds: list[float] = field(default_factory=list)
    micro_data_seconds: list[float] = field(default_factory=list)
    measured_update_seconds: list[float] = field(default_factory=list)
    measured_data_seconds: list[float] = field(default_factory=list)
    measured_micro_steps: list[int] = field(default_factory=list)
    in_original_mfu_block: bool = False

    @property
    def micro_global_batch(self) -> int:
        return self.batch_size_per_rank * self.world_size

    @property
    def effective_global_batch(self) -> int:
        return self.micro_global_batch * self.last_update_micro_steps


def _argv_int(flag: str) -> int:
    try:
        return int(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Gradient accumulation requires {flag}") from exc


def _metric(line: str, name: str) -> float:
    match = re.search(
        rf"(?:^|\|)\s*{re.escape(name)}:\s*([-+0-9.eE]+)",
        line,
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} in benchmark log: {line}")
    return float(match.group(1))


def _set_fsdp2_gradient_sync(model, sync_boundary: bool, mode: str) -> None:
    """Apply the selected FSDP2 accumulation communication policy."""
    if mode == "all_reduce":
        set_all_reduce = getattr(model, "set_requires_all_reduce", None)
        if set_all_reduce is not None:
            # HSDP: reduce-scatter every micro-step, but all-reduce only at
            # the optimizer boundary. Pure FSDP has no replica all-reduce,
            # so this is intentionally a no-op beyond setting the flag.
            set_all_reduce(sync_boundary, recurse=True)
            return
        if getattr(model, "set_requires_gradient_sync", None) is not None:
            raise RuntimeError(
                "SPIRIT_GA_SYNC_MODE=1 (all_reduce) requires an FSDP2 build "
                "with set_requires_all_reduce()"
            )
        return
    if mode == "no_sync":
        set_gradient_sync = getattr(
            model,
            "set_requires_gradient_sync",
            None,
        )
        if set_gradient_sync is not None:
            # This suppresses both reduce-scatter and all-reduce and keeps
            # full gradients resident between micro-steps.
            set_gradient_sync(sync_boundary, recurse=True)
        return
    raise ValueError(
        "SPIRIT_GA_SYNC_MODE must be 0 (no_sync) or 1 (all_reduce), "
        f"got {mode!r}"
    )


def install_gradient_accumulation(steps: int | None = None) -> None:
    """Install the patch once, using ``SPIRIT_GRAD_ACCUM_STEPS`` by default."""
    if getattr(install_gradient_accumulation, "_installed", False):
        return

    steps = steps or int(os.environ.get("SPIRIT_GRAD_ACCUM_STEPS", "1"))
    if steps < 1:
        raise ValueError(f"gradient accumulation steps must be >= 1, got {steps}")
    if steps == 1:
        return

    sync_mode = ("no_sync", "all_reduce")[choice(
        "SPIRIT_GA_SYNC_MODE", 1,
        ("no_sync", "all_reduce"),
        {"no_sync": 0, "all_reduce": 1},
    )]

    max_micro_steps = _argv_int("--max_train_steps")
    warmup_micro_steps = _argv_int("--benchmark_warmup_steps")
    batch_size = _argv_int("--batch_size")
    _argv_int("--save_steps")

    # Every CLI step value stays in the upstream loop's native micro-step
    # unit. A short final accumulation window is handled explicitly.
    if max_micro_steps < 1:
        raise ValueError(
            "--max_train_steps must be at least 1 micro-step, "
            f"got {max_micro_steps}"
        )
    if warmup_micro_steps < 0:
        raise ValueError(
            "--benchmark_warmup_steps must be >= 0, "
            f"got {warmup_micro_steps}"
        )
    if warmup_micro_steps >= max_micro_steps:
        raise ValueError(
            "--benchmark_warmup_steps must be smaller than "
            f"--max_train_steps, got {warmup_micro_steps} >= {max_micro_steps}"
        )
    max_updates = (max_micro_steps + steps - 1) // steps

    state = _GAState(
        steps=steps,
        max_updates=max_updates,
        batch_size_per_rank=batch_size,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    flop_estimate: dict[str, float] | None = None

    from model import SpiritVLAPolicy
    from utils import Logger

    original_forward = SpiritVLAPolicy.forward

    def accumulated_forward(model, *args, **kwargs):
        nonlocal flop_estimate
        if state.micro_in_update == 0:
            remaining = max_micro_steps - state.completed_micro_steps
            state.current_accumulation_steps = min(state.steps, remaining)
        if state.micro_in_update >= state.current_accumulation_steps:
            raise RuntimeError("GA state advanced past an optimizer boundary")
        state.micro_in_update += 1
        state.completed_micro_steps += 1
        state.just_completed = False
        sync_gradients = (
            state.micro_in_update == state.current_accumulation_steps
        )

        _set_fsdp2_gradient_sync(
            model,
            sync_gradients,
            sync_mode,
        )

        estimator = None
        if flop_estimate is None:
            estimator = RuntimeModelFlopEstimator(
                model,
                state.batch_size_per_rank,
            )
            estimator.start()
        output = original_forward(model, *args, **kwargs)
        if estimator is not None:
            flop_estimate = estimator.finish()
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[RUNTIME_MODEL_FLOPS] "
                    f"forward_per_sample="
                    f"{flop_estimate['forward_flops_per_sample'] / 1.0e12:.6f} TFLOPs "
                    f"linear={flop_estimate.get('linear', 0.0) / 1.0e12:.3f}T "
                    f"conv={flop_estimate.get('conv', 0.0) / 1.0e12:.3f}T "
                    f"text_attn={flop_estimate.get('text_attention_qkav', 0.0) / 1.0e12:.3f}T "
                    f"vision_attn={flop_estimate.get('vision_attention_qkav', 0.0) / 1.0e12:.3f}T "
                    f"dit_attn={flop_estimate.get('dit_attention_qkav', 0.0) / 1.0e12:.3f}T",
                    flush=True,
                )
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("Expected model forward to return (loss, log_dict)")
        loss = output[0]
        state.losses.append(float(loss.detach().item()))
        return (loss / state.current_accumulation_steps, *output[1:])

    SpiritVLAPolicy.forward = accumulated_forward

    optimizer_class_cache = {}

    def accumulating_optimizer_class(optimizer_class):
        cached = optimizer_class_cache.get(optimizer_class)
        if cached is not None:
            return cached

        class AccumulatingOptimizer(optimizer_class):
            def step(self, *args, **kwargs):
                state.optimizer_stepped = (
                    state.micro_in_update
                    == state.current_accumulation_steps
                )
                if not state.optimizer_stepped:
                    return None
                return super().step(*args, **kwargs)

            def zero_grad(self, *args, **kwargs):
                if not state.optimizer_stepped:
                    return None
                result = super().zero_grad(*args, **kwargs)
                state.completed_updates += 1
                state.last_update_micro_steps = (
                    state.current_accumulation_steps
                )
                state.micro_in_update = 0
                state.current_accumulation_steps = 0
                state.just_completed = True
                return result

        AccumulatingOptimizer.__name__ = (
            f"Accumulating{optimizer_class.__name__}"
        )
        optimizer_class_cache[optimizer_class] = AccumulatingOptimizer
        return AccumulatingOptimizer

    global _optimizer_class_wrapper
    _optimizer_class_wrapper = accumulating_optimizer_class
    torch.optim.AdamW = accumulating_optimizer_class(torch.optim.AdamW)

    # ``train_fsdp2`` installs the selected implementation first.  Capture it
    # here so the GA gate preserves the HSDP-aware path at the update boundary.
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def accumulated_clip_grad_norm(*args, **kwargs):
        if state.micro_in_update != state.current_accumulation_steps:
            return torch.tensor(0.0)
        return original_clip_grad_norm(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = accumulated_clip_grad_norm

    # Preserve GradScaler's READY -> UNSCALED -> STEPPED state machine across
    # micro-steps. Unscaling an already accumulated gradient on every
    # micro-step would repeatedly divide earlier gradients by the scale.
    grad_scaler = torch.amp.GradScaler
    original_scaler_unscale = grad_scaler.unscale_
    original_scaler_step = grad_scaler.step
    original_scaler_update = grad_scaler.update

    def accumulated_scaler_unscale(scaler, *args, **kwargs):
        if state.micro_in_update != state.current_accumulation_steps:
            return None
        return original_scaler_unscale(scaler, *args, **kwargs)

    def accumulated_scaler_step(scaler, *args, **kwargs):
        if state.micro_in_update != state.current_accumulation_steps:
            return None
        return original_scaler_step(scaler, *args, **kwargs)

    def accumulated_scaler_update(scaler, *args, **kwargs):
        if state.micro_in_update != state.current_accumulation_steps:
            return None
        return original_scaler_update(scaler, *args, **kwargs)

    grad_scaler.unscale_ = accumulated_scaler_unscale
    grad_scaler.step = accumulated_scaler_step
    grad_scaler.update = accumulated_scaler_update

    # LambdaLR.__init__ performs one initial scheduler step before training;
    # allow that call, then advance the schedule only after a real update.
    scheduler_base = torch.optim.lr_scheduler.LRScheduler
    original_scheduler_step = scheduler_base.step

    def accumulated_scheduler_step(scheduler, *args, **kwargs):
        if state.micro_in_update and not state.optimizer_stepped:
            return None
        return original_scheduler_step(scheduler, *args, **kwargs)

    scheduler_base.step = accumulated_scheduler_step

    # Keep the allocator peak over all micro-steps in one optimizer update.
    original_reset_peak = torch.cuda.reset_peak_memory_stats

    def accumulated_reset_peak(*args, **kwargs):
        if state.micro_in_update == 0:
            return original_reset_peak(*args, **kwargs)
        return None

    torch.cuda.reset_peak_memory_stats = accumulated_reset_peak

    original_logger_log = Logger.log

    def accumulated_log(logger, data, step):
        if not state.just_completed:
            return None
        return original_logger_log(logger, data, state.completed_micro_steps)

    Logger.log = accumulated_log
    original_logger_print = Logger.print

    def print_corrected_step(logger, line: str) -> None:
        step_seconds = sum(state.micro_step_seconds)
        data_seconds = sum(state.micro_data_seconds)
        train_seconds = step_seconds - data_seconds
        loss = statistics.mean(state.losses)
        samples_per_second = state.effective_global_batch / step_seconds
        images_per_sample = int(os.environ.get("SPIRIT_IMAGES_PER_SAMPLE", "3"))
        seq_len = int(os.environ.get("SPIRIT_MFU_SEQ_LEN", "295"))
        image_tokens = int(os.environ.get("SPIRIT_IMAGE_TOKENS", "240"))
        peak_per_gpu = float(os.environ.get("PEAK_TFLOPS_PER_GPU", "500"))
        if flop_estimate is None:
            raise RuntimeError("Runtime model FLOPs estimate is unavailable")
        model_flops = (
            3
            * flop_estimate["forward_flops_per_sample"]
            * state.effective_global_batch
        )
        if train_seconds <= 0:
            raise RuntimeError(
                "MFU train_time must be positive; "
                f"step_time={step_seconds:.6f}s data_time={data_seconds:.6f}s"
            )
        model_tflops_per_second = model_flops / train_seconds / 1.0e12
        mfu = (
            model_tflops_per_second
            / (peak_per_gpu * state.world_size)
            * 100.0
        )

        update_start_micro = (
            state.completed_micro_steps - state.last_update_micro_steps
        )
        if update_start_micro >= warmup_micro_steps:
            state.measured_update_seconds.append(step_seconds)
            state.measured_data_seconds.append(data_seconds)
            state.measured_micro_steps.append(
                state.last_update_micro_steps
            )

        original_logger_print(
            logger,
            f"Step {state.completed_micro_steps}/{max_micro_steps} | "
            f"Update {state.completed_updates}/{state.max_updates} | "
            f"GA: {state.steps} | "
            f"accum: {state.last_update_micro_steps} | "
            f"Loss: {loss:.4f} | "
            f"LR: {_metric(line, 'LR'):.2e} | "
            f"step_time: {step_seconds:.3f}s | "
            f"data_time: {data_seconds:.3f}s | "
            f"train_time: {train_seconds:.3f}s | "
            f"micro_global_batch: {state.micro_global_batch} | "
            f"effective_global_batch: {state.effective_global_batch} | "
            f"samples/s: {samples_per_second:.2f} | "
            f"images/s: {samples_per_second * images_per_sample:.2f} | "
            f"seq_len: {float(seq_len):.1f} | "
            f"img_tok: {float(image_tokens):.1f} | "
            f"tok/s: {samples_per_second * seq_len:.1f} | "
            f"model_train: {model_tflops_per_second:.1f} TFLOPs/s | "
            f"MFU_train: {mfu:.2f}% | "
            f"peak_alloc: {_metric(line, 'peak_alloc'):.2f} GiB | "
            f"peak_reserved: {_metric(line, 'peak_reserved'):.2f} GiB | "
            f"end_alloc: {_metric(line, 'end_alloc'):.2f} GiB | "
            f"end_free: {_metric(line, 'end_free'):.2f} GiB",
        )

    def corrected_summary_lines() -> dict[str, str]:
        values = state.measured_update_seconds
        data_values = state.measured_data_seconds
        if not values:
            raise RuntimeError("No measured GA optimizer updates")
        ordered = sorted(values)
        count = len(ordered)
        mean_s = statistics.mean(values)
        measured_samples = (
            state.micro_global_batch * sum(state.measured_micro_steps)
        )
        return {
            "measured_steps:": f"measured_optimizer_updates: {count}",
            "step_mean_s:": f"step_mean_s: {mean_s:.6f}",
            "step_median_s:": f"step_median_s: {statistics.median(values):.6f}",
            "step_p10_s:": f"step_p10_s: {ordered[max(0, int(0.10 * (count - 1)))]:.6f}",
            "step_p90_s:": f"step_p90_s: {ordered[min(count - 1, int(0.90 * (count - 1)))]:.6f}",
            "data_mean_s:": f"data_mean_s: {statistics.mean(data_values):.6f}",
            "samples_per_s_global:": (
                "samples_per_s_global: "
                f"{measured_samples / sum(values):.6f}"
            ),
        }

    def accumulated_print(logger, *args, **kwargs):
        line = str(args[0]) if args else ""
        if line.startswith("[TRAINABLE_SCOPE]"):
            match = re.search(r"total_params=(\d+)", line)
            if match:
                state.total_params = int(match.group(1))
        if line.startswith("Step "):
            state.micro_step_seconds.append(_metric(line, "step_time"))
            state.micro_data_seconds.append(_metric(line, "data_time"))
            if not state.just_completed:
                return None
            print_corrected_step(logger, line)
            state.losses.clear()
            state.micro_step_seconds.clear()
            state.micro_data_seconds.clear()
            return None
        if line.startswith("Starting training"):
            original_logger_print(
                logger,
                "[GRADIENT_ACCUMULATION] "
                f"steps={state.steps} "
                f"max_micro_steps={max_micro_steps} "
                f"max_optimizer_updates={state.max_updates} "
                f"warmup_micro_steps={warmup_micro_steps} "
                f"micro_batch_per_rank={state.batch_size_per_rank} "
                f"micro_global_batch={state.micro_global_batch} "
                f"full_effective_global_batch="
                f"{state.micro_global_batch * state.steps} "
                f"loss_scale=1/actual_accumulation sync_mode={sync_mode}",
            )
        for prefix, replacement in corrected_summary_lines().items() if state.measured_update_seconds else ():
            if line.startswith(prefix):
                return original_logger_print(logger, replacement, **kwargs)
        return original_logger_print(logger, *args, **kwargs)

    Logger.print = accumulated_print

    # The final MFU report uses builtins.print rather than Logger.print.
    original_print = builtins.print

    def corrected_final_mfu() -> None:
        values = state.measured_update_seconds
        if not values or flop_estimate is None:
            raise RuntimeError("Missing measured updates for final GA MFU")
        seq_len = int(os.environ.get("SPIRIT_MFU_SEQ_LEN", "295"))
        peak_per_gpu = float(os.environ.get("PEAK_TFLOPS_PER_GPU", "500"))
        measured_flops = (
            3
            * flop_estimate["forward_flops_per_sample"]
            * state.micro_global_batch
            * sum(state.measured_micro_steps)
        )
        cluster_peak = peak_per_gpu * 1.0e12 * state.world_size
        total_step_seconds = sum(values)
        total_data_seconds = sum(state.measured_data_seconds)
        total_train_seconds = total_step_seconds - total_data_seconds
        if total_train_seconds <= 0:
            raise RuntimeError(
                "Aggregate MFU train_time must be positive; "
                f"step_time={total_step_seconds:.6f}s "
                f"data_time={total_data_seconds:.6f}s"
            )
        aggregate_mfu = measured_flops / total_train_seconds / cluster_peak * 100.0
        original_print(
            "========== MFU (RUNTIME SHAPE-AWARE MODEL FLOPS) ==========",
            flush=True,
        )
        if state.total_params is not None:
            original_print(f"total_params_reference: {state.total_params}", flush=True)
        original_print(f"world_size: {state.world_size}", flush=True)
        original_print(f"gradient_accumulation_steps: {state.steps}", flush=True)
        original_print(f"micro_batch_size_per_rank: {state.batch_size_per_rank}", flush=True)
        original_print(f"micro_global_batch_size: {state.micro_global_batch}", flush=True)
        original_print(
            "full_effective_global_batch_size: "
            f"{state.micro_global_batch * state.steps}",
            flush=True,
        )
        original_print(
            f"measured_micro_steps: {sum(state.measured_micro_steps)}",
            flush=True,
        )
        original_print(f"seq_len: {seq_len}", flush=True)
        original_print(
            "forward_tflops_per_sample: "
            f"{flop_estimate['forward_flops_per_sample'] / 1.0e12:.6f}",
            flush=True,
        )
        original_print(f"peak_tflops_per_gpu: {peak_per_gpu:.3f}", flush=True)
        original_print(
            f"cluster_peak_tflops: {peak_per_gpu * state.world_size:.3f}",
            flush=True,
        )
        original_print(
            f"measured_flops_pflops: {measured_flops / 1.0e15:.6f}",
            flush=True,
        )
        original_print(
            f"total_train_time_seconds: {total_train_seconds:.6f}",
            flush=True,
        )
        original_print(
            f"mfu_aggregate_train_time_percent: {aggregate_mfu:.3f}",
            flush=True,
        )
        original_print(
            "mfu_formula: 3*runtime_forward_flops_per_sample*"
            "micro_global_batch*measured_micro_steps/total_train_time",
            flush=True,
        )

    def accumulated_builtin_print(*args, **kwargs):
        line = str(args[0]) if args else ""
        if line.startswith("========== MFU "):
            state.in_original_mfu_block = True
            corrected_final_mfu()
            return None
        if state.in_original_mfu_block:
            if line.startswith("mfu_formula:"):
                state.in_original_mfu_block = False
            return None
        return original_print(*args, **kwargs)

    builtins.print = accumulated_builtin_print
    install_gradient_accumulation._installed = True
