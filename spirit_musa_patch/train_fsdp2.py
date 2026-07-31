# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import argparse
import statistics
import time
import json
import math
import os
import sys
import types
from dataclasses import dataclass, fields

import torchada

# Keep the repository's model and dataset packages separate from this MUSA
# runtime while allowing this file to be launched directly by torchrun.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_ROOT = os.environ.get("SPIRIT_PATCH_ROOT", _SCRIPT_DIR)
_REPO_ROOT = os.environ.get(
    "SPIRIT_REPO_ROOT",
    os.path.dirname(_PATCH_ROOT),
)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PATCH_ROOT)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from model import SpiritVLAPolicy, SpiritVLAConfig
from dataset import RoboChallengeDataset, DataConfig
from utils import (
    setup_distributed,
    apply_fsdp,
    cleanup,
    compute_norm_stats,
    save_model,
    Logger,
)

from fsdp2_runtime.config import choice, flag
from fsdp2_runtime import (
    apply_fsdp as apply_fsdp2,
    apply_model_patches,
    install_torch_compile_compat,
    load_training_checkpoint,
    pack_qwen_text_mlps,
    save_model as save_fsdp2_model,
    setup_distributed as setup_fsdp2_distributed,
)
from fsdp2_runtime.activation_checkpointing import (
    install_selective_qwen_recompute,
)
from fsdp2_runtime.checkpoint import read_training_checkpoint_step


def _configure_launcher_runtime():
    """Install launcher-only hooks before importing model dependencies."""
    install_selective_qwen_recompute()

    resume_training_state = os.environ.get(
        "SPIRIT_RESUME_TRAINING_STATE", ""
    ).strip()
    if resume_training_state:
        resumed_micro_step = read_training_checkpoint_step(
            resume_training_state
        )
        target_micro_steps = int(
            os.environ.get("SPIRIT_MAX_TRAIN_STEPS", "50")
        )
        if resumed_micro_step < 0 or resumed_micro_step >= target_micro_steps:
            raise RuntimeError(
                f"Resume step {resumed_micro_step} must be smaller than target "
                f"SPIRIT_MAX_TRAIN_STEPS={target_micro_steps}"
            )
        accumulation_steps = int(
            os.environ.get("SPIRIT_GRAD_ACCUM_STEPS", "1")
        )
        if resumed_micro_step % accumulation_steps:
            raise RuntimeError(
                "Training state was not saved at an accumulation boundary: "
                f"step={resumed_micro_step}, GA={accumulation_steps}"
            )
        os.environ["SPIRIT_RESUME_MICRO_STEP"] = str(resumed_micro_step)
        remaining_micro_steps = target_micro_steps - resumed_micro_step
        os.environ["SPIRIT_MAX_TRAIN_STEPS"] = str(remaining_micro_steps)
        warmup_micro_steps = int(
            os.environ.get("SPIRIT_BENCHMARK_WARMUP_STEPS", "5")
        )
        os.environ["SPIRIT_BENCHMARK_WARMUP_STEPS"] = str(
            min(warmup_micro_steps, max(remaining_micro_steps - 1, 0))
        )


_configure_launcher_runtime()


def _ensure_runtime_argv():
    """Expose argparse defaults to legacy runtime patches that inspect argv."""
    defaults = {
        "--data_root": os.environ.get(
            "DATA_ROOT", os.path.join(_REPO_ROOT, "fake_data", "move_objects_into_box")
        ),
        "--pretrained_path": os.environ.get(
            "SPIRIT_PRETRAINED_PATH", os.path.join(_REPO_ROOT, "checkpoint_config")
        ),
        "--output_dir": os.environ.get(
            "SPIRIT_OUTPUT_DIR", os.path.join(_REPO_ROOT, "outputs_fsdp2")
        ),
        "--batch_size": os.environ.get("SPIRIT_BATCH_SIZE", "240"),
        "--max_train_steps": os.environ.get("SPIRIT_MAX_TRAIN_STEPS", "50"),
        "--benchmark_warmup_steps": os.environ.get(
            "SPIRIT_BENCHMARK_WARMUP_STEPS", "5"
        ),
        "--save_steps": os.environ.get("SPIRIT_SAVE_STEPS", "999999"),
        "--log_interval": "999999",
        "--num_workers": "4",
        "--prefetch_factor": "2",
        "--norm_num_samples": "1",
        "--norm_batch_size": "1",
    }
    for flag_name, value in defaults.items():
        if flag_name not in sys.argv:
            sys.argv.extend((flag_name, str(value)))


def _install_runtime_hooks():
    """Install the FSDP2 training hooks formerly owned by the wrapper."""
    rank = int(os.environ.get("RANK", "0"))
    original_preprocess = SpiritVLAPolicy.preprocess_rb_batch

    def measured_preprocess(self, batch):
        outputs = original_preprocess(self, batch)
        input_ids_list = outputs[2]
        lengths = [int(input_ids.numel()) for input_ids in input_ids_list]
        if rank == 0 and lengths:
            print(
                "[SEQ295_CONTEXT] "
                f"num_images={len(outputs[1][0])} "
                f"effective_tokens_mean={sum(lengths) / len(lengths):.3f} "
                f"min={min(lengths)} max={max(lengths)}",
                flush=True,
            )
        return outputs

    SpiritVLAPolicy.preprocess_rb_batch = measured_preprocess

    from fsdp2_runtime.gradient_accumulation import (
        install_gradient_accumulation,
    )
    from fsdp2_runtime.gradient_clipping import install_gradient_clipping
    from fsdp2_runtime.profiler import install_runtime_profiler

    install_gradient_clipping()
    install_gradient_accumulation()
    install_runtime_profiler()


_training_checkpoint_state = {
    "model": None,
    "optimizer": None,
    "scheduler": None,
    "scaler": None,
    "resume_loaded": False,
}


@dataclass
class LoggerConfig:
    wandb_project: str = "spirit-v1.5"
    wandb_mode: str = "disabled"


def build_cosine_scheduler(optimizer, warmup_steps, decay_steps, base_lr, final_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if decay_steps <= 0:
            return 1.0
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1.0 - final_lr / base_lr) * cosine

    return LambdaLR(optimizer, lr_lambda)


def set_norm_stats(model, norm_stats, device):
    model.normalize_inputs.buffer_observation_state["min"].data.copy_(
        norm_stats["state_min"].to(device)
    )
    model.normalize_inputs.buffer_observation_state["max"].data.copy_(
        norm_stats["state_max"].to(device)
    )
    model.normalize_targets.buffer_action["min"].data.copy_(
        norm_stats["action_min"].to(device)
    )
    model.normalize_targets.buffer_action["max"].data.copy_(
        norm_stats["action_max"].to(device)
    )
    model.unnormalize_outputs.buffer_action["min"].data.copy_(
        norm_stats["action_min"].to(device)
    )
    model.unnormalize_outputs.buffer_action["max"].data.copy_(
        norm_stats["action_max"].to(device)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Spirit-v1.5 RoboChallenge finetune")
    # Defaults are environment-backed so torchrun can invoke this file
    # directly without a second Python wrapper.
    parser.add_argument(
        "--pretrained_path", type=str,
        default=os.environ.get(
            "SPIRIT_PRETRAINED_PATH", os.path.join(_REPO_ROOT, "checkpoint_config")
        ),
        help="pretrained model path",
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.environ.get(
            "DATA_ROOT", os.path.join(_REPO_ROOT, "fake_data", "move_objects_into_box")
        ),
        help="dataset path",
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=int(os.environ.get("SPIRIT_BATCH_SIZE", "240")),
    )
    parser.add_argument(
        "--max_train_steps", type=int,
        default=int(os.environ.get("SPIRIT_MAX_TRAIN_STEPS", "50")),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.environ.get(
            "SPIRIT_OUTPUT_DIR", os.path.join(_REPO_ROOT, "outputs_fsdp2")
        ),
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=999999)
    parser.add_argument(
        "--save_steps", type=int,
        default=int(os.environ.get("SPIRIT_SAVE_STEPS", "999999")),
    )
    parser.add_argument("--wandb_project", type=str, default="spirit-v1.5")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--norm_num_samples", type=int, default=1)
    parser.add_argument("--norm_batch_size", type=int, default=1)
    parser.add_argument(
        "--benchmark_warmup_steps", type=int,
        default=int(os.environ.get("SPIRIT_BENCHMARK_WARMUP_STEPS", "5")),
    )
    return parser.parse_args()


def _training_main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank, global_rank, world_size, mesh = setup_distributed()
    device = torch.device("cuda", local_rank)

    logger_config = LoggerConfig(
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )
    logger = Logger(logger_config, global_rank)
    logger.print(f"pretrained_path: {args.pretrained_path}")

    config_path = os.path.join(args.pretrained_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw_config = types.SimpleNamespace(**json.load(f))
    else:
        raise FileNotFoundError(f"No config.json found in {args.pretrained_path}")
    
    # n_action_steps should equal action_horizon while training
    logger.print(f"Model config loaded: dit={raw_config.dit_hidden_size}, "
                 f"n_action_steps={raw_config.n_action_steps}, "
                 f"attention={raw_config.attention_implementation}")

    logger.print("Loading dataset...")
    data_config = DataConfig(data_root=args.data_root, chunk_size=raw_config.chunk_size)
    dataset = RoboChallengeDataset(data_config)

    logger.print("Computing normalization stats...")
    norm_stats = compute_norm_stats(
        dataset,
        num_samples=args.norm_num_samples,
        batch_size=args.norm_batch_size,
        num_workers=args.num_workers,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    logger.print("Creating model...")
    model = SpiritVLAPolicy.from_pretrained(
        ckpt_path=args.pretrained_path,
        strict=False,
        train=True,
    ).to(device)
    set_norm_stats(model, norm_stats, device)

    trainable_scope = os.environ.get(
        "SPIRIT_TRAINABLE_SCOPE",
        "no-qwen",
    ).strip().lower()

    if trainable_scope not in {"full", "no-qwen"}:
        raise ValueError(
            "SPIRIT_TRAINABLE_SCOPE must be "
            "full or no-qwen, got "
            f"{trainable_scope!r}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    qwen_params = sum(p.numel() for p in model.qwen.parameters())

    if trainable_scope == "no-qwen":
        for param in model.qwen.parameters():
            param.requires_grad_(False)

        # A frozen Qwen does not need activation recomputation.
        # Disabling it also avoids checkpoint warnings when none of
        # the Qwen inputs require gradients.
        if hasattr(model.qwen, "gradient_checkpointing_disable"):
            model.qwen.gradient_checkpointing_disable()
    else:
        model.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

    trainable_params_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    frozen_params_count = total_params - trainable_params_count

    logger.print(
        "[TRAINABLE_SCOPE] "
        f"scope={trainable_scope} "
        f"total_params={total_params} "
        f"qwen_params={qwen_params} "
        f"trainable_params={trainable_params_count} "
        f"frozen_params={frozen_params_count}"
    )

    model = apply_fsdp(model, mesh)

    lr = args.lr or getattr(raw_config, "optimizer_lr", 2.5e-5)
    betas = tuple(getattr(raw_config, "optimizer_betas", [0.9, 0.95]))
    eps = getattr(raw_config, "optimizer_eps", 1e-8)
    weight_decay = getattr(raw_config, "optimizer_weight_decay", 1e-10)
    grad_clip_norm = getattr(raw_config, "optimizer_grad_clip_norm", 1.0)
    warmup_steps = args.warmup_steps or getattr(raw_config, "scheduler_warmup_steps", 1000)
    decay_steps = getattr(raw_config, "scheduler_decay_steps", 50000)
    decay_lr = getattr(raw_config, "scheduler_decay_lr", 2.5e-6)

    optimizer_params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]
    if not optimizer_params:
        raise RuntimeError("No trainable parameters remain")

    optimizer = AdamW(
        optimizer_params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
    )
    scheduler = build_cosine_scheduler(optimizer, warmup_steps, decay_steps, lr, decay_lr)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    logger.print("Starting training...")
    model.train()

    # model.train() also switches the frozen Qwen back to train mode.
    # Keep it in evaluation mode for deterministic frozen inference.
    if trainable_scope == "no-qwen":
        unwrapped_model = (
            model.module
            if hasattr(model, "module")
            else model
        )
        unwrapped_model.qwen.eval()

    epoch_counter = 0
    data_iter = iter(dataloader)
    benchmark_step_times = []
    benchmark_data_times = []

    # True allocator peaks, sampled using the peak-memory APIs rather
    # than current allocated memory after HSDP has resharded.
    benchmark_peak_allocated_bytes = 0
    benchmark_peak_reserved_bytes = 0
    benchmark_max_end_allocated_bytes = 0
    benchmark_max_end_reserved_bytes = 0
    benchmark_min_end_free_bytes = None
    benchmark_device_total_bytes = 0

    for step in range(args.max_train_steps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(local_rank)
        step_start = time.perf_counter()
        data_start = step_start
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch_counter += 1
            if sampler is not None:
                sampler.set_epoch(epoch_counter)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        torch.cuda.synchronize()
        data_seconds = time.perf_counter() - data_start

        with torch.autocast("cuda", dtype=amp_dtype):
            loss, log_dict = model(batch)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_start

        step_peak_allocated_bytes = (
            torch.cuda.max_memory_allocated(local_rank)
        )
        step_peak_reserved_bytes = (
            torch.cuda.max_memory_reserved(local_rank)
        )
        step_end_allocated_bytes = (
            torch.cuda.memory_allocated(local_rank)
        )
        step_end_reserved_bytes = (
            torch.cuda.memory_reserved(local_rank)
        )
        (
            step_end_free_bytes,
            step_device_total_bytes,
        ) = torch.cuda.mem_get_info(local_rank)

        benchmark_peak_allocated_bytes = max(
            benchmark_peak_allocated_bytes,
            step_peak_allocated_bytes,
        )
        benchmark_peak_reserved_bytes = max(
            benchmark_peak_reserved_bytes,
            step_peak_reserved_bytes,
        )
        benchmark_max_end_allocated_bytes = max(
            benchmark_max_end_allocated_bytes,
            step_end_allocated_bytes,
        )
        benchmark_max_end_reserved_bytes = max(
            benchmark_max_end_reserved_bytes,
            step_end_reserved_bytes,
        )
        benchmark_min_end_free_bytes = (
            step_end_free_bytes
            if benchmark_min_end_free_bytes is None
            else min(
                benchmark_min_end_free_bytes,
                step_end_free_bytes,
            )
        )
        benchmark_device_total_bytes = max(
            benchmark_device_total_bytes,
            step_device_total_bytes,
        )
        phase = (
            'WARMUP'
            if step < args.benchmark_warmup_steps
            else 'MEASURE'
        )
        if step >= args.benchmark_warmup_steps:
            benchmark_step_times.append(step_seconds)
            benchmark_data_times.append(data_seconds)

        if global_rank == 0:
            log_dict["lr"] = scheduler.get_last_lr()[0]

            mfu_seq_len = int(
                os.environ.get("SPIRIT_MFU_SEQ_LEN", "295")
            )
            image_tokens = int(
                os.environ.get("SPIRIT_IMAGE_TOKENS", "240")
            )
            images_per_sample = int(
                os.environ.get("SPIRIT_IMAGES_PER_SAMPLE", "3")
            )
            peak_tflops_per_gpu = float(
                os.environ.get("PEAK_TFLOPS_PER_GPU", "500")
            )

            global_batch_size = args.batch_size * world_size
            train_seconds = step_seconds - data_seconds
            samples_per_second = global_batch_size / step_seconds
            images_per_second = (
                samples_per_second * images_per_sample
            )
            tokens_per_second = (
                samples_per_second * mfu_seq_len
            )
            actual_tflops = (
                6
                * total_params
                * global_batch_size
                * mfu_seq_len
                / step_seconds
                / 1.0e12
            )
            step_mfu = (
                actual_tflops
                / (peak_tflops_per_gpu * world_size)
                * 100.0
            )

            logger.log(log_dict, step)
            logger.print(
                f"Step {step}/{args.max_train_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {log_dict['lr']:.2e} | "
                f"step_time: {step_seconds:.3f}s | "
                f"data_time: {data_seconds:.3f}s | "
                f"train_time: {train_seconds:.3f}s | "
                f"samples/s: {samples_per_second:.2f} | "
                f"images/s: {images_per_second:.2f} | "
                f"seq_len: {float(mfu_seq_len):.1f} | "
                f"img_tok: {float(image_tokens):.1f} | "
                f"tok/s: {tokens_per_second:.1f} | "
                f"actual: {actual_tflops:.1f} TFLOPs/s | "
                f"MFU: {step_mfu:.2f}% | "
                f"peak_alloc: "
                f"{step_peak_allocated_bytes / 1024**3:.2f} GiB | "
                f"peak_reserved: "
                f"{step_peak_reserved_bytes / 1024**3:.2f} GiB | "
                f"end_alloc: "
                f"{step_end_allocated_bytes / 1024**3:.2f} GiB | "
                f"end_free: "
                f"{step_end_free_bytes / 1024**3:.2f} GiB"
            )

        if (step + 1) % args.save_steps == 0:
            save_model(model, step + 1, args.output_dir, global_rank)

    if not benchmark_step_times:
        raise RuntimeError('No measured benchmark steps')

    # Aggregate the worst memory values across all 16 ranks after the
    # timed region, so this diagnostic collective does not affect step
    # timing.
    max_memory_stats = torch.tensor(
        [
            benchmark_peak_allocated_bytes,
            benchmark_peak_reserved_bytes,
            benchmark_max_end_allocated_bytes,
            benchmark_max_end_reserved_bytes,
            benchmark_device_total_bytes,
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(
        max_memory_stats,
        op=torch.distributed.ReduceOp.MAX,
    )

    min_free_stat = torch.tensor(
        [benchmark_min_end_free_bytes],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(
        min_free_stat,
        op=torch.distributed.ReduceOp.MIN,
    )
    torch.cuda.synchronize()

    (
        cluster_peak_allocated_bytes,
        cluster_peak_reserved_bytes,
        cluster_max_end_allocated_bytes,
        cluster_max_end_reserved_bytes,
        cluster_device_total_bytes,
    ) = [
        int(value)
        for value in max_memory_stats.cpu().tolist()
    ]
    cluster_min_end_free_bytes = int(
        min_free_stat.cpu().item()
    )

    ordered = sorted(benchmark_step_times)
    count = len(ordered)
    p10 = ordered[max(0, int(0.10 * (count - 1)))]
    p90 = ordered[min(count - 1, int(0.90 * (count - 1)))]
    median_s = statistics.median(ordered)
    mean_s = statistics.mean(ordered)
    logger.print('========== STEADY BASELINE ==========')
    logger.print(f'measured_steps: {count}')
    logger.print(f'step_mean_s: {mean_s:.6f}')
    logger.print(f'step_median_s: {median_s:.6f}')
    logger.print(f'step_p10_s: {p10:.6f}')
    logger.print(f'step_p90_s: {p90:.6f}')
    logger.print(
        f'data_mean_s: '
        f'{statistics.mean(benchmark_data_times):.6f}',
    )
    logger.print(
        f'samples_per_s_global: '
        f'{args.batch_size * world_size / mean_s:.6f}',
    )
    logger.print(
        f'true_peak_allocated_gib_all_ranks: '
        f'{cluster_peak_allocated_bytes / 1024**3:.3f}',
    )
    logger.print(
        f'true_peak_reserved_gib_all_ranks: '
        f'{cluster_peak_reserved_bytes / 1024**3:.3f}',
    )
    logger.print(
        f'max_step_end_allocated_gib_all_ranks: '
        f'{cluster_max_end_allocated_bytes / 1024**3:.3f}',
    )
    logger.print(
        f'max_step_end_reserved_gib_all_ranks: '
        f'{cluster_max_end_reserved_bytes / 1024**3:.3f}',
    )
    logger.print(
        f'min_step_end_free_gib_all_ranks: '
        f'{cluster_min_end_free_bytes / 1024**3:.3f}',
    )
    logger.print(
        f'device_total_gib: '
        f'{cluster_device_total_bytes / 1024**3:.3f}',
    )
    logger.print('checkpoint_save: skipped')

    # Moore Threads report-compatible full-model 6N MFU formula.
    if global_rank == 0:
        mfu_seq_len = int(
            os.environ.get("SPIRIT_MFU_SEQ_LEN", "295")
        )
        peak_tflops_per_gpu = float(
            os.environ.get(
                "PEAK_TFLOPS_PER_GPU",
                "500",
            )
        )

        global_batch_size = args.batch_size * world_size
        flops_per_step = (
            6
            * total_params
            * global_batch_size
            * mfu_seq_len
        )
        cluster_peak_flops = (
            peak_tflops_per_gpu
            * 1.0e12
            * world_size
        )

        mfu_by_mean = (
            flops_per_step
            / mean_s
            / cluster_peak_flops
            * 100.0
        )
        mfu_by_median = (
            flops_per_step
            / median_s
            / cluster_peak_flops
            * 100.0
        )

        print(
            "========== MFU "
            "(6N FULL-MODEL VENDOR FORMULA) ==========",
            flush=True,
        )
        print(
            f"total_params: {total_params}",
            flush=True,
        )
        print(
            f"world_size: {world_size}",
            flush=True,
        )
        print(
            f"batch_size_per_rank: {args.batch_size}",
            flush=True,
        )
        print(
            f"global_batch_size: {global_batch_size}",
            flush=True,
        )
        print(
            f"seq_len: {mfu_seq_len}",
            flush=True,
        )
        print(
            "peak_tflops_per_gpu: "
            f"{peak_tflops_per_gpu:.3f}",
            flush=True,
        )
        print(
            "cluster_peak_tflops: "
            f"{peak_tflops_per_gpu * world_size:.3f}",
            flush=True,
        )
        print(
            "flops_per_step_pflops: "
            f"{flops_per_step / 1.0e15:.6f}",
            flush=True,
        )
        print(
            "mfu_by_mean_step_percent: "
            f"{mfu_by_mean:.3f}",
            flush=True,
        )
        print(
            "mfu_by_median_step_percent: "
            f"{mfu_by_median:.3f}",
            flush=True,
        )
        print(
            "mfu_formula: "
            "6*N*global_batch*seq_len",
            flush=True,
        )

    logger.finish()
    cleanup()
    logger.print("Training complete!")


def configure_optimizer():
    """Select the reference or MUSA fused optimizer."""
    global AdamW

    optimizer_name = ("adamw", "fused_adamw")[choice(
        "SPIRIT_OPTIMIZER", 1, ("adamw", "fused_adamw"),
        {"adamw": 0, "fused_adamw": 1},
    )]
    if optimizer_name == "adamw":
        return

    from torch.distributed.tensor import DTensor
    from torch_musa.optim import FusedAdamW
    from fsdp2_runtime.gradient_accumulation import (
        wrap_optimizer_for_gradient_accumulation,
    )

    accumulating_fused_adamw = wrap_optimizer_for_gradient_accumulation(
        FusedAdamW
    )

    def fused_adamw_factory(params, **kwargs):
        kwargs.pop("foreach", None)
        params = list(params)
        if not params:
            raise RuntimeError("FusedAdamW received no trainable parameters")
        if int(os.environ.get("RANK", "0")) == 0:
            first_param = params[0]
            print(
                "[FSDP2_OPTIMIZER] "
                "requested=fused_adamw "
                f"selected={FusedAdamW.__module__}.{FusedAdamW.__name__} "
                f"ga_wrapped={accumulating_fused_adamw is not FusedAdamW} "
                f"param_count={len(params)} "
                f"first_param_type={type(first_param).__name__} "
                f"first_param_is_dtensor={isinstance(first_param, DTensor)} "
                f"first_param_device={first_param.device}",
                flush=True,
            )
        return accumulating_fused_adamw(params, **kwargs)

    AdamW = fused_adamw_factory


def maybe_compile(model):
    """Optionally compile an already sharded FSDP2 model."""
    if not flag("SPIRIT_TORCH_COMPILE", 0):
        return model

    install_torch_compile_compat()
    compile_kwargs = {"mode": "default", "fullgraph": False}
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[FSDP2_TORCH_COMPILE] "
            "enabled=True placement=after_fsdp "
            "musa_streams_compat=True musa_triton_benchmark_compat=True "
            "musa_graph_tree_alias=True torchada_cpp_extension_compat=True "
            "vision_geometry_cache_eager=True text_rope_eager=True "
            "dit_blocks_eager=True batch_preprocess_eager=True "
            "musa_combo_kernel_max_args=64 mode=default "
            "fullgraph=False backend=default "
            f"before_type={type(model)}",
            flush=True,
        )
    model = torch.compile(model, **compile_kwargs)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[FSDP2_TORCH_COMPILE] after_type={type(model)}", flush=True)
    return model


def prepare_model_for_training(model, mesh):
    """Pack optimized modules, apply mandatory FSDP2, then optional compile."""
    if int(os.environ.get("RANK", "0")) == 0:
        qwen_dtype_numel = {}
        for parameter in model.qwen.parameters():
            dtype_name = str(parameter.dtype)
            qwen_dtype_numel[dtype_name] = (
                qwen_dtype_numel.get(dtype_name, 0) + parameter.numel()
            )
        patch_embed = model.qwen.model.visual.patch_embed
        patch_embed_impl = (
            "gemm"
            if getattr(type(patch_embed), "_spirit_gemm_patch_installed", False)
            else "conv3d"
        )
        print(
            "[FSDP2_SOURCE_PARITY] "
            f"qwen_parameter_numel_by_dtype={qwen_dtype_numel} "
            f"patch_embed_impl={patch_embed_impl}",
            flush=True,
        )

    pack_qwen_text_mlps(model)
    fsdp_model = apply_fsdp2(model, mesh)
    prepared_model = maybe_compile(fsdp_model)
    _training_checkpoint_state["model"] = prepared_model
    return prepared_model


def install_training_checkpoint_hooks():
    """Capture optimizer state for FSDP2 model and training checkpoints."""
    global AdamW, build_cosine_scheduler, save_model

    original_adamw = AdamW
    original_scheduler_factory = build_cosine_scheduler
    original_grad_scaler = torch.amp.GradScaler

    def tracked_adamw(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        _training_checkpoint_state["optimizer"] = optimizer
        return optimizer

    def tracked_scheduler_factory(*args, **kwargs):
        scheduler = original_scheduler_factory(*args, **kwargs)
        _training_checkpoint_state["scheduler"] = scheduler
        resume_path = os.environ.get("SPIRIT_RESUME_TRAINING_STATE", "").strip()
        if resume_path:
            load_training_checkpoint(
                _training_checkpoint_state["model"],
                _training_checkpoint_state["optimizer"],
                resume_path,
                scheduler=scheduler,
            )
            _training_checkpoint_state["resume_loaded"] = True
        return scheduler

    def tracked_grad_scaler(*args, **kwargs):
        scaler = original_grad_scaler(*args, **kwargs)
        _training_checkpoint_state["scaler"] = scaler
        resume_path = os.environ.get("SPIRIT_RESUME_TRAINING_STATE", "").strip()
        if resume_path:
            load_training_checkpoint(
                _training_checkpoint_state["model"],
                _training_checkpoint_state["optimizer"],
                resume_path,
                scheduler=_training_checkpoint_state["scheduler"],
                scaler=scaler,
            )
            _training_checkpoint_state["resume_loaded"] = True
        return scaler

    def checkpointing_save_model(model, step, output_dir, rank):
        optimizer = _training_checkpoint_state["optimizer"]
        if optimizer is None:
            raise RuntimeError("Optimizer was not captured before checkpoint save")
        absolute_step = step + int(os.environ.get("SPIRIT_RESUME_MICRO_STEP", "0"))
        return save_fsdp2_model(
            model,
            absolute_step,
            output_dir,
            rank,
            optimizer=optimizer,
            scheduler=_training_checkpoint_state["scheduler"],
            scaler=_training_checkpoint_state["scaler"],
        )

    AdamW = tracked_adamw
    build_cosine_scheduler = tracked_scheduler_factory
    torch.amp.GradScaler = tracked_grad_scaler
    save_model = checkpointing_save_model


def main():
    global setup_distributed, apply_fsdp

    _ensure_runtime_argv()
    _install_runtime_hooks()
    apply_model_patches()
    configure_optimizer()
    setup_distributed = setup_fsdp2_distributed
    apply_fsdp = prepare_model_for_training
    install_training_checkpoint_hooks()
    _training_main()

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "QWEN FULL SEQ295 FSDP2 "
            f"mode={os.environ.get('SPIRIT_FSDP2_MODE', 'auto')} "
            f"world_size={os.environ.get('WORLD_SIZE', '1')}: PASSED",
            flush=True,
        )


if __name__ == "__main__":
    main()
