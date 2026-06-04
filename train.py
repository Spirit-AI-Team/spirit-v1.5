# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import argparse
import json
import math
import os
import types
from dataclasses import dataclass, fields

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
    # Required param
    parser.add_argument("--pretrained_path", type=str, required=True, help="pretrained model path")
    parser.add_argument("--data_root", type=str, required=True, help="dataset path")
    # Optional param
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_steps", type=int, default=40000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--wandb_project", type=str, default="spirit-v1.5")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument("--norm_num_samples", type=int, default=20000)
    parser.add_argument("--norm_batch_size", type=int, default=32)
    return parser.parse_args()


def main():
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
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)

    logger.print("Creating model...")
    model = SpiritVLAPolicy.from_pretrained(ckpt_path= args.pretrained_path, strict=False, train=True).to(device)
    set_norm_stats(model, norm_stats, device)

    model.qwen.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
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

    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = build_cosine_scheduler(optimizer, warmup_steps, decay_steps, lr, decay_lr)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    logger.print("Starting training...")
    model.train()

    epoch_counter = 0
    data_iter = iter(dataloader)
    for step in range(args.max_train_steps):
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

        if step % args.log_interval == 0:
            log_dict["lr"] = scheduler.get_last_lr()[0]
            logger.log(log_dict, step)
            logger.print(
                f"Step {step}/{args.max_train_steps} | "
                f"Loss: {loss.item():.4f} | LR: {log_dict['lr']:.2e}"
            )

        if (step + 1) % args.save_steps == 0:
            save_model(model, step + 1, args.output_dir, global_rank)

    save_model(model, args.max_train_steps, args.output_dir, global_rank)

    logger.finish()
    cleanup()
    logger.print("Training complete!")


if __name__ == "__main__":
    main()
