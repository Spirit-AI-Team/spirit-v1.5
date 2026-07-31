"""Opt-in distributed PyTorch/MUSA profiler for the training loop."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def install_runtime_profiler() -> None:
    """Capture complete compute micro-steps from train_fsdp2.py."""
    enabled = os.environ.get("SPIRIT_PROFILE", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    if getattr(install_runtime_profiler, "_installed", False):
        return

    rank = int(os.environ.get("RANK", "0"))

    profile_dir = Path(
        os.environ.get("SPIRIT_PROFILER_DIR", "./profiles/fsdp2")
    )
    start_step = _env_int("SPIRIT_PROFILE_START_STEP", 2)
    active_steps = _env_int("SPIRIT_PROFILE_ACTIVE_STEPS", 1)
    ga_steps = _env_int("SPIRIT_GRAD_ACCUM_STEPS", 1)
    max_micro_steps = _env_int("SPIRIT_MAX_TRAIN_STEPS", 1)
    if active_steps < 1:
        raise ValueError("SPIRIT_PROFILE_ACTIVE_STEPS must be >= 1")
    if ga_steps < 1:
        raise ValueError("SPIRIT_GRAD_ACCUM_STEPS must be >= 1")
    if start_step + active_steps > max_micro_steps:
        raise ValueError(
            "Profiler range exceeds SPIRIT_MAX_TRAIN_STEPS: "
            f"start={start_step} active={active_steps} "
            f"max={max_micro_steps}"
        )
    profile_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    musa_activity = getattr(torch.profiler.ProfilerActivity, "MUSA", None)
    if musa_activity is None:
        raise RuntimeError("This torch build does not expose ProfilerActivity.MUSA")
    activities.append(musa_activity)

    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    state = {
        "forward_index": -1,
        "started": False,
        "captured": 0,
        "finished": False,
    }

    from model import SpiritVLAPolicy

    original_forward = SpiritVLAPolicy.forward

    def distributed_barrier() -> None:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()

    def profiled_forward(model, *args, **kwargs):
        state["forward_index"] += 1
        if (
            not state["started"]
            and not state["finished"]
            and state["forward_index"] == start_step
        ):
            # Profiling only rank 0 makes the other ranks enter FSDP
            # collectives much earlier and can trip the MUSA communication
            # watchdog. Start every rank together instead.
            distributed_barrier()
            torch.cuda.synchronize()
            profiler.start()
            state["started"] = True
            if rank == 0:
                print(
                    "[FSDP2_PROFILER] "
                    f"capture_started micro_step={start_step} "
                    f"active_micro_steps={active_steps} "
                    f"active_optimizer_updates={active_steps // ga_steps} "
                    f"profiled_ranks={os.environ.get('WORLD_SIZE', '1')} "
                    f"dir={profile_dir}",
                    flush=True,
                )
        return original_forward(model, *args, **kwargs)

    SpiritVLAPolicy.forward = profiled_forward

    original_adamw = torch.optim.AdamW

    class ProfiledAdamW(original_adamw):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            if state["started"] and not state["finished"]:
                state["captured"] += 1
                if state["captured"] >= active_steps:
                    torch.cuda.synchronize()
                    profiler.stop()
                    trace_path = profile_dir / f"rank{rank}_trace.json"
                    summary_path = (
                        profile_dir / f"rank{rank}_key_averages.txt"
                    )
                    profiler.export_chrome_trace(str(trace_path))
                    events = profiler.key_averages(group_by_input_shape=True)
                    sort_key = "self_privateuse1_time_total"
                    try:
                        table = events.table(sort_by=sort_key, row_limit=300)
                    except Exception:
                        sort_key = "self_cpu_time_total"
                        table = events.table(sort_by=sort_key, row_limit=300)
                    summary_path.write_text(
                        f"sort_by: {sort_key}\n{table}\n",
                        encoding="utf-8",
                    )
                    state["finished"] = True
                    state["started"] = False
                    # Every rank performs equivalent trace/table work and then
                    # re-synchronizes before the next training micro-step.
                    distributed_barrier()
                    if rank == 0:
                        print(
                            "[FSDP2_PROFILER] capture_finished "
                            f"trace_pattern={profile_dir}/rank*_trace.json "
                            f"summary_pattern={profile_dir}/"
                            "rank*_key_averages.txt",
                            flush=True,
                        )
            return result

    torch.optim.AdamW = ProfiledAdamW
    install_runtime_profiler._installed = True
