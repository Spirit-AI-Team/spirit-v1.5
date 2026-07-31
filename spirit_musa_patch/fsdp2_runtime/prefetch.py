"""Asynchronous host-to-device DataLoader prefetch for the FSDP2 runtime."""

from __future__ import annotations

import os
import sys

import torch

from .config import flag


def _move_batch_to_device(batch, device):
    """Match the stock training loop's top-level tensor transfer semantics."""
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


class _AsyncDevicePrefetchIterator:
    def __init__(self, source_iterator, device, stream_api):
        self._source_iterator = source_iterator
        self._device = device
        self._stream_api = stream_api
        self._copy_stream = stream_api.Stream(device=device)
        self._next_batch = None
        self._finished = False
        self._preload()

    def _preload(self) -> None:
        try:
            host_batch = next(self._source_iterator)
        except StopIteration:
            self._finished = True
            self._next_batch = None
            return

        with self._stream_api.stream(self._copy_stream):
            self._next_batch = _move_batch_to_device(host_batch, self._device)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            if self._finished:
                raise StopIteration
            raise RuntimeError("Async DataLoader prefetcher has no queued batch")

        # Make the default training stream wait for the copy of the current
        # batch, then immediately begin copying the following batch on the
        # separate stream.  The following copy overlaps model execution.
        self._stream_api.current_stream(self._device).wait_stream(self._copy_stream)
        batch = self._next_batch
        self._preload()
        return batch


class _AsyncDevicePrefetchDataLoader:
    """Transparent iterable wrapper retaining the original DataLoader API."""

    def __init__(self, dataloader, device, stream_api):
        self._dataloader = dataloader
        self._device = device
        self._stream_api = stream_api

    def __iter__(self):
        return _AsyncDevicePrefetchIterator(
            iter(self._dataloader), self._device, self._stream_api
        )

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


def _training_module():
    training = sys.modules.get("train_fsdp2") or sys.modules.get("__main__")
    if training is None or not hasattr(training, "DataLoader"):
        raise RuntimeError(
            "train_fsdp2.py must be active before installing DataLoader patches"
        )
    return training


def install_persistent_dataloader_workers() -> None:
    """Keep worker-local CPU caches alive across DataLoader iterators.

    This is intentionally independent from device prefetch: ``VideoCapture``
    handles and worker-local tokenizer/processor objects only survive epoch
    boundaries when DataLoader workers persist, while retaining a batch on the
    accelerator is unnecessary for that benefit.
    """
    if not flag("SPIRIT_PERSISTENT_DATALOADER_WORKERS", 1):
        return

    training = _training_module()
    if getattr(training, "_spirit_persistent_dataloader_workers_installed", False):
        return

    original_dataloader = training.DataLoader

    def persistent_dataloader(*args, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            kwargs.setdefault("persistent_workers", True)
        return original_dataloader(*args, **kwargs)

    training.DataLoader = persistent_dataloader
    training._spirit_persistent_dataloader_workers_installed = True
    if int(os.environ.get("RANK", "0")) == 0:
        print("[SPIRIT_PERSISTENT_DATALOADER_WORKERS] enabled=True", flush=True)


def _get_device_stream_api(local_rank: int):
    """Return the stream module matching the active accelerator backend."""
    musa = getattr(torch, "musa", None)
    if musa is not None and getattr(musa, "is_available", lambda: False)():
        if all(hasattr(musa, name) for name in ("Stream", "stream", "current_stream")):
            return torch.device("musa", local_rank), musa

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return torch.device("cuda", local_rank), cuda

    raise RuntimeError(
        "SPIRIT_ASYNC_H2D_PREFETCH=1 requires an available MUSA or CUDA stream API"
    )


def install_async_h2d_prefetch() -> None:
    """Optionally wrap the unchanged training module's DataLoader factory.

    The stock loop already requests ``pin_memory=True`` and transfers each
    top-level Tensor with ``non_blocking=True``.  This runtime-only wrapper
    moves that same transfer onto a separate stream one batch ahead.
    """
    install_persistent_dataloader_workers()

    # A one-batch device queue costs approximately one additional processed
    # batch of VRAM. Keep it opt-in until a background producer can prove a
    # throughput gain for this training loop.
    if not flag("SPIRIT_ASYNC_H2D_PREFETCH", 0):
        return

    training = _training_module()
    if getattr(training, "_spirit_async_h2d_prefetch_installed", False):
        return

    original_dataloader = training.DataLoader
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device, stream_api = _get_device_stream_api(local_rank)

    def prefetched_dataloader(*args, **kwargs):
        dataloader = original_dataloader(*args, **kwargs)
        return _AsyncDevicePrefetchDataLoader(dataloader, device, stream_api)

    training.DataLoader = prefetched_dataloader
    training._spirit_async_h2d_prefetch_installed = True
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[ASYNC_H2D_PREFETCH] "
            f"enabled=True device={device}",
            flush=True,
        )
