"""Runtime-only dataset behavior patches for the unchanged training source."""

import os
from collections import OrderedDict

from ..config import flag, int_env


def install_video_capture_cache() -> None:
    """Reuse OpenCV decoders within each DataLoader worker.

    The stock dataset opens and releases a ``cv2.VideoCapture`` for every
    sample/camera read.  A DataLoader worker owns its dataset copy, so a cache
    attached to that copy is process-local and never shares decoder state
    between workers.
    """
    if not flag("SPIRIT_VIDEO_CAPTURE_CACHE", 1):
        return

    cache_size = int_env("SPIRIT_VIDEO_CAPTURE_CACHE_SIZE", 64, minimum=1)

    from dataset import RoboChallengeDataset
    import cv2
    import torch

    if getattr(RoboChallengeDataset, "_spirit_video_capture_cache_installed", False):
        return

    original_init = RoboChallengeDataset.__init__

    def patched_init(self, config):
        original_init(self, config)
        self._spirit_video_captures = OrderedDict()

    def open_capture(self, path):
        path_key = str(path)
        captures = self._spirit_video_captures
        capture = captures.get(path_key)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(path_key)
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"无法打开视频: {path}")
            captures[path_key] = capture
            while len(captures) > cache_size:
                _, evicted_capture = captures.popitem(last=False)
                evicted_capture.release()
        else:
            captures.move_to_end(path_key)
        return capture

    def cached_decode_video_frame(self, video_path, frame_idx):
        capture = open_capture(self, video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            # Retry once with a fresh decoder. This preserves the original
            # failure semantics while recovering from a stale seek state.
            path_key = str(video_path)
            capture.release()
            self._spirit_video_captures.pop(path_key, None)
            capture = open_capture(self, video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
        if not success:
            raise ValueError(f"Frame {frame_idx} 读取失败 {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        return frame_tensor.float() / 255.0

    RoboChallengeDataset.__init__ = patched_init
    RoboChallengeDataset._decode_video_frame = cached_decode_video_frame
    RoboChallengeDataset._spirit_video_capture_cache_installed = True
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[SPIRIT_VIDEO_CAPTURE_CACHE] "
            "enabled=True scope=per_dataloader_worker "
            f"capacity={cache_size} retry=fresh_decoder_once",
            flush=True,
        )


def install_dataset_repeat() -> None:
    """Apply ``SPIRIT_DATASET_REPEAT`` without editing dataset or train code."""
    from dataset import RoboChallengeDataset

    if getattr(RoboChallengeDataset, "_spirit_repeat_installed", False):
        return

    original_init = RoboChallengeDataset.__init__
    original_len = RoboChallengeDataset.__len__
    original_getitem = RoboChallengeDataset.__getitem__
    original_get_lowdim_item = RoboChallengeDataset.get_lowdim_item

    def patched_init(self, config):
        original_init(self, config)
        repeat = int(os.environ.get("SPIRIT_DATASET_REPEAT", "1"))
        if repeat < 1:
            raise ValueError(f"SPIRIT_DATASET_REPEAT must be >= 1, got {repeat}")
        self._spirit_dataset_repeat = repeat
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[SPIRIT_DATASET_REPEAT] "
                f"base_samples={original_len(self)} repeat={repeat} "
                f"effective_samples={original_len(self) * repeat}",
                flush=True,
            )

    def patched_len(self):
        return original_len(self) * getattr(self, "_spirit_dataset_repeat", 1)

    def patched_getitem(self, index):
        return original_getitem(self, index % original_len(self))

    def patched_get_lowdim_item(self, index):
        return original_get_lowdim_item(self, index % original_len(self))

    RoboChallengeDataset.__init__ = patched_init
    RoboChallengeDataset.__len__ = patched_len
    RoboChallengeDataset.__getitem__ = patched_getitem
    RoboChallengeDataset.get_lowdim_item = patched_get_lowdim_item
    RoboChallengeDataset._spirit_repeat_installed = True
