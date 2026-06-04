import argparse
import io
import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Ensure repo root is on sys.path when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robochallenge.runner.executor import RoboChallengeExecutor
from robochallenge.robot.interface_client import InterfaceClient
from robochallenge.robot.job_worker import job_loop
from robochallenge.runner.task_info import TASK_INFO


def _blank_png_bytes(width=320, height=240, color=(0, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _make_dry_run_item(task_name):
    robot_type = TASK_INFO[task_name]["robot_type"]
    task = TASK_INFO[task_name]["task"]
    item = {
        "job_id": "dry-run",
        "task": task,
        "images": {},
    }

    if robot_type == "ARX5":
        item["action"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif robot_type == "UR5":
        item["action"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0]
    elif robot_type == "Franka":
        item["action"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    elif robot_type == "aloha":
        item["action"] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ]
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    for image_key in {
        TASK_INFO[task_name]["observation.images.cam_high"],
        TASK_INFO[task_name]["observation.images.cam_left_wrist"],
        TASK_INFO[task_name]["observation.images.cam_right_wrist"],
    }:
        if image_key:
            item["images"][image_key] = _blank_png_bytes()

    return item


def control_robot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_task", type=str, required=True)
    parser.add_argument("--robochallenge_job_id", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--user_token", type=str, required=True)
    parser.add_argument("--used_chunk_size", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true", help="Load the checkpoint and exit without connecting to RoboChallenge")
    parser.add_argument("--dry_run_infer", action="store_true", help="Run one local synthetic inference after loading the checkpoint")
    cfg = parser.parse_args()

    executor = RoboChallengeExecutor(cfg)
    logger.info(
        "Task name=%s run_id=%s checkpoint loaded.",
        cfg.single_task,
        cfg.robochallenge_job_id,
    )

    if cfg.dry_run:
        logger.info("Dry-run mode enabled: skipping RoboChallenge network loop.")
        if cfg.dry_run_infer:
            sample = _make_dry_run_item(cfg.single_task)
            result = executor.infer(sample)
            logger.info("Dry-run inference succeeded; produced %d action steps.", len(result))
        return

    logger.info("Waiting RC to prepare the task and send observation...")

    client = InterfaceClient(cfg.user_token)
    job_loop(
        client,
        executor,
        cfg.robochallenge_job_id,
        image_size=[320, 240],
        image_type=["high", "left_hand", "right_hand"] if TASK_INFO[cfg.single_task]["robot_type"] != "UR5" else ["left_hand", "right_hand"],
        action_type=TASK_INFO[cfg.single_task]["action_type"],
        duration=1 / 15,
    )


if __name__ == "__main__":
    control_robot()
