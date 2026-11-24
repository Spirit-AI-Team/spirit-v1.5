import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure repo root is on sys.path when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robochallenge.runner.executor import RoboChallengeExecutor
from robochallenge.robot.interface_client import InterfaceClient
from robochallenge.robot.job_worker import job_loop
from robochallenge.runner.task_info import TASK_INFO


def control_robot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_task", type=str, required=True)
    parser.add_argument("--robochallenge_job_id", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--user_token", type=str, required=True)
    parser.add_argument("--used_chunk_size", type=int, default=60)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for inference, for example cuda or xpu (default: CUDA, then XPU, then CPU).",
    )
    cfg = parser.parse_args()

    executor = RoboChallengeExecutor(cfg)
    logger.info(
        "Task name=%s run_id=%s checkpoint loaded.",
        cfg.single_task,
        cfg.robochallenge_job_id,
    )
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
