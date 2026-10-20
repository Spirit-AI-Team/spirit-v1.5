import json
import logging
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchvision.utils import save_image
from typing import Optional

from model.modeling_spirit_vla import SpiritVLAPolicy
from utils.device import resolve_device
from .task_info import TASK_INFO, TASKS_USE_LESS_CHUNK_SIZE, TASTS_APPLY_GRIPPER_BINARIZATION

logger = logging.getLogger(__name__)


def _img_byte_to_tensor(img_byte, target_size_for_PIL=(320, 240)):
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(img_byte)).convert("RGB").resize(target_size_for_PIL, Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _post_process_action(action_np, state_np, robot_type, used_chunk_size, raw_embodiment_stats, binarization_threshold: Optional[float] = None):
    result_list = []
    if raw_embodiment_stats is not None:
        left_gripper_min, left_gripper_max = (
            raw_embodiment_stats[robot_type]["action"]["min"][6],
            raw_embodiment_stats[robot_type]["action"]["max"][6],
        )
        right_gripper_min, right_gripper_max = (
            raw_embodiment_stats[robot_type]["action"]["min"][13],
            raw_embodiment_stats[robot_type]["action"]["max"][13],
        )
    eps = 1e-8

    for i in range(min(action_np.shape[0], used_chunk_size)):
        action_i = action_np[i]

        if robot_type == "ARX5":
            target_xyz = action_i[:3] + state_np[:3]
            taget_rot = (Rotation.from_rotvec(action_i[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_euler = Rotation.from_rotvec(taget_rot).as_euler("xyz", degrees=False)
            target_gripper = action_i[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min

            list_i = target_xyz.tolist() + target_euler.tolist() + [target_gripper]
        elif robot_type == "UR5":
            target_joint = action_i[:6] + state_np[:6]
            target_gripper = 0.1 - action_i[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min
            else:
                target_gripper = target_gripper / 0.1 * 255

            list_i = target_joint.tolist() + [target_gripper]
        elif robot_type == "Franka":
            target_xyz = action_i[:3] + state_np[:3]
            taget_rot = (Rotation.from_rotvec(action_i[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_quat = Rotation.from_rotvec(taget_rot).as_quat()
            target_gripper = action_i[6].item()
            if raw_embodiment_stats is not None:
                target_gripper = target_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min

            list_i = target_xyz.tolist() + target_quat.tolist() + [target_gripper]
        elif robot_type == "aloha":
            target_left_xyz = action_i[:3] + state_np[:3]
            target_left_rot = (Rotation.from_rotvec(action_i[3:6]) * Rotation.from_rotvec(state_np[3:6])).as_rotvec()
            target_left_euler = Rotation.from_rotvec(target_left_rot).as_quat()
            target_left_gripper = action_i[6].item()
            if raw_embodiment_stats is not None:
                target_left_gripper = (
                    target_left_gripper / 0.1 * (left_gripper_max - left_gripper_min + eps) + left_gripper_min
                )

            target_right_xyz = action_i[7:10] + state_np[7:10]
            target_right_rot = (
                Rotation.from_rotvec(action_i[10:13]) * Rotation.from_rotvec(state_np[10:13])
            ).as_rotvec()
            target_right_euler = Rotation.from_rotvec(target_right_rot).as_quat()
            target_right_gripper = action_i[13].item()
            if raw_embodiment_stats is not None:
                target_right_gripper = (
                    target_right_gripper / 0.1 * (right_gripper_max - right_gripper_min + eps) + right_gripper_min
                )

            list_i = (
                target_left_xyz.tolist()
                + target_left_euler.tolist()
                + [target_left_gripper]
                + target_right_xyz.tolist()
                + target_right_euler.tolist()
                + [target_right_gripper]
            )
            assert len(list_i) == 16
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        if binarization_threshold is not None:
            if list_i[7] < binarization_threshold:
                list_i[7] = 0.0
            if len(list_i) == 16 and list_i[15] < binarization_threshold:
                list_i[15] = 0.0
        result_list.append(list_i)

    return result_list


def _prepare_batch(item_tmp, task_name, img_save_path, save_idx, raw_embodiment_stats, used_embodiment_stats=None, device=None):
    device = resolve_device(device)
    robot_type = TASK_INFO[task_name]["robot_type"]
    item = {}
    if raw_embodiment_stats is not None:
        left_gripper_min, left_gripper_max = (
            raw_embodiment_stats[robot_type]["observation.state"]["min"][6],
            raw_embodiment_stats[robot_type]["observation.state"]["max"][6],
        )
        right_gripper_min, right_gripper_max = (
            raw_embodiment_stats[robot_type]["observation.state"]["min"][13],
            raw_embodiment_stats[robot_type]["observation.state"]["max"][13],
        )
    eps = 1e-8

    item["task"] = [TASK_INFO[task_name]["task"]]

    item["observation.state"] = torch.zeros(14)
    state_tmp = item_tmp["action"]
    if robot_type == "ARX5":
        item["observation.state"][:3] = torch.tensor(state_tmp[:3], dtype=torch.float32)
        item["observation.state"][3:6] = torch.tensor(
            Rotation.from_euler("xyz", state_tmp[3:6], degrees=False).as_rotvec(), dtype=torch.float32
        )
        if raw_embodiment_stats is not None:
            gripper = (state_tmp[6] - left_gripper_min) / (left_gripper_max - left_gripper_min + eps) * 0.1
        else:
            gripper = state_tmp[6]
        item["observation.state"][6] = torch.tensor(gripper, dtype=torch.float32)
    elif robot_type == "UR5":
        assert len(state_tmp) == 7
        item["observation.state"][:6] = torch.tensor(state_tmp[:6], dtype=torch.float32)
        assert state_tmp[6] > 1
        if raw_embodiment_stats is not None:
            gripper = 0.1 - (state_tmp[6] - left_gripper_min) / (left_gripper_max - left_gripper_min + eps) * 0.1
        else:
            gripper = 0.1 - state_tmp[6] / 255 * 0.1
        item["observation.state"][6] = torch.tensor(gripper, dtype=torch.float32)
    elif robot_type == "Franka":
        assert len(state_tmp) == 8
        item["observation.state"][:3] = torch.tensor(state_tmp[:3], dtype=torch.float32)
        quater = state_tmp[3:7]
        item["observation.state"][3:6] = torch.tensor(Rotation.from_quat(quater).as_rotvec(), dtype=torch.float32)
        if raw_embodiment_stats is not None:
            gripper = (state_tmp[7] - left_gripper_min) / (left_gripper_max - left_gripper_min + eps) * 0.1
        else:
            gripper = state_tmp[7]
        item["observation.state"][6] = torch.tensor(gripper, dtype=torch.float32)
    elif robot_type == "aloha":
        assert len(state_tmp) == 16, "Expected 16-dim state for aloha (quat-based)."
        item["observation.state"][:3] = torch.tensor(state_tmp[:3], dtype=torch.float32)
        quater = state_tmp[3:7]
        item["observation.state"][3:6] = torch.tensor(Rotation.from_quat(quater).as_rotvec(), dtype=torch.float32)
        if raw_embodiment_stats is not None:
            gripper = (state_tmp[7] - left_gripper_min) / (left_gripper_max - left_gripper_min + eps) * 0.1
        else:
            gripper = state_tmp[7]
        item["observation.state"][6] = torch.tensor(gripper, dtype=torch.float32)

        item["observation.state"][7:10] = torch.tensor(state_tmp[8:11], dtype=torch.float32)
        quater = state_tmp[11:15]
        item["observation.state"][10:13] = torch.tensor(Rotation.from_quat(quater).as_rotvec(), dtype=torch.float32)
        if raw_embodiment_stats is not None:
            gripper = (state_tmp[15] - right_gripper_min) / (right_gripper_max - right_gripper_min + eps) * 0.1
        else:
            gripper = state_tmp[15]
        item["observation.state"][13] = torch.tensor(gripper, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    item["observation.state.before_norm"] = item["observation.state"].clone()
    item["normalized_in_getitem"] = torch.tensor([False])

    item["observation.state"] = item["observation.state"].unsqueeze(0).to(device)

    img_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    for key in img_keys:
        if TASK_INFO[task_name][key] not in item_tmp["images"]:
            assert robot_type == "UR5" and key == "observation.images.cam_right_wrist"
            item[key] = torch.zeros_like(item["observation.images.cam_high"])
            continue
        img = _img_byte_to_tensor(item_tmp["images"][TASK_INFO[task_name][key]])
        save_image(img, img_save_path / f"{key.replace('.', '_')}_{save_idx}.png")
        item[key] = img.unsqueeze(0).to(device)

    item["batch_source"] = "rb"
    item["robot_type"] = [robot_type]

    return item


class RoboChallengeExecutor:
    def __init__(self, cfg):
        task_name = cfg.single_task
        run_id = cfg.robochallenge_job_id
        used_chunk_size = cfg.used_chunk_size
        if task_name in TASKS_USE_LESS_CHUNK_SIZE:
            used_chunk_size = 40
            logger.info("Task %s uses smaller chunk size: %s", task_name, used_chunk_size)
        binarization_threshold = TASTS_APPLY_GRIPPER_BINARIZATION.get(task_name)
        ckpt_path = cfg.ckpt_path
        self.policy = SpiritVLAPolicy.from_pretrained(ckpt_path)
        self.device = resolve_device(getattr(cfg, "device", None))
        self.policy.to(self.device)
        self.policy.eval()
        self.task_name = task_name
        self.run_id = run_id
        self.used_chunk_size = used_chunk_size
        self.binarization_threshold = binarization_threshold

        raw_stats_path = getattr(cfg, "raw_embodiment_stats_json_path", None)
        self.raw_embodiment_stats = None
        if raw_stats_path:
            with open(raw_stats_path, "r") as f:
                self.raw_embodiment_stats = json.load(f)
        self.use_embodiment_stats = bool(getattr(cfg, "use_embodiment_specific_norm", False))
        self.used_embodiment_stats = None
        if self.use_embodiment_stats:
            emb_stats_path = getattr(cfg, "embodiment_stats_path", None)
            if emb_stats_path is None:
                raise ValueError("use_embodiment_specific_norm=True requires cfg.embodiment_stats_path")
            with open(emb_stats_path, "r") as f:
                self.used_embodiment_stats = json.load(f)
        logger.info(
            "raw_embodiment_stats=%s use_embodiment_specific_norm=%s",
            "on" if self.raw_embodiment_stats is not None else "off",
            self.use_embodiment_stats,
        )

        self.out_base_path = Path(f"output/{self.task_name}/{self.run_id}")
        os.makedirs(self.out_base_path, exist_ok=True)
        self.execution_idx = 0
        self.img_save_path: Optional[Path] = None
        self.pkl_save_path: Optional[Path] = None
        self.save_idx = 0

    def _start_new_execution(self, job_id):
        while any(self.out_base_path.glob(f"execution{self.execution_idx}_*")):
            self.execution_idx += 1
        execution_path = self.out_base_path / f"execution{self.execution_idx}_{job_id}"
        os.makedirs(execution_path, exist_ok=True)
        self.img_save_path = execution_path / "images"
        self.pkl_save_path = execution_path / "data"
        os.makedirs(self.img_save_path, exist_ok=True)
        os.makedirs(self.pkl_save_path, exist_ok=True)
        self.save_idx = 0
        logger.info("Start new execution: idx=%s path=%s", self.execution_idx, execution_path)
        self.execution_idx += 1

    def infer(self, item_tmp, new_execution=False):
        if new_execution or self.img_save_path is None:
            self._start_new_execution(item_tmp["job_id"])

        item = _prepare_batch(
            item_tmp,
            self.task_name,
            self.img_save_path,
            self.save_idx,
            self.raw_embodiment_stats,
            self.used_embodiment_stats,
            self.device,
        )

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type in ("cuda", "xpu")
            else nullcontext(),
        ):
            action_tensor = self.policy.select_action(item).cpu()

        action_list = _post_process_action(
            action_tensor.squeeze(0).cpu().numpy(),
            item["observation.state.before_norm"].numpy(),
            TASK_INFO[self.task_name]["robot_type"],
            self.used_chunk_size,
            self.raw_embodiment_stats,
            self.binarization_threshold,
        )

        item_tmp["our_infer_action"] = action_list
        with open(self.pkl_save_path / f"data_{self.save_idx}.pkl", "wb") as f:
            pickle.dump(item_tmp, f)
        self.save_idx += 1

        return action_list

