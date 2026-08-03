#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/workspace/spirit-v1.5-musa_optimization/checkpoint_config}"
DATA_ROOT="${DATA_ROOT:-/workspace/spirit-v1.5-musa_optimization/fake_data/move_objects_into_box}"
CKPT_PATH="${PRETRAINED_PATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "[ERROR] PRETRAINED_PATH not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in PRETRAINED_PATH: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in PRETRAINED_PATH: ${CKPT_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NUM_GPUS:-8}" \
    "${REPO_ROOT}/train.py" \
    --data_root "${DATA_ROOT:?DATA_ROOT must be set}" \
    --pretrained_path "${PRETRAINED_PATH:?PRETRAINED_PATH must be set}" \
    --output_dir "${OUTPUT_DIR:-${REPO_ROOT}/outputs}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --max_train_steps "${MAX_TRAIN_STEPS:-60000}" \
    --log_interval "${LOG_INTERVAL:-25}" \
    --save_steps "${SAVE_STEPS:-2500}" \
    --num_workers "${NUM_WORKERS:-32}" \
    --prefetch_factor "${PREFETCH_FACTOR:-8}" \
    --wandb_mode "${WANDB_MODE:-disabled}"
