#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"


# Parameters (no defaults, except USED_CHUNK_SIZE)
: "${TASK_NAME:?Please set TASK_NAME}"
: "${ROBOCHALLENGE_JOB_ID:?Please set ROBOCHALLENGE_JOB_ID}"
: "${USER_TOKEN:?Please set USER_TOKEN}"
: "${CKPT_PATH:?Please set CKPT_PATH}"

USED_CHUNK_SIZE="${USED_CHUNK_SIZE:-60}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in CKPT_PATH: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in CKPT_PATH: ${CKPT_PATH}" >&2
  exit 1
fi

echo "[INFO] TASK_NAME=${TASK_NAME}"
echo "[INFO] ROBOCHALLENGE_JOB_ID=${ROBOCHALLENGE_JOB_ID}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[INFO] USED_CHUNK_SIZE=${USED_CHUNK_SIZE}"

python -m robochallenge.run_robochallenge \
  --single_task "${TASK_NAME}" \
  --robochallenge_job_id "${ROBOCHALLENGE_JOB_ID}" \
  --ckpt_path "${CKPT_PATH}" \
  --user_token "${USER_TOKEN}" \
  --used_chunk_size "${USED_CHUNK_SIZE}"
