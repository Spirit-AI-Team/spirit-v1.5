#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
if [[ -z "$BACKEND" ]]; then
  echo "Usage: $0 <cuda|rocm>" >&2
  exit 1
fi

case "$BACKEND" in
  cuda)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    VENV_DIR=".venv-cuda"
    ;;
  rocm)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm7.2"
    VENV_DIR=".venv-rocm"
    ;;
  *)
    echo "Unsupported backend: $BACKEND (expected cuda or rocm)" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR"
fi

uv pip install --python "$VENV_DIR/bin/python" -r requirements-common.txt
uv pip install --python "$VENV_DIR/bin/python" torch torchvision --index-url "$TORCH_INDEX_URL"

if [[ "${2:-}" == "train" ]]; then
  if [[ "$BACKEND" == "cuda" ]]; then
    uv pip install --python "$VENV_DIR/bin/python" -r requirements-train.txt
  else
    uv pip install --python "$VENV_DIR/bin/python" \
      "opencv-python>=4.8.0" \
      "wandb>=0.16.0" \
      "tqdm>=4.66.0" \
      "einops==0.8.1"
    echo "Skipping flash-attn on ROCm backend."
  fi
fi

echo "Done. Activate with: source $VENV_DIR/bin/activate"
