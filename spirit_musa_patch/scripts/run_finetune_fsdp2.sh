#!/usr/bin/env bash
# Integer configuration contract (see fsdp2_runtime/config.py):
# 0=off/reference/coarse/varlen/views; 1=on/MUSA-fused/block/dense/packed.
# Performance defaults below are the validated production path.  Override a
# value only for an explicit A/B or rollback; paths remain numerically guarded.
export SPIRIT_VISION_GEOM_CACHE="${SPIRIT_VISION_GEOM_CACHE:-1}"       # 0=off, 1=cache geometry
export SPIRIT_TEXT_POS_CACHE="${SPIRIT_TEXT_POS_CACHE:-1}"             # 0=off, 1=cache MRoPE positions
export SPIRIT_PATCH_EMBED_IMPL="${SPIRIT_PATCH_EMBED_IMPL:-1}"         # 0=conv3d, 1=GEMM
export SPIRIT_MUSA_BATCHED_PREPROCESS="${SPIRIT_MUSA_BATCHED_PREPROCESS:-1}" # 0=legacy PIL, 1=MUSA batch
export SPIRIT_QWEN_SKIP_LM_HEAD="${SPIRIT_QWEN_SKIP_LM_HEAD:-1}"       # 0=compute logits, 1=skip unused LM head
export MASTER_PORT="${MASTER_PORT:-29935}"

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PATCH_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$PATCH_ROOT/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/fake_data/move_objects_into_box}"
export SPIRIT_PRETRAINED_PATH="${SPIRIT_PRETRAINED_PATH:-$REPO_ROOT/checkpoint_config}"
export SPIRIT_OUTPUT_DIR="${SPIRIT_OUTPUT_DIR:-$REPO_ROOT/outputs_fsdp2}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset directory not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$SPIRIT_PRETRAINED_PATH/config.json" || ! -f "$SPIRIT_PRETRAINED_PATH/model.safetensors" ]]; then
    echo "Checkpoint must contain config.json and model.safetensors: $SPIRIT_PRETRAINED_PATH" >&2
    exit 1
fi
HOST=$(hostname)
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29935}"
EXP="${EXP:-$REPO_ROOT}"
MCCL_HCA="${MCCL_HCA:-}"
MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-}"
export PEAK_TFLOPS_PER_GPU="${PEAK_TFLOPS_PER_GPU:-500}"
export SPIRIT_MFU_SEQ_LEN="${SPIRIT_MFU_SEQ_LEN:-295}"
export SPIRIT_GRAD_ACCUM_STEPS="${SPIRIT_GRAD_ACCUM_STEPS:-4}"
export SPIRIT_DATASET_REPEAT="${SPIRIT_DATASET_REPEAT:-10}"
export SPIRIT_GA_SYNC_MODE="${SPIRIT_GA_SYNC_MODE:-1}"                    # 0=no_sync, 1=all_reduce
export SPIRIT_QWEN_TEXT_RECOMPUTE_LAYERS="${SPIRIT_QWEN_TEXT_RECOMPUTE_LAYERS:-36}"
export SPIRIT_QWEN_VISION_RECOMPUTE_LAYERS="${SPIRIT_QWEN_VISION_RECOMPUTE_LAYERS:-24}"
export SPIRIT_BATCH_SIZE="${SPIRIT_BATCH_SIZE:-320}"
export SPIRIT_MAX_TRAIN_STEPS="${SPIRIT_MAX_TRAIN_STEPS:-50}"

if [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR="127.0.0.1"
fi
if [[ "$NNODES" -gt 1 && "$MASTER_ADDR" == "127.0.0.1" ]]; then
    echo "MASTER_ADDR must be set to the rank-0 address for multi-node runs" >&2
    exit 1
fi

export SPIRIT_EXP_ROOT="$EXP"
export SPIRIT_REPO_ROOT="${SPIRIT_REPO_ROOT:-$REPO_ROOT}"
export SPIRIT_PATCH_ROOT="${SPIRIT_PATCH_ROOT:-$PATCH_ROOT}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export SPIRIT_FSDP2_MODE="${SPIRIT_FSDP2_MODE:-0}"                     # 0=auto, 1=FSDP, 2=HSDP
export SPIRIT_FSDP2_SHARD_SIZE="${SPIRIT_FSDP2_SHARD_SIZE:-$NPROC_PER_NODE}"
export SPIRIT_FSDP2_WRAP_LEVEL="${SPIRIT_FSDP2_WRAP_LEVEL:-1}"          # 0=coarse, 1=block
export SPIRIT_OPTIMIZER="${SPIRIT_OPTIMIZER:-1}"                       # 0=AdamW, 1=fused AdamW
export SPIRIT_TORCH_COMPILE="${SPIRIT_TORCH_COMPILE:-0}"              # 0=disabled, 1=experimental compile
export SPIRIT_QWEN_RMS_NORM_IMPL="${SPIRIT_QWEN_RMS_NORM_IMPL:-1}"     # 0=reference, 1=MUSA fused
export SPIRIT_QWEN_TEXT_ROPE_IMPL="${SPIRIT_QWEN_TEXT_ROPE_IMPL:-1}"   # 0=reference, 1=MUSA fused
export SPIRIT_QWEN_TEXT_SWIGLU_IMPL="${SPIRIT_QWEN_TEXT_SWIGLU_IMPL:-1}" # 0=reference, 1=packed/fused
# Optional directory or training_state*.pt file for optimizer/scheduler resume.
export SPIRIT_RESUME_TRAINING_STATE="${SPIRIT_RESUME_TRAINING_STATE:-}"
# HSDP foreach clipping passed correctness and distributed validation; 0 is the
# immediate rollback path for a reference comparison.
export SPIRIT_GRAD_CLIP_IMPL="${SPIRIT_GRAD_CLIP_IMPL:-1}"              # 0=reference, 1=HSDP foreach
export SPIRIT_QWEN_VISION_ATTN_LAYOUT="${SPIRIT_QWEN_VISION_ATTN_LAYOUT:-1}" # 0=varlen, 1=dense fixed-grid
export SPIRIT_QWEN_VISION_ROPE_IMPL="${SPIRIT_QWEN_VISION_ROPE_IMPL:-1}" # 0=reference, 1=MUSA fused
export SPIRIT_QWEN_VISION_QKV_LAYOUT="${SPIRIT_QWEN_VISION_QKV_LAYOUT:-0}" # 0=views, 1=packed contiguous
export SPIRIT_QWEN_VISION_FLAT_INPUTS="${SPIRIT_QWEN_VISION_FLAT_INPUTS:-1}"
# Opt-in framework ranges for attribution-only profiling. Keep disabled for
# normal throughput runs so record_function adds no hot-path overhead.
export SPIRIT_PROFILE_RANGES="${SPIRIT_PROFILE_RANGES:-0}"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export SPIRIT_IMAGE_HISTORY="${SPIRIT_IMAGE_HISTORY:-1}"               # integer history length
export SPIRIT_IMAGE_HISTORY_STRIDE="${SPIRIT_IMAGE_HISTORY_STRIDE:-1}" # integer frame stride
export SPIRIT_TRAINABLE_SCOPE=full

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export TORCH_MUSA_FSDP2_COMM_TYPE=1
export FSDP_EXPLICIT_PREFETCH_LAYER_NUM="${FSDP_EXPLICIT_PREFETCH_LAYER_NUM:-10}"

export MCCL_DEBUG=WARN
export MCCL_IB_DISABLE=0
export MCCL_IB_GID_INDEX=3
export MCCL_CROSS_NIC=0
export MCCL_PROTOS=2
export MCCL_NET_SHARED_BUFFERS=0
export IMCCL_MUMEM_HOST_ENABLE=0
if [[ -n "$MCCL_SOCKET_IFNAME" ]]; then
    export MCCL_SOCKET_IFNAME
fi
if [[ -n "$MCCL_HCA" ]]; then
    export MCCL_IB_HCA="$MCCL_HCA"
fi

mkdir -p "$EXP/logs"
LOG="$EXP/logs/dual16_hsdp_skip_lm_head_formal50_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"

echo "host=$HOST"
echo "node_rank=$NODE_RANK"
echo "nnodes=$NNODES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "fsdp2_mode=$SPIRIT_FSDP2_MODE shard_size=$SPIRIT_FSDP2_SHARD_SIZE"
echo "fsdp2_wrap_level=$SPIRIT_FSDP2_WRAP_LEVEL"
echo "optimizer=$SPIRIT_OPTIMIZER"
echo "torch_compile=$SPIRIT_TORCH_COMPILE mode=default"
echo "qwen_rms_norm_impl=$SPIRIT_QWEN_RMS_NORM_IMPL"
echo "qwen_text_rope_impl=$SPIRIT_QWEN_TEXT_ROPE_IMPL"
echo "qwen_text_swiglu_impl=$SPIRIT_QWEN_TEXT_SWIGLU_IMPL"
echo "resume_training_state=${SPIRIT_RESUME_TRAINING_STATE:-none}"
echo "grad_clip_impl=$SPIRIT_GRAD_CLIP_IMPL"
echo "qwen_vision_attn_layout=$SPIRIT_QWEN_VISION_ATTN_LAYOUT"
echo "qwen_vision_rope_impl=$SPIRIT_QWEN_VISION_ROPE_IMPL"
echo "qwen_vision_qkv_layout=$SPIRIT_QWEN_VISION_QKV_LAYOUT"
echo "profile_ranges=$SPIRIT_PROFILE_RANGES"
echo "qwen_vision_flat_inputs=$SPIRIT_QWEN_VISION_FLAT_INPUTS"
echo "fsdp_explicit_prefetch_layer_num=$FSDP_EXPLICIT_PREFETCH_LAYER_NUM"
echo "gradient_accumulation_steps=$SPIRIT_GRAD_ACCUM_STEPS"
echo "dataset_repeat=$SPIRIT_DATASET_REPEAT"
echo "ga_sync_mode=$SPIRIT_GA_SYNC_MODE"
echo "qwen_text_recompute_layers=$SPIRIT_QWEN_TEXT_RECOMPUTE_LAYERS"
echo "qwen_vision_recompute_layers=$SPIRIT_QWEN_VISION_RECOMPUTE_LAYERS"
echo "repo_root=$SPIRIT_REPO_ROOT"
echo "patch_root=$SPIRIT_PATCH_ROOT"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "hca=${MCCL_HCA:-none}"
echo "log=$LOG"
echo "data_root=$DATA_ROOT"
echo "training_steps=$SPIRIT_MAX_TRAIN_STEPS"

cd "$REPO_ROOT"

set +e
PYTHONPATH="$PATCH_ROOT:$REPO_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PATCH_ROOT/train_fsdp2.py" \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
set -e

echo "docker_exit=$STATUS"
echo "log=$LOG"
exit "$STATUS"
