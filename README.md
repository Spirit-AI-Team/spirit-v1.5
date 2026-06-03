<div align="center">

# Spirit-v1.5 <br> <sub>A Robotic Foundation Model by Spirit AI</sub>

[![Project](https://img.shields.io/badge/Project-Page-blue?logo=homepage&logoColor=white)](https://www.spirit-ai.com/en/blog/spirit-v1-5) &ensp; [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5)
</div>


This repository contains the official implementation of the **Spirit-v1.5 VLA model**, as well as the runtime wrapper required to reproduce our results on the RoboChallenge benchmark. 
![image](assets/rc_results.png)
As of Jan 11, 2026, Spirit-v1.5 ranks **#1** on the [RoboChallenge](https://robochallenge.cn/home) Table30 benchmark.

## 📰 News
* **[2026-01]** Initial Release: Technical blog, inference code, base model checkpoint, and a fine-tuned checkpoint are now available.
* **[2026-4]** Fine-tuning code released!

## Directory Structure

```text
spirit-v1.5/
├── model/                        # Model architecture
│   ├── modeling_spirit_vla.py    # Main model architecture (Qwen3-VL backbone + DiT head + policy API)
│
├── dataset/                      # Dataset and data processing
│   ├── dataset.py                # Dataset implementation
│   └── transforms.py             # Data transformations
│
├── utils/                        # Utility functions
│   ├── checkpoint.py             # Checkpoint loading/saving utilities
│   ├── distributed.py            # Distributed training utilities
│   ├── logger.py                 # Logging utilities
│   ├── normalization.py          # Data normalization utilities
│   ├── sampling.py               # Sampling strategies
│   ├── tensor_ops.py             # Tensor operations
│   └── vlm_utils.py              # Vision-Language Model utilities
│
├── robochallenge/                # RoboChallenge integration
│   ├── run_robochallenge.py      # Python entrypoint
│   ├── runner/
│   │   ├── executor.py           # RoboChallengeExecutor (Checkpoint loading, inference, I/O)
│   │   └── task_info.py          # Task metadata (robot type, action type, prompts, etc.)
│   ├── robot/                    # Derived from open-source RoboChallengeInference
│   │   ├── interface_client.py   # RoboChallenge HTTP client
│   │   └── job_worker.py         # Job polling loop and execution flow
│   └── utils/                    # Derived from open-source RoboChallengeInference
│       ├── enums.py              # Shared enums/constants
│       ├── log.py                # Logging helpers
│       └── util.py               # Misc utilities
│
├── scripts/                      # Execution scripts
│   ├── run_robochallenge.sh      # RoboChallenge runtime launcher
│   └── run_finetune.sh           # Training launcher
│
├── train.py                      # Main training script
├── requirements-base.txt         # Core dependencies (inference only)
├── requirements-train.txt        # Additional training dependencies
├── requirements.txt              # Complete dependencies (base + training)
├── pyproject.toml                # Project configuration with optional dependencies
└── README.md                     # This file
```

### Installation & Setup

### System Requirements
- **Hardware**: Tested on NVIDIA A100 80GB GPU.
- **Software**: Python 3.10+.

### Backend-specific virtual environments

PyTorch backend packages (`torch` / `torchvision`) are mutually exclusive between CUDA and ROCm in a single venv. To support both, use separate environments:

- `.venv-cuda` for CUDA PyTorch
- `.venv-rocm` for ROCm PyTorch

A helper script is provided:

```bash
./scripts/setup_env.sh cuda
./scripts/setup_env.sh rocm
./scripts/setup_env.sh cuda train   # optional training extras
./scripts/setup_env.sh rocm train   # optional training extras
```

Activate the environment you want:

```bash
source .venv-cuda/bin/activate
# or
source .venv-rocm/bin/activate
```

### Dependency Files
- `requirements-common.txt` - Shared dependencies for both backends
- `requirements-base.txt` - Inference dependencies without backend-specific torch wheels
- `requirements-train.txt` - Additional dependencies for training
- `requirements.txt` - Complete shared dependencies
- `pyproject.toml` - Project configuration with optional `[train]` extra

### Key Dependencies
**Shared (Inference):**
- `transformers==4.57.1` - Hugging Face transformers library
- `diffusers==0.35.2` - Diffusion models library
- `safetensors==0.5.3` - Safe tensor serialization
- `numpy==2.2.6` - Numerical computing
- `pillow==10.4.0` - Image processing
- `requests==2.32.5` - HTTP library
- `scipy==1.15.2` - Scientific computing

**Backend-specific:**
- `torch` / `torchvision` - Install from the CUDA or ROCm wheel index via `scripts/setup_env.sh`

**Training (Additional):**
- `opencv-python>=4.8.0` - Image processing for data augmentation
- `wandb>=0.16.0` - Experiment tracking and logging
- `tqdm>=4.66.0` - Progress bars
- `einops==0.8.1` - Tensor operations (required by flash-attn)
- `flash-attn==2.8.3` - Flash Attention 2 for efficient training

> **Note**: `flash-attn` requires matching CUDA and PyTorch versions. If installation fails, refer to the [flash-attn installation guide](https://github.com/Dao-AILab/flash-attention#installation-and-features). ROCm installs should omit `flash-attn` unless you know your environment supports it.

## Model Checkpoints
| Model | Type | Link |
|----------|-------------|-------------|
| Spirit-v1.5 | Base Model | [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-yellow)](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5) |
| Spirit-v1.5-move-objects-into-box | Fine-tuned Model| [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-yellow)](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box) |

## Run RoboChallenge
We provide a minimal launcher script located at `scripts/run_robochallenge.sh`.

### Environment Configuration
The script requires the following environment variables to be set. Note that `USED_CHUNK_SIZE` defaults to 60 if not specified; all other variables are mandatory.

| Variable | Description |
|----------|-------------|
| `TASK_NAME` | Must correspond to a task defined in `robochallenge/runner/task_info.py`. |
| `ROBOCHALLENGE_JOB_ID` | The unique ID for the job collection. |
| `USER_TOKEN` | Your authentication token. |
| `CKPT_PATH` | Directory containing the `model.safetensors` file. |
| `USED_CHUNK_SIZE` | Action chunk size (Default: 60). |

### Execution Example:

Below is an example for the RoboChallenge task `move_objects_into_box`.

```bash
cd /path/to/spirit_vla_repo

export TASK_NAME=move_objects_into_box
export ROBOCHALLENGE_JOB_ID=your_job_collection_id
export USER_TOKEN=your_user_token
# Download / reference checkpoint:
# https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box
export CKPT_PATH=/path/to/your_checkpoint_dir
export USED_CHUNK_SIZE=60

./scripts/run_robochallenge.sh
```

### Local dry-run

You can validate the checkpoint and run one synthetic local inference without connecting to RoboChallenge:

```bash
cd /path/to/spirit_vla_repo
python -m robochallenge.run_robochallenge \
  --single_task move_objects_into_box \
  --robochallenge_job_id dummy_job \
  --ckpt_path /path/to/your_checkpoint_dir \
  --user_token dummy \
  --used_chunk_size 60 \
  --dry_run \
  --dry_run_infer
```

## Finetune Guide

### Prerequisites
- Pretrained Spirit-v1.5 base model checkpoint
- Training dataset (see [Dataset](#dataset) section below)
- Multi-GPU setup (recommended: 8x A100 80GB)

### Training Script

The training script is located at `scripts/run_finetune.sh`. It uses PyTorch's `torchrun` for distributed training.

### Environment Variables

Set the following environment variables before training:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATA_ROOT` | Path to training dataset directory | Yes |
| `PRETRAINED_PATH` | Path to pretrained model checkpoint (must contain `model.safetensors` and `config.json`) | Yes |
| `OUTPUT_DIR` | Directory for saving checkpoints (default: `./outputs`) | No |
| `NUM_GPUS` | Number of GPUs to use (default: 8) | No |
| `BATCH_SIZE` | Training batch size per GPU (default: 32) | No |
| `MAX_TRAIN_STEPS` | Maximum training steps (default: 40000) | No |
| `LOG_INTERVAL` | Logging interval in steps (default: 25) | No |
| `SAVE_STEPS` | Checkpoint saving interval (default: 2500) | No |
| `NUM_WORKERS` | Number of data loading workers (default: 32) | No |
| `PREFETCH_FACTOR` | Data prefetch factor (default: 8) | No |
| `WANDB_MODE` | Weights & Biases logging mode (default: `disable`) | No |

### Training Example

```bash
cd /path/to/spirit-v1.5

# Set required environment variables
export DATA_ROOT=/path/to/your/training/dataset
export PRETRAINED_PATH=/path/to/pretrained/checkpoint
export OUTPUT_DIR=./outputs/my_finetuned_model

# Optional: Configure training parameters
export NUM_GPUS=8
export BATCH_SIZE=32
export MAX_TRAIN_STEPS=40000
export WANDB_MODE=online  # Enable W&B logging
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="my_wandb_api_key"

# Run training
./scripts/run_finetune.sh
```

### Training Output

Training outputs will be saved to the `OUTPUT_DIR`:
- **Checkpoints**: Saved every `SAVE_STEPS` steps
- **W&B Logs**: If enabled, metrics are logged to Weights & Biases

### Dataset

Download the training dataset from Hugging Face:
```bash
huggingface-cli download RoboChallenge/task_table30_move_objects_into_box \
  --repo-type dataset \
  --local-dir /path/to/your/dataset
```

Set `DATA_ROOT` to the downloaded dataset directory when running training.

### Distributed Training

The training script uses PyTorch's Fully Sharded Data Parallel (FSDP) for efficient multi-GPU training:
- Automatically shards model parameters across GPUs
- Reduces memory footprint for large models
- Supports gradient checkpointing for memory efficiency

## Intended Uses

Spirit-v1.5 is a Vision-Language-Action (VLA) model designed specifically for robotic control. The model accepts current observations and textual descriptions as input and generates the next action chunk for the robot to execute.

## Out-of-scope Uses

Our models are not specifically designed for any tasks or scenarios other than robotic manipulations. 

Developers should expect failures in generation results regarding the out-of-scope scenarios. 

Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case, and evaluate and mitigate for privacy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios.

## Bibtex
```bibtex
@article{spiritai2026spiritv15,
  author = {Spirit AI Team},
  title = {Spirit-v1.5: Clean Data Is the Enemy of Great Robot Foundation Models},
  journal = {Spirit AI Blog},
  year = {2026},
  note = {https://www.spirit-ai.com/en/blog/spirit-v1-5},
}
```

## Acknowledgments
This codebase borrows code from [openpi](https://github.com/Physical-Intelligence/openpi), [qwen-vl](https://github.com/QwenLM/Qwen-VL) and [RoboChallengeInference](https://github.com/RoboChallenge/RoboChallengeInference). We thank them for their efforts and innovations, which have made the development process more efficient and convenient.

Thank you to everyone who contributed their wisdom and efforts to this project.

## Contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us through `guojunliang AT spirit-ai.com` and `miaotianrun AT spirit-ai.com`.
