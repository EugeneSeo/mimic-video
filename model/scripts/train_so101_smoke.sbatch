#!/bin/bash
#SBATCH --job-name=mimic_so101_smoke
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80G
#SBATCH --output=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.out
#SBATCH --error=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.err

set -euo pipefail

# Euler compute nodes need this if uv/Hugging Face/checkpoint downloads are triggered.
module load eth_proxy

# transformer_engine shells out to `ldconfig`; on Euler this may live outside the default PATH.
export PATH="/usr/sbin:/sbin:$PATH"

################################################################################
# mimic-video SO-101 action-decoder finetuning smoke test
################################################################################
# Required:
#   DATA_DIR=/path/to/converted/so101_zarr
#
# Typical submit:
#   DATA_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke \
#     sbatch model/scripts/train_so101_smoke.sbatch
#
# Optional knobs:
#   INIT_NAME=random | partial_bridge_init
#   VIDEO_CKPT=v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused | v2w_pretrained_cosmos
#   LR_TAG=1.000e-04 | 3.000e-04
#   GLOBAL_BATCH_SIZE=4 | 8 | 16 | 32 | 64
#   GRAD_ACCUM_ITER=8 | 32  # optional; empty uses experiment default
#   MAX_ITER=50
#   RUN_VALIDATION=false
#   WANDB_ENABLED=1 | 0
#   WANDB_MODE=online | offline | disabled
#   WANDB_PROJECT=mimic-video-so101
#   WANDB_ENTITY=dreamdifferent
#   WANDB_LOG_EVERY_N=1
#   NUM_WORKERS=1
#   MIMIC_DATASET_STATS_NUM_WORKERS=0
#   MIMIC_DATASET_STATS_BATCH_SIZE=4
################################################################################

REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/eugseo/workspace/mimic-video}"
MODEL_DIR="$REPO_ROOT/model"
LOG_DIR="${LOG_DIR:-/cluster/scratch/eugseo/mimic_video_logs}"
RUNS_DIR="${RUNS_DIR:-/cluster/scratch/eugseo/mimic_video_runs}"
WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-/cluster/scratch/eugseo/mimic_video_wandb}"

VENV_NVIDIA_DIR="$MODEL_DIR/.venv/lib/python3.10/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_DIR" ]; then
  # transformer_engine searches CUDA_HOME recursively for libnvrtc/libcudnn.
  # imaginaire distributed init also dlopens libcudart.so by soname.
  export CUDA_HOME="${CUDA_HOME:-$VENV_NVIDIA_DIR}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export LD_LIBRARY_PATH="$VENV_NVIDIA_DIR/cuda_runtime/lib:$VENV_NVIDIA_DIR/cuda_nvrtc/lib:$VENV_NVIDIA_DIR/cudnn/lib:$VENV_NVIDIA_DIR/nccl/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${DATA_DIR:-}" ]; then
  echo "ERROR: DATA_DIR must point to the converted SO-101 zarr directory."
  echo "Example:"
  echo "  DATA_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke sbatch model/scripts/train_so101_smoke.sbatch"
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
  exit 1
fi

if ! find "$DATA_DIR" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "ERROR: DATA_DIR contains no .zarr episodes: $DATA_DIR"
  exit 1
fi

INIT_NAME="${INIT_NAME:-random}"
VIDEO_CKPT="${VIDEO_CKPT:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused}"
LR_TAG="${LR_TAG:-1.000e-04}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-}"
MAX_ITER="${MAX_ITER:-50}"
LOGGING_ITER="${LOGGING_ITER:-10}"
SAVE_ITER="${SAVE_ITER:-50}"
VALIDATION_ITER="${VALIDATION_ITER:-50}"
RUN_VALIDATION="${RUN_VALIDATION:-false}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MIMIC_DATASET_STATS_NUM_WORKERS="${MIMIC_DATASET_STATS_NUM_WORKERS:-0}"
MIMIC_DATASET_STATS_BATCH_SIZE="${MIMIC_DATASET_STATS_BATCH_SIZE:-4}"

EXPERIMENT="${EXPERIMENT:-w2a_so101_${INIT_NAME}_${VIDEO_CKPT}_lr${LR_TAG}_layer20_bsz${GLOBAL_BATCH_SIZE}}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT}_${SLURM_JOB_ID:-manual}}"
# Keep TMPDIR short: Python multiprocessing AF_UNIX socket paths fail when
# TMPDIR includes long experiment names on Euler.
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mvs_${SLURM_JOB_ID:-manual}}"

mkdir -p "$LOG_DIR" "$RUNS_DIR" "$WANDB_ROOT_DIR" "$WANDB_ROOT_DIR/cache"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/torch-cache" "$RUN_TMP_DIR/triton-cache"

CUDA_COMPAT_DIR="$RUN_TMP_DIR/cuda-compat-lib"
mkdir -p "$CUDA_COMPAT_DIR"
if [ -f "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CUDA_COMPAT_DIR/libcudart.so"
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi
mkdir -p /cluster/scratch/eugseo/hf_home /cluster/scratch/eugseo/hf_datasets_cache

export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export TORCH_HOME="$RUN_TMP_DIR/torch-cache"
export TRITON_CACHE_DIR="$RUN_TMP_DIR/triton-cache"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RUNS_DIR}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-dreamdifferent}"
export WANDB_PROJECT="${WANDB_PROJECT:-mimic-video-so101}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-1}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT_DIR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT_DIR/cache}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export PYTHONUNBUFFERED=1
export SO101_DATA_DIR="$DATA_DIR"
export SO101_NUM_VAL_EPISODES="${SO101_NUM_VAL_EPISODES:-0}"
export MIMIC_DATASET_STATS_NUM_WORKERS
export MIMIC_DATASET_STATS_BATCH_SIZE

cd "$MODEL_DIR"

if [ "$INIT_NAME" = "partial_bridge_init" ]; then
  if [ "$VIDEO_CKPT" = "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused" ]; then
    PARTIAL_CKPT="$MODEL_DIR/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
  elif [ "$VIDEO_CKPT" = "v2w_pretrained_cosmos" ]; then
    PARTIAL_CKPT="$MODEL_DIR/checkpoints/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
  else
    echo "ERROR: no partial Bridge-init checkpoint is configured for VIDEO_CKPT=$VIDEO_CKPT"
    exit 1
  fi
  if [ ! -f "$PARTIAL_CKPT" ]; then
    echo "ERROR: partial_bridge_init requires missing checkpoint:"
    echo "  $PARTIAL_CKPT"
    echo "Use INIT_NAME=random or download the pretrained Cosmos Bridge checkpoint bundle."
    exit 1
  fi
fi

VIDEO_CKPT_PATH="$MODEL_DIR/checkpoints/video_backbone/${VIDEO_CKPT}.pt"
if [ ! -f "$VIDEO_CKPT_PATH" ]; then
  echo "WARNING: expected frozen video backbone not found:"
  echo "  $VIDEO_CKPT_PATH"
  echo "Training will fail unless checkpoints are available through another configured path."
fi

MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}"

echo "===== SO-101 mimic-video smoke training ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Repo: $REPO_ROOT"
echo "Model dir: $MODEL_DIR"
echo "Data dir: $DATA_DIR"
echo "Video checkpoint: $VIDEO_CKPT"
echo "Experiment: $EXPERIMENT"
if [ -n "$GRAD_ACCUM_ITER" ]; then
  echo "Gradient accumulation override: $GRAD_ACCUM_ITER"
else
  echo "Gradient accumulation override: <experiment default>"
fi
echo "Run tmp dir: $RUN_TMP_DIR"
echo "Output root: $IMAGINAIRE_OUTPUT_ROOT"
echo "W&B: enabled=$WANDB_ENABLED entity=$WANDB_ENTITY project=$WANDB_PROJECT mode=$WANDB_MODE dir=$WANDB_DIR"
echo "W&B scalar log frequency: every $WANDB_LOG_EVERY_N optimizer step(s)"
echo "Run validation: $RUN_VALIDATION"
echo "SO-101 val episodes: $SO101_NUM_VAL_EPISODES"
echo "Training dataloader workers: $NUM_WORKERS"
echo "Stats workers: $MIMIC_DATASET_STATS_NUM_WORKERS"
echo "Stats batch size: $MIMIC_DATASET_STATS_BATCH_SIZE"
echo "============================================="
nvidia-smi || true

TRAIN_CMD=(
  torchrun
  --nproc_per_node=1
  --master_port="$MASTER_PORT"
  -m scripts.train
  --config=cosmos_predict2/configs/config.py
  --
  "experiment=$EXPERIMENT"
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=$LOGGING_ITER"
  "trainer.run_validation=$RUN_VALIDATION"
  "trainer.validation_iter=$VALIDATION_ITER"
  "checkpoint.save_iter=$SAVE_ITER"
  "dataloader_train.num_workers=$NUM_WORKERS"
)

if [ -n "$GRAD_ACCUM_ITER" ]; then
  TRAIN_CMD+=("trainer.grad_accum_iter=$GRAD_ACCUM_ITER")
fi

if [ -x .venv/bin/python ] && [ -x .venv/bin/torchrun ]; then
  source .venv/bin/activate
  "${TRAIN_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${TRAIN_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  echo "Set up the environment first:"
  echo "  cd $MODEL_DIR && uv sync --extra cu126"
  exit 1
fi

echo "Finished SO-101 smoke run: $RUN_NAME"
