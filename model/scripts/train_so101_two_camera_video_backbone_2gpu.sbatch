#!/bin/bash
#SBATCH --job-name=mimic_so101_2cam_v2w_2gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:80G
#SBATCH --output=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.out
#SBATCH --error=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.err

set -euo pipefail
module load eth_proxy
export PATH="/usr/sbin:/sbin:$PATH"

REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/eugseo/workspace/mimic-video}"
MODEL_DIR="$REPO_ROOT/model"
DATASET_DIR="${DATASET_DIR:-/cluster/scratch/eugseo/mimic_video_video_data/so101_bottle_front_wrist_hstack_5fps_full}"
RUNS_DIR="${RUNS_DIR:-/cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_2gpu}"
WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-/cluster/scratch/eugseo/mimic_video_wandb}"
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mvs_2cam_v2w_2gpu_${SLURM_JOB_ID:-manual}}"
SO101_TWO_CAMERA_INIT_DIT_PATH="${SO101_TWO_CAMERA_INIT_DIT_PATH:-/cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"

NUM_GPUS="${NUM_GPUS:-2}"
EXPERIMENT="${EXPERIMENT:-v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-2}}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-8}"
MAX_ITER="${MAX_ITER:-50}"
SAVE_ITER="${SAVE_ITER:-100}"
KEEP_LATEST_ONLY="${KEEP_LATEST_ONLY:-true}"
LOGGING_ITER="${LOGGING_ITER:-10}"
RUN_VALIDATION="${RUN_VALIDATION:-false}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-dreamdifferent}"
WANDB_PROJECT="${WANDB_PROJECT:-mimic-video-so101-v2w}"
WANDB_MODE="${WANDB_MODE:-online}"
SO101_V2W_WANDB_VIDEO_EVAL="${SO101_V2W_WANDB_VIDEO_EVAL:-1}"
SO101_V2W_WANDB_VIDEO_EVERY_N="${SO101_V2W_WANDB_VIDEO_EVERY_N:-50}"
SO101_V2W_WANDB_VIDEO_START_AFTER="${SO101_V2W_WANDB_VIDEO_START_AFTER:-300}"
SO101_V2W_WANDB_VIDEO_ITERS="${SO101_V2W_WANDB_VIDEO_ITERS:-}"
SO101_V2W_WANDB_VIDEO_NUM_SAMPLES="${SO101_V2W_WANDB_VIDEO_NUM_SAMPLES:-2}"
SO101_V2W_WANDB_VIDEO_STEPS="${SO101_V2W_WANDB_VIDEO_STEPS:-20}"

VENV_NVIDIA_DIR="$MODEL_DIR/.venv/lib/python3.10/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_DIR" ]; then
  export CUDA_HOME="${CUDA_HOME:-$VENV_NVIDIA_DIR}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export LD_LIBRARY_PATH="$VENV_NVIDIA_DIR/cuda_runtime/lib:$VENV_NVIDIA_DIR/cuda_nvrtc/lib:$VENV_NVIDIA_DIR/cudnn/lib:$VENV_NVIDIA_DIR/nccl/lib:${LD_LIBRARY_PATH:-}"
fi
CUDA_COMPAT_DIR="$RUN_TMP_DIR/cuda-compat-lib"
mkdir -p "$CUDA_COMPAT_DIR"
if [ -f "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CUDA_COMPAT_DIR/libcudart.so"
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi

if ! find "$DATASET_DIR/video" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
  echo "ERROR: no mp4 videos found under $DATASET_DIR/video"
  exit 1
fi
if ! find "$DATASET_DIR/t5_xxl" -maxdepth 1 -name '*.pickle' -type f -print -quit | grep -q .; then
  echo "ERROR: no T5 embeddings found under $DATASET_DIR/t5_xxl"
  exit 1
fi
NUM_VIDEOS="$(find "$DATASET_DIR/video" -maxdepth 1 -name '*.mp4' -type f | wc -l)"
if [ "$NUM_VIDEOS" -lt "$PER_GPU_BATCH_SIZE" ]; then
  echo "ERROR: dataset has $NUM_VIDEOS videos but PER_GPU_BATCH_SIZE=$PER_GPU_BATCH_SIZE."
  echo "The training dataloader uses drop_last=True, so this would create zero batches."
  exit 1
fi
if [ ! -f "$SO101_TWO_CAMERA_INIT_DIT_PATH" ]; then
  echo "ERROR: init video DiT checkpoint not found: $SO101_TWO_CAMERA_INIT_DIT_PATH"
  exit 1
fi

mkdir -p /cluster/scratch/eugseo/mimic_video_logs "$RUNS_DIR" "$WANDB_ROOT_DIR/cache" "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/torch-cache" "$RUN_TMP_DIR/triton-cache"
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export TORCH_HOME="$RUN_TMP_DIR/torch-cache"
export TRITON_CACHE_DIR="$RUN_TMP_DIR/triton-cache"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export IMAGINAIRE_OUTPUT_ROOT="$RUNS_DIR"
export WANDB_ENABLED WANDB_ENTITY WANDB_PROJECT WANDB_MODE
export WANDB_DIR="$WANDB_ROOT_DIR"
export WANDB_CACHE_DIR="$WANDB_ROOT_DIR/cache"
export WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export PYTHONUNBUFFERED=1
export SO101_TWO_CAMERA_VIDEO_DIR="$DATASET_DIR"
export SO101_TWO_CAMERA_INIT_DIT_PATH
export SO101_V2W_WANDB_VIDEO_EVAL SO101_V2W_WANDB_VIDEO_EVERY_N SO101_V2W_WANDB_VIDEO_START_AFTER SO101_V2W_WANDB_VIDEO_ITERS SO101_V2W_WANDB_VIDEO_NUM_SAMPLES SO101_V2W_WANDB_VIDEO_STEPS

cd "$MODEL_DIR"
MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}"

echo "===== SO-101 two-camera V2W 2-GPU training ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Dataset: $DATASET_DIR"
echo "Videos: $NUM_VIDEOS"
echo "Init video DiT: $SO101_TWO_CAMERA_INIT_DIT_PATH"
echo "Experiment: $EXPERIMENT"
echo "Output root: $IMAGINAIRE_OUTPUT_ROOT"
echo "GPUs: $NUM_GPUS"
echo "Per-GPU batch: $PER_GPU_BATCH_SIZE grad_accum=$GRAD_ACCUM_ITER effective_batch=$((PER_GPU_BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_ITER))"
echo "max_iter=$MAX_ITER save_iter=$SAVE_ITER keep_latest_only=$KEEP_LATEST_ONLY"
echo "W&B video eval: enabled=$SO101_V2W_WANDB_VIDEO_EVAL start_after=$SO101_V2W_WANDB_VIDEO_START_AFTER every_n=$SO101_V2W_WANDB_VIDEO_EVERY_N extra_iters=$SO101_V2W_WANDB_VIDEO_ITERS samples=$SO101_V2W_WANDB_VIDEO_NUM_SAMPLES steps=$SO101_V2W_WANDB_VIDEO_STEPS"
echo "==============================================="
nvidia-smi || true

TRAIN_CMD=(
  torchrun
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  -m scripts.train
  --config=cosmos_predict2/configs/config.py
  --
  "experiment=$EXPERIMENT"
  "dataloader_train.batch_size=$PER_GPU_BATCH_SIZE"
  "trainer.grad_accum_iter=$GRAD_ACCUM_ITER"
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=$LOGGING_ITER"
  "trainer.run_validation=$RUN_VALIDATION"
  "checkpoint.save_iter=$SAVE_ITER"
  "checkpoint.keep_latest_only=$KEEP_LATEST_ONLY"
  "dataloader_train.num_workers=$NUM_WORKERS"
)

if [ -x .venv/bin/python ] && [ -x .venv/bin/torchrun ]; then
  source .venv/bin/activate
  "${TRAIN_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${TRAIN_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  exit 1
fi
