#!/bin/bash
#SBATCH --job-name=mimic_so101_ee_w2a
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40G
#SBATCH --output=/cluster/scratch/%u/mimic_video/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/mimic_video/logs/%x-%j.err

set -euo pipefail

module load eth_proxy
export PATH="/usr/sbin:/sbin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/model/cosmos_predict2" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
  else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
MODEL_DIR="$REPO_ROOT/model"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-$SCRATCH/mimic_video}"
MIMIC_VIDEO_EXPERIMENT="${MIMIC_VIDEO_EXPERIMENT:-so101_hetero_ee}"
MIMIC_VIDEO_EXPERIMENT_ROOT="${MIMIC_VIDEO_EXPERIMENT_ROOT:-$MIMIC_VIDEO_ROOT/experiments/$MIMIC_VIDEO_EXPERIMENT}"
MIMIC_VIDEO_SHARED_ROOT="${MIMIC_VIDEO_SHARED_ROOT:-$MIMIC_VIDEO_ROOT/shared}"
CHECKPOINT_ROOT="${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT:-$MIMIC_VIDEO_SHARED_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-${MIMIC_VIDEO_LOG_DIR:-$MIMIC_VIDEO_EXPERIMENT_ROOT/logs}}"
RUNS_DIR="${RUNS_DIR:-${MIMIC_VIDEO_RUNS_DIR:-$MIMIC_VIDEO_EXPERIMENT_ROOT/runs/action_decoder}}"
WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-$MIMIC_VIDEO_EXPERIMENT_ROOT/wandb}"

SO101_EE_DATA_DIR="${SO101_EE_DATA_DIR:-$MIMIC_VIDEO_SHARED_ROOT/data/so101_hetero_ee_action_30hz_full}"
SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-$MIMIC_VIDEO_SHARED_ROOT/video_data/so101_hetero_front_wrist_hstack_5fps_full}"
SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-5}"

DATA_CONFIG="${DATA_CONFIG:-so101_ee}"
VIDEO_CKPT="${VIDEO_CKPT:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused}"
VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-$CHECKPOINT_ROOT/video_backbone/${VIDEO_CKPT}.pt}"
ACTION_DIT_PATH="${ACTION_DIT_PATH:-$CHECKPOINT_ROOT/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt}"
VIDEO_LORA_CKPT="${VIDEO_LORA_CKPT:?Set VIDEO_LORA_CKPT to the local SO101 video LoRA adapter checkpoint. Download it from dreamdifferent/mimic-video-so101-multi-object-2cam-hstack-5fps-v2w-lora first.}"
VIDEO_LORA_RANK="${VIDEO_LORA_RANK:-256}"
VIDEO_LORA_ALPHA="${VIDEO_LORA_ALPHA:-32}"
VIDEO_LORA_TARGET_MODULES="${VIDEO_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / NUM_GPUS))}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-64}"
MAX_ITER="${MAX_ITER:-50}"
LOGGING_ITER="${LOGGING_ITER:-10}"
SAVE_ITER="${SAVE_ITER:-50}"
VALIDATION_ITER="${VALIDATION_ITER:-50}"
RUN_VALIDATION="${RUN_VALIDATION:-false}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SO101_NUM_VAL_EPISODES="${SO101_NUM_VAL_EPISODES:-10}"
MIMIC_DATASET_STATS_NUM_WORKERS="${MIMIC_DATASET_STATS_NUM_WORKERS:-0}"
MIMIC_DATASET_STATS_BATCH_SIZE="${MIMIC_DATASET_STATS_BATCH_SIZE:-4}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-dreamdifferent}"
WANDB_PROJECT="${WANDB_PROJECT:-mimic-video-so101-ee-w2a}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-1}"

EXPERIMENT="${EXPERIMENT:-w2a_so101_ee_${VIDEO_CKPT}_lr1.000e-04_layer20_bsz1}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT}_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mimic_video_so101_ee_w2a_${SLURM_JOB_ID:-manual}}"

if [ ! -d "$SO101_EE_DATA_DIR" ]; then
  echo "ERROR: SO101_EE_DATA_DIR does not exist: $SO101_EE_DATA_DIR"
  exit 1
fi
if ! find "$SO101_EE_DATA_DIR" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "ERROR: SO101_EE_DATA_DIR contains no .zarr episodes: $SO101_EE_DATA_DIR"
  exit 1
fi
if [ ! -d "$SO101_VIDEO_DIR/video" ]; then
  echo "ERROR: SO101_VIDEO_DIR must be a video2world dataset with a video/ folder: $SO101_VIDEO_DIR"
  exit 1
fi
if ! find "$SO101_VIDEO_DIR/video" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
  echo "ERROR: SO101_VIDEO_DIR contains no mp4 files under video/: $SO101_VIDEO_DIR"
  exit 1
fi
if [ ! -f "$VIDEO_DIT_PATH" ]; then
  echo "ERROR: frozen video backbone not found: $VIDEO_DIT_PATH"
  exit 1
fi
if [ ! -f "$ACTION_DIT_PATH" ]; then
  echo "ERROR: Bridge action decoder init checkpoint not found: $ACTION_DIT_PATH"
  exit 1
fi
if [ ! -f "$VIDEO_LORA_CKPT" ]; then
  echo "ERROR: VIDEO_LORA_CKPT does not exist: $VIDEO_LORA_CKPT"
  echo "Download example:"
  echo "  hf download dreamdifferent/mimic-video-so101-multi-object-2cam-hstack-5fps-v2w-lora checkpoints/model/iter_000001000.pt --repo-type model --local-dir $CHECKPOINT_ROOT/video_lora/so101_multi_object_2cam_hstack_5fps"
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS"
  exit 1
fi
if [ "$PER_GPU_BATCH_SIZE" -lt 1 ]; then
  echo "ERROR: PER_GPU_BATCH_SIZE=$PER_GPU_BATCH_SIZE must be >= 1"
  exit 1
fi

VENV_NVIDIA_DIR="$MODEL_DIR/.venv/lib/python3.10/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_DIR" ]; then
  export CUDA_HOME="${CUDA_HOME:-$VENV_NVIDIA_DIR}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export LD_LIBRARY_PATH="$VENV_NVIDIA_DIR/cuda_runtime/lib:$VENV_NVIDIA_DIR/cuda_nvrtc/lib:$VENV_NVIDIA_DIR/cudnn/lib:$VENV_NVIDIA_DIR/nccl/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$LOG_DIR" "$RUNS_DIR" "$WANDB_ROOT_DIR/cache"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/torch-cache" "$RUN_TMP_DIR/triton-cache"
CUDA_COMPAT_DIR="$RUN_TMP_DIR/cuda-compat-lib"
mkdir -p "$CUDA_COMPAT_DIR"
if [ -f "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CUDA_COMPAT_DIR/libcudart.so"
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi

export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export TORCH_HOME="$RUN_TMP_DIR/torch-cache"
export TRITON_CACHE_DIR="$RUN_TMP_DIR/triton-cache"
export HF_HOME="${HF_HOME:-$MIMIC_VIDEO_SHARED_ROOT/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MIMIC_VIDEO_SHARED_ROOT/hf-datasets-cache}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RUNS_DIR}"
export WANDB_ENABLED WANDB_ENTITY WANDB_PROJECT WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT_DIR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT_DIR/cache}"
export WANDB_LOG_EVERY_N
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export PYTHONUNBUFFERED=1
export SO101_EE_DATA_DIR SO101_VIDEO_DIR SO101_VIDEO_FPS SO101_NUM_VAL_EPISODES
export MIMIC_DATASET_STATS_NUM_WORKERS MIMIC_DATASET_STATS_BATCH_SIZE

cd "$MODEL_DIR"
MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}"

echo "===== SO101 EE action decoder training ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Experiment: $EXPERIMENT"
echo "Data config: $DATA_CONFIG"
echo "SO101 EE data: $SO101_EE_DATA_DIR"
echo "SO101 video data: $SO101_VIDEO_DIR"
echo "Video backbone: $VIDEO_DIT_PATH"
echo "Video LoRA: $VIDEO_LORA_CKPT"
echo "Action init: $ACTION_DIT_PATH"
echo "Output root: $IMAGINAIRE_OUTPUT_ROOT"
echo "GPUs: $NUM_GPUS per_gpu_batch=$PER_GPU_BATCH_SIZE global_batch=$GLOBAL_BATCH_SIZE grad_accum=$GRAD_ACCUM_ITER"
echo "max_iter=$MAX_ITER save_iter=$SAVE_ITER validation=$RUN_VALIDATION"
echo "W&B: enabled=$WANDB_ENABLED entity=$WANDB_ENTITY project=$WANDB_PROJECT mode=$WANDB_MODE"
echo "==========================================="
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
  "dataloader_train.num_workers=$NUM_WORKERS"
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=$LOGGING_ITER"
  "trainer.run_validation=$RUN_VALIDATION"
  "trainer.validation_iter=$VALIDATION_ITER"
  "trainer.grad_accum_iter=$GRAD_ACCUM_ITER"
  "checkpoint.save_iter=$SAVE_ITER"
  "model.config.video_dit_path=$VIDEO_DIT_PATH"
  "model.config.action_dit_path=$ACTION_DIT_PATH"
  "model.config.video_lora_dit_path=$VIDEO_LORA_CKPT"
  "model.config.video_lora_rank=$VIDEO_LORA_RANK"
  "model.config.video_lora_alpha=$VIDEO_LORA_ALPHA"
  "model.config.video_lora_target_modules='$VIDEO_LORA_TARGET_MODULES'"
)

if [ -x .venv/bin/python ] && [ -x .venv/bin/torchrun ]; then
  source .venv/bin/activate
  "${TRAIN_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${TRAIN_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  echo "Set up the environment first: cd $MODEL_DIR && uv sync --extra cu126"
  exit 1
fi
