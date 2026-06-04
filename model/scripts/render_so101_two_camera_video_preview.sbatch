#!/bin/bash
#SBATCH --job-name=mimic_so101_v2w_preview
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80G
#SBATCH --output=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.out
#SBATCH --error=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.err

set -euo pipefail
module load eth_proxy
export PATH="/usr/sbin:/sbin:$PATH"

REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/eugseo/workspace/mimic-video}"
MODEL_DIR="$REPO_ROOT/model"
DATASET_DIR="${DATASET_DIR:-/cluster/scratch/eugseo/mimic_video_video_data/so101_bottle_front_wrist_hstack_5fps_full}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/tmp/so101_v2w_preview_iter1000}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-20}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
SEED="${SEED:-0}"

VENV_NVIDIA_DIR="$MODEL_DIR/.venv/lib/python3.10/site-packages/nvidia"
if [ -d "$VENV_NVIDIA_DIR" ]; then
  export CUDA_HOME="${CUDA_HOME:-$VENV_NVIDIA_DIR}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export LD_LIBRARY_PATH="$VENV_NVIDIA_DIR/cuda_runtime/lib:$VENV_NVIDIA_DIR/cuda_nvrtc/lib:$VENV_NVIDIA_DIR/cudnn/lib:$VENV_NVIDIA_DIR/nccl/lib:${LD_LIBRARY_PATH:-}"
fi

RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mvs_so101_v2w_preview_${SLURM_JOB_ID:-manual}}"
mkdir -p /cluster/scratch/eugseo/mimic_video_logs "$OUTPUT_DIR" "$RUN_TMP_DIR/torch-cache" "$RUN_TMP_DIR/triton-cache"
export TMPDIR="$RUN_TMP_DIR"
export TORCH_HOME="$RUN_TMP_DIR/torch-cache"
export TRITON_CACHE_DIR="$RUN_TMP_DIR/triton-cache"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export PYTHONUNBUFFERED=1

cd "$MODEL_DIR"
MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}"

echo "===== SO-101 two-camera V2W preview ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Dataset: $DATASET_DIR"
echo "Checkpoint: $MODEL_CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES sampling_steps=$NUM_SAMPLING_STEPS conditional_frames=$NUM_CONDITIONAL_FRAMES seed=$SEED"
echo "========================================="
nvidia-smi || true

source .venv/bin/activate
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" -m scripts.render_so101_two_camera_video_preview \
  --dataset-dir "$DATASET_DIR" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --num-conditional-frames "$NUM_CONDITIONAL_FRAMES" \
  --seed "$SEED"
