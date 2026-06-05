#!/bin/bash
#SBATCH --job-name=mimic_so101_2cam_video_t5
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --output=/cluster/scratch/%u/mimic_video/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/mimic_video/logs/%x-%j.err

set -euo pipefail
module load eth_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_DIR="$REPO_ROOT/model"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-$SCRATCH/mimic_video}"
LOG_DIR="${LOG_DIR:-${MIMIC_VIDEO_LOG_DIR:-$MIMIC_VIDEO_ROOT/logs}}"
DATASET_PATH="${DATASET_PATH:-$MIMIC_VIDEO_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_smoke}"
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mimic_video_2cam_t5_${SLURM_JOB_ID:-manual}}"

mkdir -p "$LOG_DIR" "$RUN_TMP_DIR/uv-cache" "$MIMIC_VIDEO_ROOT/shared/hf-home"
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export HF_HOME="${HF_HOME:-${MIMIC_VIDEO_SHARED_ROOT:+$MIMIC_VIDEO_SHARED_ROOT/hf-home}}"
export HF_HOME="${HF_HOME:-$MIMIC_VIDEO_ROOT/shared/hf-home}"
export PYTHONUNBUFFERED=1

if ! find "$DATASET_PATH/video" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
  echo "ERROR: no mp4 videos found under $DATASET_PATH/video"
  exit 1
fi

cd "$MODEL_DIR"
echo "Computing T5 embeddings for $DATASET_PATH"
CMD=(python "$REPO_ROOT/data_preprocessing/video/get_t5_embeddings.py" --dataset_path "$DATASET_PATH")
if [ -x .venv/bin/python ]; then
  source .venv/bin/activate
  "${CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  exit 1
fi

find "$DATASET_PATH/t5_xxl" -maxdepth 1 -name '*.pickle' -type f | wc -l
