#!/bin/bash
#SBATCH --job-name=mimic_so101_2cam_video_convert
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.out
#SBATCH --error=/cluster/scratch/eugseo/mimic_video_logs/%x-%j.err

set -euo pipefail
module load eth_proxy

REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/eugseo/workspace/mimic-video}"
MODEL_DIR="$REPO_ROOT/model"
LOG_DIR="${LOG_DIR:-/cluster/scratch/eugseo/mimic_video_logs}"
REPO_ID="${REPO_ID:-dreamdifferent/so101_bottle}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/scratch/eugseo/mimic_video_video_data/so101_bottle_front_wrist_hstack_5fps_smoke}"
MAX_EPISODES="${MAX_EPISODES-2}"
OVERWRITE="${OVERWRITE:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SOURCE_FPS="${SOURCE_FPS:-30}"
TARGET_FPS="${TARGET_FPS:-5}"
VIEW_LAYOUT="${VIEW_LAYOUT:-hstack}"
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mvs_2cam_convert_${SLURM_JOB_ID:-manual}}"

mkdir -p "$LOG_DIR" "$RUN_TMP_DIR/uv-cache" /cluster/scratch/eugseo/hf_home /cluster/scratch/eugseo/hf_datasets_cache
export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export HF_HOME="${HF_HOME:-/cluster/scratch/eugseo/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cluster/scratch/eugseo/hf_datasets_cache}"
export PYTHONUNBUFFERED=1

if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" = "1" ]; then
  rm -rf "$OUTPUT_DIR"
fi
if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" != "1" ]; then
  echo "ERROR: OUTPUT_DIR exists: $OUTPUT_DIR"
  echo "Set OVERWRITE=1 or choose a new OUTPUT_DIR."
  exit 1
fi

cd "$REPO_ROOT"
echo "===== SO-101 two-camera video conversion ====="
echo "Repo id: $REPO_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Max episodes: ${MAX_EPISODES:-full}"
echo "Video backend: $VIDEO_BACKEND"
echo "FPS: source=$SOURCE_FPS target=$TARGET_FPS"
echo "View layout: $VIEW_LAYOUT"
echo "=============================================="

CMD=(
  python "$REPO_ROOT/data_preprocessing/video/process_so101_two_camera_video.py"
  --repo-id "$REPO_ID"
  --output-dir "$OUTPUT_DIR"
  --video-backend "$VIDEO_BACKEND"
  --source-fps "$SOURCE_FPS"
  --target-fps "$TARGET_FPS"
  --view-layout "$VIEW_LAYOUT"
  --overwrite
)
if [ -n "$MAX_EPISODES" ]; then
  CMD+=(--max-episodes "$MAX_EPISODES")
fi

if [ -x "$MODEL_DIR/.venv/bin/python" ]; then
  source "$MODEL_DIR/.venv/bin/activate"
  "${CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  cd "$MODEL_DIR"
  uv run --extra cu126 "${CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  exit 1
fi

find "$OUTPUT_DIR" -maxdepth 2 -type f | sort | sed -n '1,40p'
