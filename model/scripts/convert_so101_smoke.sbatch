#!/bin/bash
#SBATCH --job-name=mimic_so101_convert
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/mimic_video/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/mimic_video/logs/%x-%j.err

set -euo pipefail

# Euler compute nodes need this for external PyPI/Hugging Face access.
module load eth_proxy

################################################################################
# Convert LeRobot SO-101 episodes to mimic-video zarr format.
################################################################################
# Default is a 2-episode smoke conversion:
#   sbatch model/scripts/convert_so101_smoke.sbatch
#
# Useful overrides:
#   MAX_EPISODES=2 | empty for full dataset
#   OUTPUT_DIR=/cluster/scratch/$USER/mimic_video/shared/data/so101_bottle_smoke
#   CAMERA_KEY=observation.images.front | observation.images.wrist
#   SKIP_VIDEO=1  # write only state/action/language; use SO101_VIDEO_DIR during training
#   VIDEO_BACKEND=pyav | torchcodec | video_reader
#   OVERWRITE=1
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_DIR="$REPO_ROOT/model"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-${MVS_ROOT:-$SCRATCH/mimic_video}}"
LOG_DIR="${LOG_DIR:-${MVS_LOG_DIR:-$MIMIC_VIDEO_ROOT/logs}}"

REPO_ID="${REPO_ID:-dreamdifferent/so101_bottle}"
CAMERA_KEY="${CAMERA_KEY:-observation.images.front}"
SKIP_VIDEO="${SKIP_VIDEO:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
OUTPUT_DIR="${OUTPUT_DIR:-$MIMIC_VIDEO_ROOT/shared/data/so101_bottle_smoke}"
# Use `${VAR-2}` rather than `${VAR:-2}` so `MAX_EPISODES=` means full dataset.
MAX_EPISODES="${MAX_EPISODES-2}"
OVERWRITE="${OVERWRITE:-0}"
RUN_NAME="${RUN_NAME:-so101_convert_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$REPO_ROOT/tmp/$RUN_NAME}"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_TMP_DIR/uv-cache"
mkdir -p "$MIMIC_VIDEO_ROOT/shared/hf-home" "$MIMIC_VIDEO_ROOT/shared/hf-datasets-cache"
mkdir -p "$(dirname "$OUTPUT_DIR")"

export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export HF_HOME="${HF_HOME:-${MVS_SHARED_ROOT:+$MVS_SHARED_ROOT/hf-home}}"
export HF_HOME="${HF_HOME:-$MIMIC_VIDEO_ROOT/shared/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MVS_SHARED_ROOT:+$MVS_SHARED_ROOT/hf-datasets-cache}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MIMIC_VIDEO_ROOT/shared/hf-datasets-cache}"
export PYTHONUNBUFFERED=1

if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" != "1" ]; then
  echo "ERROR: OUTPUT_DIR already exists: $OUTPUT_DIR"
  echo "Set OVERWRITE=1 to remove and recreate it, or choose a new OUTPUT_DIR."
  exit 1
fi

if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" = "1" ]; then
  echo "Removing existing OUTPUT_DIR because OVERWRITE=1: $OUTPUT_DIR"
  rm -rf "$OUTPUT_DIR"
fi

cd "$REPO_ROOT"

echo "===== SO-101 LeRobot -> mimic zarr conversion ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Repo: $REPO_ROOT"
echo "Repo id: $REPO_ID"
echo "Camera key: $CAMERA_KEY"
echo "Skip video: $SKIP_VIDEO"
echo "Video backend: $VIDEO_BACKEND"
echo "Output dir: $OUTPUT_DIR"
echo "Max episodes: ${MAX_EPISODES:-full}"
echo "Run tmp dir: $RUN_TMP_DIR"
echo "==================================================="

CONVERT_CMD=(
  python
  "$REPO_ROOT/data_preprocessing/action/process_so101_lerobot.py"
  --repo-id "$REPO_ID"
  --output-dir "$OUTPUT_DIR"
  --camera-key "$CAMERA_KEY"
  --video-backend "$VIDEO_BACKEND"
)

if [ "$SKIP_VIDEO" = "1" ]; then
  CONVERT_CMD+=(--skip-video)
fi

if [ -n "$MAX_EPISODES" ]; then
  CONVERT_CMD+=(--max-episodes "$MAX_EPISODES")
fi

if [ -x "$MODEL_DIR/.venv/bin/python" ] && "$MODEL_DIR/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import lerobot
import numcodecs
import numpy
import tqdm
import zarr
PY
then
  source "$MODEL_DIR/.venv/bin/activate"
  "${CONVERT_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  cd "$MODEL_DIR"
  uv run --extra cu126 "${CONVERT_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  echo "Set up the environment first:"
  echo "  cd $MODEL_DIR && uv sync --extra cu126"
  exit 1
fi

echo "Conversion finished. Created:"
find "$OUTPUT_DIR" -maxdepth 2 -name '*.zarr' -type d | sort
