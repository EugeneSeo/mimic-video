#!/bin/bash
#SBATCH --job-name=mimic_so101_2cam_video
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/mimic_video/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/mimic_video/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cluster/home/dohkim/miniforge3/envs/lerobot}"
CONDA_SH="${CONDA_SH:-/cluster/home/dohkim/miniforge3/etc/profile.d/conda.sh}"
PYTHON_BIN="${PYTHON_BIN:-}"

LEROBOT_ROOT="${LEROBOT_ROOT:?Set LEROBOT_ROOT to the local LeRobot v3 dataset root.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the hstack video dataset.}"
REPO_ID="${REPO_ID:-dreamdifferent/so101_bottle}"
SOURCE_FPS="${SOURCE_FPS:-30}"
TARGET_FPS="${TARGET_FPS:-5}"
VIEW_LAYOUT="${VIEW_LAYOUT:-hstack}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
MAX_EPISODES="${MAX_EPISODES-}"
OVERWRITE="${OVERWRITE:-0}"

if [ -z "$PYTHON_BIN" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_PATH"
elif [ -z "$PYTHON_BIN" ]; then
  export PATH="$CONDA_ENV_PATH/bin:$PATH"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" = "1" ]; then
  rm -rf "$OUTPUT_DIR"
fi
if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" != "1" ]; then
  echo "ERROR: OUTPUT_DIR exists: $OUTPUT_DIR"
  echo "Set OVERWRITE=1 or choose a new OUTPUT_DIR."
  exit 1
fi

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/data_preprocessing/video/process_so101_two_camera_video.py"
  --repo-id "$REPO_ID"
  --root "$LEROBOT_ROOT"
  --output-dir "$OUTPUT_DIR"
  --source-fps "$SOURCE_FPS"
  --target-fps "$TARGET_FPS"
  --view-layout "$VIEW_LAYOUT"
  --video-backend "$VIDEO_BACKEND"
  --overwrite
)
if [ -n "$MAX_EPISODES" ]; then
  cmd+=(--max-episodes "$MAX_EPISODES")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
