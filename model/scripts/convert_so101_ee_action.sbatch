#!/bin/bash
#SBATCH --job-name=mimic_so101_ee_action
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
URDF_PATH="${URDF_PATH:?Set URDF_PATH to local Simulation/SO101/so101_new_calib.urdf.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the SO101 EE action zarr dataset.}"
TARGET_FRAME_NAME="${TARGET_FRAME_NAME:-gripper_frame_link}"
MAX_EPISODES="${MAX_EPISODES-}"
EPISODES="${EPISODES-}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$PYTHON_BIN" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_PATH"
elif [ -z "$PYTHON_BIN" ]; then
  export PATH="$CONDA_ENV_PATH/bin:$PATH"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" - <<'PY'
import importlib.util

missing = []
for module in ("lerobot", "pandas", "zarr", "numcodecs", "placo"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
if missing:
    raise SystemExit(
        "SO101 EE conversion requires lerobot, pandas, zarr, numcodecs, and placo.\n"
        "Use PYTHON_BIN or CONDA_ENV_PATH pointing to an environment with those packages.\n"
        "Missing modules: "
        + ", ".join(missing)
    )
PY

if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" = "1" ]; then
  rm -rf "$OUTPUT_DIR"
fi
if [ -e "$OUTPUT_DIR" ] && [ "$OVERWRITE" != "1" ] && [ "$DRY_RUN" != "1" ]; then
  echo "ERROR: OUTPUT_DIR exists: $OUTPUT_DIR"
  echo "Set OVERWRITE=1 or choose a new OUTPUT_DIR."
  exit 1
fi

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/data_preprocessing/action/process_so101_lerobot_ee.py"
  --root "$LEROBOT_ROOT"
  --urdf-path "$URDF_PATH"
  --target-frame-name "$TARGET_FRAME_NAME"
  --output-dir "$OUTPUT_DIR"
)
if [ -n "$MAX_EPISODES" ]; then
  cmd+=(--max-episodes "$MAX_EPISODES")
fi
if [ -n "$EPISODES" ]; then
  # shellcheck disable=SC2206
  episode_args=($EPISODES)
  cmd+=(--episodes "${episode_args[@]}")
fi
if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi
if [ "$DRY_RUN" = "1" ]; then
  cmd+=(--dry-run)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
