#!/bin/bash
#SBATCH --job-name=mimic_so101_t5
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --output=/cluster/scratch/%u/mimic_video/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/mimic_video/logs/%x-%j.err

set -euo pipefail

# Euler compute nodes need this if Hugging Face/checkpoint access is triggered.
module load eth_proxy

################################################################################
# Precompute T5 language embeddings for converted SO-101 mimic-video zarr data.
################################################################################
# Default smoke dataset:
#   sbatch model/scripts/precompute_so101_t5_smoke.sbatch
#
# Useful overrides:
#   DATA_DIR=/cluster/scratch/$USER/mimic_video/shared/data/so101_bottle_smoke
#   PROMPT="pick up the bottle and place it into the container"
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_DIR="$REPO_ROOT/model"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-${MVS_ROOT:-$SCRATCH/mimic_video}}"
LOG_DIR="${LOG_DIR:-${MVS_LOG_DIR:-$MIMIC_VIDEO_ROOT/logs}}"
DATA_DIR="${DATA_DIR:-$MIMIC_VIDEO_ROOT/shared/data/so101_bottle_smoke}"
RUN_NAME="${RUN_NAME:-so101_t5_${SLURM_JOB_ID:-manual}}"
RUN_TMP_DIR="${RUN_TMP_DIR:-$REPO_ROOT/tmp/$RUN_NAME}"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_TMP_DIR/uv-cache" "$RUN_TMP_DIR/torch-cache"
mkdir -p "$MIMIC_VIDEO_ROOT/shared/hf-home" "$MIMIC_VIDEO_ROOT/shared/hf-datasets-cache"

export TMPDIR="$RUN_TMP_DIR"
export UV_CACHE_DIR="$RUN_TMP_DIR/uv-cache"
export TORCH_HOME="$RUN_TMP_DIR/torch-cache"
export HF_HOME="${HF_HOME:-${MVS_SHARED_ROOT:+$MVS_SHARED_ROOT/hf-home}}"
export HF_HOME="${HF_HOME:-$MIMIC_VIDEO_ROOT/shared/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MVS_SHARED_ROOT:+$MVS_SHARED_ROOT/hf-datasets-cache}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MIMIC_VIDEO_ROOT/shared/hf-datasets-cache}"
export PYTHONUNBUFFERED=1
export DATA_DIR

# Make checkpoint resolution explicit even when running from unusual cwd.
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints $MODEL_DIR/checkpoints}"

# T5 import path is under model/; run from MODEL_DIR.
cd "$MODEL_DIR"

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
  exit 1
fi

if ! find "$DATA_DIR" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "ERROR: DATA_DIR contains no .zarr episodes: $DATA_DIR"
  exit 1
fi

T5_DIR="$MODEL_DIR/checkpoints/text_encoder/t5-11b"
if [ ! -d "$T5_DIR" ]; then
  echo "ERROR: T5 checkpoint directory is missing: $T5_DIR"
  echo "Download/symlink checkpoints before running T5 precompute."
  exit 1
fi

PRECOMPUTE_CMD=(
  python
  "$REPO_ROOT/data_preprocessing/action/precompute_t5.py"
  --dataset-path "$DATA_DIR"
)

if [ -n "${PROMPT:-}" ]; then
  PRECOMPUTE_CMD+=(--prompt "$PROMPT")
fi

echo "===== SO-101 T5 precompute ====="
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Repo: $REPO_ROOT"
echo "Model dir: $MODEL_DIR"
echo "Data dir: $DATA_DIR"
echo "T5 dir: $T5_DIR"
echo "Prompt override: ${PROMPT:-<per-episode language_instruction>}"
echo "Run tmp dir: $RUN_TMP_DIR"
echo "COSMOS_PREDICT2_ARGS: $COSMOS_PREDICT2_ARGS"
echo "==============================="
nvidia-smi || true

if [ -x .venv/bin/python ]; then
  source .venv/bin/activate
  "${PRECOMPUTE_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${PRECOMPUTE_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available."
  echo "Set up the environment first:"
  echo "  cd $MODEL_DIR && uv sync --extra cu126"
  exit 1
fi

echo "T5 precompute finished. Checking language_embedding keys:"
python - <<'PY'
from pathlib import Path
import os
import zarr

root = Path(os.environ["DATA_DIR"])
for ep in sorted(root.glob("*.zarr")):
    z = zarr.open(str(ep), "r")
    arr = z["language_embedding"]
    print(f"{ep.name}: language_embedding shape={arr.shape} dtype={arr.dtype}")
PY
