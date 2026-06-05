#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MIMIC_VIDEO_EXPERIMENT:-so101-bottle-delta30}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mimic_video_load_paths "${experiment}"
mimic_video_create_layout

mimic_video_require_file "${MIMIC_VIDEO_BACKBONE_PATH}" "base video backbone"
mimic_video_require_dir "${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATASET_DIR="${DATASET_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
export RUNS_DIR="${RUNS_DIR:-${MIMIC_VIDEO_RUNS_DIR}/video_lora}"
export WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MIMIC_VIDEO_EXPERIMENT_ROOT}/wandb}"
export SO101_TWO_CAMERA_INIT_DIT_PATH="${SO101_TWO_CAMERA_INIT_DIT_PATH:-${MIMIC_VIDEO_BACKBONE_PATH}}"
export EXPERIMENT="${VIDEO_LORA_EXPERIMENT_NAME:-v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4}"

mkdir -p "${MIMIC_VIDEO_LOG_DIR}" "${RUNS_DIR}" "${WANDB_ROOT_DIR}"

cmd=(
  sbatch
  --output="${MIMIC_VIDEO_LOG_DIR}/%x-%j.out"
  --error="${MIMIC_VIDEO_LOG_DIR}/%x-%j.err"
  "$@"
  "${REPO_ROOT}/model/scripts/train_so101_two_camera_video_backbone_2gpu.sbatch"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

"${cmd[@]}"
