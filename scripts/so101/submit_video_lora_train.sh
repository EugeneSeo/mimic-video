#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-homogeneous-rel}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mvs_load_paths "${experiment}"
mvs_create_layout

mvs_require_file "${MVS_VIDEO_BACKBONE_PATH}" "base video backbone"
mvs_require_dir "${MVS_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATASET_DIR="${DATASET_DIR:-${MVS_SHARED_VIDEO_DATA_DIR}}"
export RUNS_DIR="${RUNS_DIR:-${MVS_RUNS_DIR}/video_lora}"
export WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MVS_EXP_ROOT}/wandb}"
export SO101_TWO_CAMERA_INIT_DIT_PATH="${SO101_TWO_CAMERA_INIT_DIT_PATH:-${MVS_VIDEO_BACKBONE_PATH}}"
export EXPERIMENT="${VIDEO_LORA_EXPERIMENT_NAME:-v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4}"

mkdir -p "${MVS_LOG_DIR}" "${RUNS_DIR}" "${WANDB_ROOT_DIR}"

sbatch \
  --output="${MVS_LOG_DIR}/%x-%j.out" \
  --error="${MVS_LOG_DIR}/%x-%j.err" \
  "$@" \
  "${REPO_ROOT}/model/scripts/train_so101_two_camera_video_backbone_2gpu.sbatch"
