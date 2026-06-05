#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-homogeneous-delta30}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi

mvs_load_paths "${experiment}"
mvs_create_layout

mvs_require_dir "${MVS_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"
mvs_require_file "${MVS_VIDEO_LORA_PATH}" "video LoRA"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATASET_DIR="${DATASET_DIR:-${MVS_SHARED_VIDEO_DATA_DIR}}"
export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MVS_VIDEO_LORA_PATH}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MVS_EXP_ROOT}/previews/video_lora_iter1000}"
export LOG_DIR="${LOG_DIR:-${MVS_LOG_DIR}}"

mkdir -p "${MVS_LOG_DIR}" "${OUTPUT_DIR}"

cmd=(
  sbatch
  --output="${MVS_LOG_DIR}/%x-%j.out"
  --error="${MVS_LOG_DIR}/%x-%j.err"
  "$@"
  "${REPO_ROOT}/model/scripts/render_so101_two_camera_video_preview.sbatch"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

"${cmd[@]}"
