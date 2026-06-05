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

mimic_video_require_dir "${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"
mimic_video_require_file "${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}" "video LoRA"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATASET_DIR="${DATASET_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}}"
checkpoint_tag="$(basename "${MODEL_CHECKPOINT}" .pt)"
export OUTPUT_DIR="${OUTPUT_DIR:-${MIMIC_VIDEO_EXPERIMENT_ROOT}/previews/video_lora_${checkpoint_tag}}"
export LOG_DIR="${LOG_DIR:-${MIMIC_VIDEO_LOG_DIR}}"

mkdir -p "${MIMIC_VIDEO_LOG_DIR}" "${OUTPUT_DIR}"

cmd=(
  sbatch
  --output="${MIMIC_VIDEO_LOG_DIR}/%x-%j.out"
  --error="${MIMIC_VIDEO_LOG_DIR}/%x-%j.err"
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
