#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-bottle-delta30.env}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  config_env="$1"
  shift
fi
mimic_video_load_paths "${config_env}"
mimic_video_create_layout
mimic_video_prepare_runtime_env
mimic_video_activate_model_venv_if_present

mimic_video_require_file "${MIMIC_VIDEO_BACKBONE_PATH}" "base video backbone"
mimic_video_require_file "${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}" "video LoRA"
mimic_video_require_file "${MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH}" "${ACTION_TRANSFORM} action decoder"

use_stats="${MIMIC_VIDEO_SERVER_USE_STATS:-1}"
if [[ "${use_stats}" == "1" ]]; then
  mimic_video_require_file "${MIMIC_VIDEO_ACTION_DECODER_STATS_PATH}" "${ACTION_TRANSFORM} normalizer stats"
else
  mimic_video_require_dir "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" "action zarr"
fi

cd "${REPO_ROOT}"

cmd=(
  python eval/so101/policy_server.py
  --host "${MIMIC_VIDEO_SO101_HOST:-127.0.0.1}" \
  --port "${MIMIC_VIDEO_SO101_PORT:-8000}" \
  --experiment-name "${POLICY_EXPERIMENT_NAME}" \
  --video-model-path "${MIMIC_VIDEO_BACKBONE_PATH}" \
  --video-lora-path "${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}" \
  --action-model-path "${MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH}" \
  --num-val-episodes "${MIMIC_VIDEO_SERVER_NUM_VAL_EPISODES:-0}" \
  --num-sampling-steps "${MIMIC_VIDEO_NUM_SAMPLING_STEPS}" \
  --stop-video-denoising-step "${MIMIC_VIDEO_STOP_VIDEO_DENOISING_STEP}"
)

if [[ "${use_stats}" == "1" ]]; then
  cmd+=(--stats-path "${MIMIC_VIDEO_ACTION_DECODER_STATS_PATH}")
else
  cmd+=(--data-dir "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}")
fi

"${cmd[@]}" "$@"
