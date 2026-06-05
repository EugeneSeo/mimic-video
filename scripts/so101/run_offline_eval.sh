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
mvs_prepare_runtime_env
mvs_activate_model_venv_if_present

mvs_require_file "${MVS_VIDEO_BACKBONE_PATH}" "base video backbone"
mvs_require_file "${MVS_VIDEO_LORA_PATH}" "video LoRA"
mvs_require_file "${MVS_ACTION_MODEL_PATH}" "${ACTION_DECODER_KIND} action decoder"
mvs_require_dir "${MVS_SHARED_ACTION_DATA_DIR}" "action zarr"

split="${MVS_EVAL_SPLIT:-val}"
mode="${MVS_EVAL_MODE:-generated}"
samples="${MVS_EVAL_NUM_SAMPLES:-4}"
num_val_episodes="${MVS_EVAL_NUM_VAL_EPISODES:-${MVS_NUM_VAL_EPISODES}}"
dump_videos="${MVS_DUMP_VIDEOS:-0}"
output_name="${mode}_${split}_${samples}_${MVS_NUM_SAMPLING_STEPS}_${MVS_STOP_VIDEO_DENOISING_STEP}"
if [[ "${dump_videos}" == "1" ]]; then
  output_name="${output_name}_video"
fi
output_dir="${MVS_EVAL_OUTPUT_DIR:-${MVS_EVAL_OUTPUT_ROOT}/${output_name}}"

cmd=(
  python eval/so101/offline_eval.py
  --split "${split}"
  --mode "${mode}"
  --num-samples "${samples}"
  --num-val-episodes "${num_val_episodes}"
  --experiment-name "${POLICY_EXPERIMENT_NAME}"
  --video-model-path "${MVS_VIDEO_BACKBONE_PATH}"
  --video-lora-path "${MVS_VIDEO_LORA_PATH}"
  --action-model-path "${MVS_ACTION_MODEL_PATH}"
  --data-dir "${MVS_SHARED_ACTION_DATA_DIR}"
  --num-sampling-steps "${MVS_NUM_SAMPLING_STEPS}"
  --stop-video-denoising-step "${MVS_STOP_VIDEO_DENOISING_STEP}"
  --output-dir "${output_dir}"
)

if [[ "${dump_videos}" == "1" ]]; then
  cmd+=(--dump-videos)
fi

cd "${REPO_ROOT}"
"${cmd[@]}" "$@"
