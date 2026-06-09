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

prompt="${MIMIC_VIDEO_PROMPT:-pick up the bottle and place it into the container}"
output="${MIMIC_VIDEO_PROMPT_EMBEDDING_PATH:-${MIMIC_VIDEO_PROMPT_EMBEDDING_DIR}/so101_bottle_container.npy}"
manifest="${MIMIC_VIDEO_PROMPT_EMBEDDING_MANIFEST_PATH}"

cd "${REPO_ROOT}"

python eval/so101/export_prompt_embedding.py \
  --prompt "${prompt}" \
  --output "${output}" \
  --manifest "${manifest}" \
  "$@"
