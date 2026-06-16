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
mimic_video_prepare_runtime_env
mimic_video_activate_model_venv_if_present

prompt="${MIMIC_VIDEO_PROMPT:-pick up the bottle and place it into the container}"

cmd=(
  python eval/so101/synthetic_policy_client.py
  --host "${MIMIC_VIDEO_SO101_HOST:-127.0.0.1}"
  --port "${MIMIC_VIDEO_SO101_PORT:-8000}"
  --prompt "${prompt}"
  --num-requests "${MIMIC_VIDEO_SYNTHETIC_NUM_REQUESTS:-3}"
  --image-height "${MIMIC_VIDEO_SYNTHETIC_IMAGE_HEIGHT:-480}"
  --image-width "${MIMIC_VIDEO_SYNTHETIC_IMAGE_WIDTH:-640}"
)

if [[ "${MIMIC_VIDEO_SYNTHETIC_USE_ZERO_PROMPT_EMBEDDING:-0}" == "1" ]]; then
  cmd+=(--use-prompt-embedding)
fi

cd "${REPO_ROOT}"
"${cmd[@]}" "$@"
