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
mvs_prepare_runtime_env
mvs_activate_model_venv_if_present

cd "${REPO_ROOT}"

python eval/so101/synthetic_policy_client.py \
  --host "${MVS_SO101_HOST:-127.0.0.1}" \
  --port "${MVS_SO101_PORT:-8000}" \
  --use-prompt-embedding \
  --num-requests "${MVS_SYNTHETIC_NUM_REQUESTS:-3}" \
  --image-height "${MVS_SYNTHETIC_IMAGE_HEIGHT:-480}" \
  --image-width "${MVS_SYNTHETIC_IMAGE_WIDTH:-640}" \
  "$@"
