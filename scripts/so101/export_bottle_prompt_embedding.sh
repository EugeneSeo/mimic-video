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
mvs_prepare_runtime_env
mvs_activate_model_venv_if_present

prompt="${MVS_PROMPT:-pick up the bottle and place it into the container}"
output="${MVS_PROMPT_EMBEDDING_PATH:-${MVS_EXP_ROOT}/prompt_embeddings/so101_bottle_container.npy}"
manifest="${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}"

cd "${REPO_ROOT}"

python eval/so101/export_prompt_embedding.py \
  --prompt "${prompt}" \
  --output "${output}" \
  --manifest "${manifest}" \
  "$@"
