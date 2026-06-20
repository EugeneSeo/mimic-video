#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-ee}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mvs_load_paths "${experiment}"
mvs_create_layout
mvs_prepare_runtime_env
mvs_activate_model_venv_if_present

python - <<'PY'
import sys

try:
    import numpy as np
    import msgpack
except ImportError as exc:
    print("ERROR: SO-101 policy server requires msgpack for cross-environment clients.", file=sys.stderr)
    print("The pickle fallback is unsafe across NumPy major versions.", file=sys.stderr)
    print("Install it in the server environment, for example:", file=sys.stderr)
    print("  cd model && source .venv/bin/activate && pip install msgpack", file=sys.stderr)
    raise SystemExit(1) from exc

print(f"SO-101 payload dependency check: numpy={np.__version__} msgpack={msgpack.__version__}")
PY

mvs_require_file "${MVS_VIDEO_BACKBONE_PATH}" "base video backbone"
mvs_require_file "${MVS_VIDEO_LORA_PATH}" "video LoRA"
mvs_require_file "${MVS_ACTION_MODEL_PATH}" "${ACTION_DECODER_KIND} action decoder"

use_stats="${MVS_SERVER_USE_STATS:-1}"
if [[ "${use_stats}" == "1" ]]; then
  mvs_require_file "${MVS_ACTION_STATS_PATH}" "${ACTION_DECODER_KIND} normalizer stats"
else
  mvs_require_dir "${MVS_SHARED_ACTION_DATA_DIR}" "action zarr"
fi

use_prompt_cache="${MVS_SERVER_USE_PROMPT_CACHE:-0}"
if [[ "${use_prompt_cache}" == "1" ]]; then
  mvs_require_file "${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}" "prompt embedding manifest"
fi

cd "${REPO_ROOT}"

cmd=(
  python eval/so101/policy_server.py
  --host "${MVS_SO101_HOST:-127.0.0.1}" \
  --port "${MVS_SO101_PORT:-8000}" \
  --experiment-name "${POLICY_EXPERIMENT_NAME}" \
  --video-model-path "${MVS_VIDEO_BACKBONE_PATH}" \
  --video-lora-path "${MVS_VIDEO_LORA_PATH}" \
  --action-model-path "${MVS_ACTION_MODEL_PATH}" \
  --num-val-episodes "${MVS_SERVER_NUM_VAL_EPISODES:-0}" \
  --num-sampling-steps "${MVS_NUM_SAMPLING_STEPS}" \
  --stop-video-denoising-step "${MVS_STOP_VIDEO_DENOISING_STEP}"
)

if [[ "${use_stats}" == "1" ]]; then
  cmd+=(--stats-path "${MVS_ACTION_STATS_PATH}")
fi

if [[ "${use_prompt_cache}" == "1" ]]; then
  cmd+=(--prompt-embedding-manifest "${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}")
fi

"${cmd[@]}" "$@"
