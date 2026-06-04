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

if [[ -f "${MVS_VIDEO_BACKBONE_PATH}" ]]; then
  echo "Base video backbone already present:"
  echo "  ${MVS_VIDEO_BACKBONE_PATH}"
  exit 0
fi

echo "Downloading Bridge-finetuned Mimic Video backbone into:"
echo "  ${MVS_SHARED_CHECKPOINT_ROOT}"
echo
echo "Expected backbone after download:"
echo "  ${MVS_VIDEO_BACKBONE_PATH}"
echo

run_repo_downloader() {
  local python_bin="$1"
  "${python_bin}" "${REPO_ROOT}/model/scripts/download_checkpoints.py" \
    --models finetuned_cosmos_bridge \
    --checkpoint-dir "${MVS_SHARED_CHECKPOINT_ROOT}"
}

can_import_hf() {
  local python_bin="$1"
  "${python_bin}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
}

if [[ -x "${REPO_ROOT}/model/.venv/bin/python" ]] && can_import_hf "${REPO_ROOT}/model/.venv/bin/python"; then
  run_repo_downloader "${REPO_ROOT}/model/.venv/bin/python"
elif command -v python >/dev/null 2>&1 && can_import_hf python; then
  run_repo_downloader python
elif command -v python3 >/dev/null 2>&1 && can_import_hf python3; then
  run_repo_downloader python3
elif command -v hf >/dev/null 2>&1; then
  echo "No Python with huggingface_hub found; falling back to hf CLI."
  hf download jonpai/mimic-video \
    --repo-type model \
    --local-dir "${MVS_SHARED_CHECKPOINT_ROOT}" \
    --include "text_encoder/*" \
    --include "video_backbone/tokenizer/*" \
    --include "video_backbone/v2w_bridge_lora*" \
    --include "dataset_statistics/bridge.json" \
    --include "action_decoder/w2a_bridge_v2w_bridge_lora*"
else
  echo "ERROR: need either model/.venv with huggingface_hub, system python with huggingface_hub, or hf CLI." >&2
  echo "Run scripts/so101/setup_hf_cli.sh ${MVS_EXPERIMENT}, then ensure its uv-bin is on PATH." >&2
  exit 1
fi

if [[ ! -f "${MVS_VIDEO_BACKBONE_PATH}" ]]; then
  echo "ERROR: download finished, but expected backbone is still missing:" >&2
  echo "  ${MVS_VIDEO_BACKBONE_PATH}" >&2
  echo "Files under ${MVS_SHARED_VIDEO_BACKBONE_DIR}:" >&2
  find "${MVS_SHARED_VIDEO_BACKBONE_DIR}" -maxdepth 2 -type f -print | sort >&2
  exit 1
fi

echo
echo "Base video backbone ready:"
echo "  ${MVS_VIDEO_BACKBONE_PATH}"
