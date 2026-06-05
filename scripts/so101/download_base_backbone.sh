#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-${SCRATCH}/mimic_video}"
MIMIC_VIDEO_SHARED_ROOT="${MIMIC_VIDEO_SHARED_ROOT:-${MIMIC_VIDEO_ROOT}/shared}"
MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT="${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT:-${MIMIC_VIDEO_SHARED_ROOT}/checkpoints}"
MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR="${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR:-${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}/video_backbone}"
UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-bin}"
BASE_VIDEO_FILE="${BASE_VIDEO_FILE:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"
MIMIC_VIDEO_BACKBONE_PATH="${MIMIC_VIDEO_BACKBONE_PATH:-${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}/${BASE_VIDEO_FILE}}"
export REPO_ROOT SCRATCH MIMIC_VIDEO_ROOT MIMIC_VIDEO_SHARED_ROOT
export MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR
export UV_TOOL_BIN_DIR BASE_VIDEO_FILE MIMIC_VIDEO_BACKBONE_PATH
export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

if [[ $# -gt 0 ]]; then
  echo "ERROR: download_base_backbone.sh does not take an experiment name." >&2
  echo "Run it once per scratch workspace: bash scripts/so101/download_base_backbone.sh" >&2
  exit 1
fi

mkdir -p "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}"

if [[ -f "${MIMIC_VIDEO_BACKBONE_PATH}" ]]; then
  echo "Base video backbone already present:"
  echo "  ${MIMIC_VIDEO_BACKBONE_PATH}"
  exit 0
fi

echo "Downloading Bridge-finetuned Mimic Video backbone into:"
echo "  ${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}"
echo
echo "Expected backbone after download:"
echo "  ${MIMIC_VIDEO_BACKBONE_PATH}"
echo

run_repo_downloader() {
  local python_bin="$1"
  "${python_bin}" "${REPO_ROOT}/model/scripts/download_checkpoints.py" \
    --models finetuned_cosmos_bridge \
    --checkpoint-dir "${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}"
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
    --local-dir "${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}" \
    --include "text_encoder/*" \
    --include "video_backbone/tokenizer/*" \
    --include "video_backbone/v2w_bridge_lora*" \
    --include "dataset_statistics/bridge.json" \
    --include "action_decoder/w2a_bridge_v2w_bridge_lora*"
else
  echo "ERROR: need either model/.venv with huggingface_hub, system python with huggingface_hub, or hf CLI." >&2
  echo "Run scripts/so101/setup_env.sh, then ensure its uv-bin is on PATH." >&2
  exit 1
fi

if [[ ! -f "${MIMIC_VIDEO_BACKBONE_PATH}" ]]; then
  echo "ERROR: download finished, but expected backbone is still missing:" >&2
  echo "  ${MIMIC_VIDEO_BACKBONE_PATH}" >&2
  echo "Files under ${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}:" >&2
  find "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}" -maxdepth 2 -type f -print | sort >&2
  exit 1
fi

echo
echo "Base video backbone ready:"
echo "  ${MIMIC_VIDEO_BACKBONE_PATH}"
