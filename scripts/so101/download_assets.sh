#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

mvs_load_paths "${1:-${MVS_EXPERIMENT:-so101-homogeneous-delta30}}"
mvs_create_layout

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found. Run scripts/so101/setup_hf_cli.sh first or activate model/.venv." >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: hf is not authenticated. Run 'hf auth login' or export HF_TOKEN first." >&2
  exit 1
fi

download_file() {
  local repo="$1"
  local file="$2"
  local local_dir="$3"

  echo
  echo "Downloading ${repo}:${file}"
  hf download "${repo}" "${file}" \
    --repo-type model \
    --local-dir "${local_dir}"
}

download_file "${VIDEO_LORA_REPO}" "${VIDEO_LORA_FILE}" "${MVS_EXP_VIDEO_LORA_DIR}"

case "${ACTION_DECODER_KIND}" in
  absolute)
    download_file "${ACTION_ABSOLUTE_REPO}" "${ACTION_ABSOLUTE_FILE}" "${MVS_EXP_ACTION_ABSOLUTE_DIR}"
    download_file "${ACTION_ABSOLUTE_REPO}" "${ACTION_ABSOLUTE_STATS_FILE}" "${MVS_EXP_ACTION_ABSOLUTE_DIR}"
    ;;
  delta_30hz)
    download_file "${ACTION_DELTA30_REPO}" "${ACTION_DELTA30_FILE}" "${MVS_EXP_ACTION_DELTA30_DIR}"
    download_file "${ACTION_DELTA30_REPO}" "${ACTION_DELTA30_STATS_FILE}" "${MVS_EXP_ACTION_DELTA30_DIR}"
    ;;
  relative)
    download_file "${ACTION_RELATIVE_REPO}" "${ACTION_RELATIVE_FILE}" "${MVS_EXP_ACTION_RELATIVE_DIR}"
    download_file "${ACTION_RELATIVE_REPO}" "${ACTION_RELATIVE_STATS_FILE}" "${MVS_EXP_ACTION_RELATIVE_DIR}"
    ;;
  *)
    echo "ERROR: unsupported ACTION_DECODER_KIND=${ACTION_DECODER_KIND}" >&2
    exit 1
    ;;
esac

if [[ "${DOWNLOAD_ABSOLUTE}" == "1" ]]; then
  download_file "${ACTION_ABSOLUTE_REPO}" "${ACTION_ABSOLUTE_FILE}" "${MVS_EXP_ACTION_ABSOLUTE_DIR}"
  download_file "${ACTION_ABSOLUTE_REPO}" "${ACTION_ABSOLUTE_STATS_FILE}" "${MVS_EXP_ACTION_ABSOLUTE_DIR}"
fi

if [[ -n "${BASE_VIDEO_REPO}" ]]; then
  download_file "${BASE_VIDEO_REPO}" "${BASE_VIDEO_FILE}" "${MVS_SHARED_VIDEO_BACKBONE_DIR}"
else
  echo
  echo "Base video backbone repo not set."
  echo "Download it from the original Mimic Video release with:"
  echo "  bash scripts/so101/download_base_backbone.sh ${MVS_EXPERIMENT}"
  echo
  echo "Or place this file manually if you already have it:"
  echo "  ${MVS_VIDEO_BACKBONE_PATH}"
  echo
  echo "If the backbone is in a different HF repo, rerun with:"
  echo "  BASE_VIDEO_REPO=<owner/repo> bash scripts/so101/download_assets.sh ${MVS_EXPERIMENT}"
fi

echo
echo "Checkpoint files currently present for ${MVS_EXPERIMENT}:"
find "${MVS_EXP_CHECKPOINT_ROOT}" "${MVS_SHARED_VIDEO_BACKBONE_DIR}" -type f -name '*.pt' -print | sort
