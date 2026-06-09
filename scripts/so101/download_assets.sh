#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-bottle-absolute5.env}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  config_env="$1"
  shift
fi
mimic_video_load_paths "${config_env}"
mimic_video_create_layout

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found. Run scripts/so101/setup_env.sh first or activate model/.venv." >&2
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

  if [[ -z "${repo}" ]]; then
    echo "ERROR: HF repo is not set for ${file}." >&2
    echo "Set it in the config env or matching scripts/so101/artifacts/*.env file." >&2
    exit 1
  fi

  echo
  echo "Downloading ${repo}:${file}"
  hf download "${repo}" "${file}" \
    --repo-type model \
    --local-dir "${local_dir}"
}

download_checkpoint() {
  local repo="$1"
  local file="$2"
  local local_dir="$3"

  if [[ -z "${repo}" ]]; then
    echo "ERROR: HF repo is not set for ${file}." >&2
    echo "Set it in the config env or matching scripts/so101/artifacts/*.env file." >&2
    exit 1
  fi

  case "${file}" in
    ""|latest|auto)
      echo
      echo "Downloading latest checkpoint candidates from ${repo}:checkpoints/model/iter_*.pt"
      hf download "${repo}" \
        --repo-type model \
        --include 'checkpoints/model/iter_*.pt' \
        --local-dir "${local_dir}"
      ;;
    *)
      download_file "${repo}" "${file}" "${local_dir}"
      ;;
  esac
}

download_prompt_embeddings() {
  local repo="$1"
  local include="$2"

  if [[ -z "${repo}" || -z "${include}" ]]; then
    return 0
  fi

  echo
  echo "Downloading prompt embeddings from ${repo}:${include}"
  hf download "${repo}" \
    --repo-type model \
    --include "${include}" \
    --local-dir "${MIMIC_VIDEO_EXPERIMENT_ROOT}"
}

download_checkpoint "${VIDEO_LORA_REPO}" "${VIDEO_LORA_FILE}" "${MIMIC_VIDEO_EXPERIMENT_VIDEO_LORA_DIR}"

if [[ "${ACTION_ASSETS_ENABLED}" != "1" ]]; then
  echo
  echo "ACTION_ASSETS_ENABLED=0; skipping action decoder and stats download."
else
  download_checkpoint \
    "${MIMIC_VIDEO_ACTION_DECODER_REPO}" \
    "${MIMIC_VIDEO_ACTION_DECODER_FILE}" \
    "${MIMIC_VIDEO_ACTION_DECODER_DIR}"
  download_file \
    "${MIMIC_VIDEO_ACTION_DECODER_REPO}" \
    "${MIMIC_VIDEO_ACTION_DECODER_STATS_FILE}" \
    "${MIMIC_VIDEO_ACTION_DECODER_DIR}"
  download_prompt_embeddings \
    "${MIMIC_VIDEO_PROMPT_EMBEDDING_REPO}" \
    "${MIMIC_VIDEO_PROMPT_EMBEDDING_FILE}"
fi

if [[ -n "${BASE_VIDEO_REPO}" ]]; then
  download_file "${BASE_VIDEO_REPO}" "${BASE_VIDEO_FILE}" "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}"
else
  echo
  echo "Base video backbone repo not set."
  echo "Download it from the original Mimic Video release with:"
  echo "  bash scripts/so101/download_base_backbone.sh"
  echo
  echo "Or place this file manually if you already have it:"
  echo "  ${MIMIC_VIDEO_BACKBONE_PATH}"
  echo
  echo "If the backbone is in a different HF repo, rerun with:"
  echo "  add BASE_VIDEO_REPO=\"<owner/repo>\" to the config env, then rerun this command."
fi

echo
echo "Checkpoint files currently present for ${MIMIC_VIDEO_EXPERIMENT}:"
find "${MIMIC_VIDEO_EXPERIMENT_CHECKPOINT_ROOT}" "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}" -type f -name '*.pt' -print | sort
