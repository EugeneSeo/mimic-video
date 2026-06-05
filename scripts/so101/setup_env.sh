#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/so101/setup_env.sh

Sets up shared SO-101 tool/cache directories only.
Choose an experiment when running preprocess/train/render/eval wrappers.
EOF
}

case "${1:-}" in
  "")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: setup_env.sh does not take an experiment name." >&2
    echo "Choose the experiment when running workflow commands instead." >&2
    echo "Example: bash scripts/so101/preprocess_data.sh so101-bottle-delta30 --target all" >&2
    exit 1
    ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-${SCRATCH}/mimic_video}"
MIMIC_VIDEO_SHARED_ROOT="${MIMIC_VIDEO_SHARED_ROOT:-${MIMIC_VIDEO_ROOT}/shared}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-cache}"
UV_TOOL_DIR="${UV_TOOL_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-tools}"
UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-bin}"
HF_HOME="${HF_HOME:-${MIMIC_VIDEO_SHARED_ROOT}/hf-home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MIMIC_VIDEO_SHARED_ROOT}/hf-datasets-cache}"
export REPO_ROOT SCRATCH MIMIC_VIDEO_ROOT MIMIC_VIDEO_SHARED_ROOT
export UV_CACHE_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

mkdir -p \
  "${MIMIC_VIDEO_SHARED_ROOT}" \
  "${MIMIC_VIDEO_SHARED_ROOT}/checkpoints" \
  "${MIMIC_VIDEO_SHARED_ROOT}/data" \
  "${MIMIC_VIDEO_SHARED_ROOT}/video_data" \
  "${UV_CACHE_DIR}" \
  "${UV_TOOL_DIR}" \
  "${UV_TOOL_BIN_DIR}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found in PATH." >&2
  echo "Install uv first or load the cluster module that provides it." >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Installing Hugging Face CLI into ${UV_TOOL_BIN_DIR}"
  uv tool install huggingface_hub
else
  echo "Found existing hf CLI: $(command -v hf)"
fi

echo
echo "SO-101 environment setup complete."
echo "Mimic Video root: ${MIMIC_VIDEO_ROOT}"
echo
echo "For this shell, run:"
echo "  export SCRATCH=${SCRATCH}"
echo "  export UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "  export UV_TOOL_DIR=${UV_TOOL_DIR}"
echo "  export UV_TOOL_BIN_DIR=${UV_TOOL_BIN_DIR}"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "  export PATH=${UV_TOOL_BIN_DIR}:\$PATH"
echo
echo "Then authenticate if needed:"
echo "  hf auth login"
echo
echo "Or use an existing token:"
echo "  export HF_TOKEN=<your-token>"
