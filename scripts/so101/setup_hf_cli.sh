#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

mvs_load_paths "${1:-${MVS_EXPERIMENT:-so101-homogeneous-rel}}"
mvs_create_layout

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
echo "HF CLI setup complete."
echo "Experiment: ${MVS_EXPERIMENT}"
echo "MVS root: ${MVS_ROOT}"
echo
echo "For this shell, run:"
echo "  export SCRATCH=${SCRATCH}"
echo "  export MVS_EXPERIMENT=${MVS_EXPERIMENT}"
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
