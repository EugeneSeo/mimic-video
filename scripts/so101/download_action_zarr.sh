#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-homogeneous-delta30}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mvs_load_paths "${experiment}"
mvs_create_layout

if [[ -d "${MVS_SHARED_ACTION_DATA_DIR}" ]]; then
  echo "Action zarr already present:"
  echo "  ${MVS_SHARED_ACTION_DATA_DIR}"
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found. Run scripts/so101/setup_hf_cli.sh first or activate model/.venv." >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: hf is not authenticated. Run 'hf auth login' or export HF_TOKEN first." >&2
  exit 1
fi

echo "Downloading action zarr dataset:"
echo "  repo: ${ACTION_ZARR_REPO}"
echo "  local root: ${MVS_SHARED_DATA_ROOT}"
echo "  expected dir: ${MVS_SHARED_ACTION_DATA_DIR}"
echo

hf download "${ACTION_ZARR_REPO}" \
  --repo-type dataset \
  --local-dir "${MVS_SHARED_DATA_ROOT}" \
  "$@"

if [[ ! -d "${MVS_SHARED_ACTION_DATA_DIR}" ]]; then
  echo "ERROR: download finished, but expected action zarr dir is missing:" >&2
  echo "  ${MVS_SHARED_ACTION_DATA_DIR}" >&2
  echo "Downloaded top-level entries under ${MVS_SHARED_DATA_ROOT}:" >&2
  find "${MVS_SHARED_DATA_ROOT}" -maxdepth 2 -type d -print | sort >&2
  exit 1
fi

if ! find "${MVS_SHARED_ACTION_DATA_DIR}" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "WARN: action zarr dir exists, but no .zarr episodes were found within maxdepth 2:" >&2
  echo "  ${MVS_SHARED_ACTION_DATA_DIR}" >&2
fi

"${SCRIPT_DIR}/repair_action_zarr_paths.sh" "${MVS_EXPERIMENT}"

echo
echo "Action zarr ready:"
echo "  ${MVS_SHARED_ACTION_DATA_DIR}"
