#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

mvs_load_paths "${1:-${MVS_EXPERIMENT:-so101-homogeneous-delta30}}"
mvs_create_layout

missing=0

check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "OK      ${label}: ${path}"
  else
    echo "MISSING ${label}: ${path}"
    missing=1
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    echo "OK      ${label}: ${path}"
  else
    echo "MISSING ${label}: ${path}"
    missing=1
  fi
}

check_recommended_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    echo "OK      ${label}: ${path}"
  else
    echo "WARN    ${label}: ${path}"
  fi
}

echo "SO-101 experiment: ${MVS_EXPERIMENT}"
echo "Config: ${MVS_SO101_CONFIG_FILE}"
echo "MVS root: ${MVS_ROOT}"
echo

check_file "base video backbone" "${MVS_VIDEO_BACKBONE_PATH}"
check_file "video LoRA" "${MVS_VIDEO_LORA_PATH}"
check_file "${ACTION_DECODER_KIND} action decoder" "${MVS_ACTION_MODEL_PATH}"
check_file "${ACTION_DECODER_KIND} normalizer stats" "${MVS_ACTION_STATS_PATH}"
check_dir "action zarr" "${MVS_SHARED_ACTION_DATA_DIR}"
check_recommended_dir "external 5fps hstack video data" "${MVS_SHARED_VIDEO_DATA_DIR}"

if [[ -f "${MVS_SHARED_ACTION_DATA_DIR}/paths.pkl" ]]; then
  python - "${MVS_SHARED_ACTION_DATA_DIR}" <<'PY'
import pathlib
import pickle
import sys

root = pathlib.Path(sys.argv[1]).resolve()
paths_cache = root / "paths.pkl"
try:
    paths = pickle.load(paths_cache.open("rb"))
except Exception as exc:
    print(f"WARN    action zarr paths cache unreadable: {paths_cache} ({exc})")
    raise SystemExit(0)

bad = [path for path in paths if not pathlib.Path(path).exists()]
if bad:
    print(f"WARN    action zarr paths cache has {len(bad)} stale path(s): {paths_cache}")
    print("        repair with: bash scripts/so101/repair_action_zarr_paths.sh <experiment>")
PY
fi

echo
echo "Eval outputs will go under: ${MVS_EVAL_OUTPUT_ROOT}"
echo "Logs will go under: ${MVS_LOG_DIR}"

exit "${missing}"
