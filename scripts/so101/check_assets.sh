#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

mimic_video_load_paths "${1:-${MIMIC_VIDEO_EXPERIMENT:-so101-bottle-delta30}}"
mimic_video_create_layout

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

echo "SO-101 experiment: ${MIMIC_VIDEO_EXPERIMENT}"
echo "Config: ${MIMIC_VIDEO_SO101_CONFIG_FILE}"
echo "Artifacts: ${MIMIC_VIDEO_SO101_ARTIFACT_FILE}"
echo "Mimic Video root: ${MIMIC_VIDEO_ROOT}"
echo

check_file "base video backbone" "${MIMIC_VIDEO_BACKBONE_PATH}"
check_file "video LoRA" "${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}"
if [[ "${ACTION_ASSETS_ENABLED}" == "1" ]]; then
  check_file "${ACTION_TRANSFORM} action decoder" "${MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH}"
  check_file "${ACTION_TRANSFORM} normalizer stats" "${MIMIC_VIDEO_ACTION_DECODER_STATS_PATH}"
  check_dir "action zarr" "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}"
else
  echo "SKIP    action decoder/stats/zarr checks: ACTION_ASSETS_ENABLED=0"
fi
check_recommended_dir "external 5fps hstack video data" "${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}"

if [[ "${ACTION_ASSETS_ENABLED}" == "1" && -f "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}/paths.pkl" ]]; then
  python - "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" <<'PY'
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
PY
fi

echo
echo "Eval outputs will go under: ${MIMIC_VIDEO_EVAL_OUTPUT_ROOT}"
echo "Logs will go under: ${MIMIC_VIDEO_LOG_DIR}"

exit "${missing}"
