#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/so101/preprocess_data.sh config.env [--target video|action|all] [sbatch args...]

Config:
  Put dataset, output, timing, and checkpoint/HF settings in config.env.
  The wrapper loads config.env and forwards the resolved values to sbatch.

Examples:
  DRY_RUN=1 scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-bottle-delta30.env --target all
EOF
}

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-bottle-delta30.env}"
target="${PREPROCESS_TARGET:-all}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  config_env="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="$2"
      shift 2
      ;;
    --target=*)
      target="${1#--target=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

case "${target}" in
  video|action|all) ;;
  *)
    echo "ERROR: --target must be video, action, or all; got ${target}" >&2
    exit 1
    ;;
esac

mimic_video_load_paths "${config_env}"
mimic_video_create_layout

run_sbatch_step() {
  local label="$1"
  shift
  {
    echo
    echo "===== ${label} ====="
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
  } >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; not submitting." >&2
    return 0
  fi
  local output
  output="$("$@")"
  echo "${output}" >&2
  awk '/Submitted batch job/ {print $4}' <<<"${output}" | tail -1
}

common_sbatch_args=(
  --output="${MIMIC_VIDEO_LOG_DIR}/%x-%j.out"
  --error="${MIMIC_VIDEO_LOG_DIR}/%x-%j.err"
  "$@"
)

video_t5_dependency=()
if [[ "${target}" == "video" || "${target}" == "all" ]]; then
  export REPO_ROOT
  export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
  export REPO_ID="${REPO_ID:-${SO101_DATASET_REPO}}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
  export MAX_EPISODES="${MAX_EPISODES-}"
  export SOURCE_FPS="${SOURCE_FPS:-${SO101_VIDEO_SOURCE_FPS}}"
  export TARGET_FPS="${TARGET_FPS:-${SO101_VIDEO_TARGET_FPS}}"
  export VIEW_LAYOUT="${VIEW_LAYOUT:-${SO101_VIEW_LAYOUT}}"
  export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
  export LOG_DIR="${MIMIC_VIDEO_LOG_DIR}"

  video_job_id="$(
    run_sbatch_step \
      "SO-101 two-camera hstack video conversion" \
      sbatch "${common_sbatch_args[@]}" "${REPO_ROOT}/model/scripts/convert_so101_two_camera_video.sbatch"
  )"
  video_job_id="$(tail -1 <<<"${video_job_id}")"
  if [[ -n "${video_job_id}" && "${DRY_RUN:-0}" != "1" ]]; then
    video_t5_dependency=(--dependency="afterok:${video_job_id}")
  fi

  export DATASET_PATH="${DATASET_PATH:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
  run_sbatch_step \
    "SO-101 video T5 precompute" \
    sbatch "${video_t5_dependency[@]}" "${common_sbatch_args[@]}" "${REPO_ROOT}/model/scripts/precompute_so101_two_camera_video_t5.sbatch"
fi

action_t5_dependency=()
if [[ "${target}" == "action" || "${target}" == "all" ]]; then
  export REPO_ROOT
  export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
  export REPO_ID="${ACTION_REPO_ID:-${SO101_ACTION_DATASET_REPO}}"
  export OUTPUT_DIR="${ACTION_OUTPUT_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
  export MAX_EPISODES="${ACTION_MAX_EPISODES-}"
  export SKIP_VIDEO="${SKIP_VIDEO:-1}"
  export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
  export OVERWRITE="${OVERWRITE:-0}"
  export LOG_DIR="${MIMIC_VIDEO_LOG_DIR}"

  action_job_id="$(
    run_sbatch_step \
      "SO-101 action zarr conversion" \
      sbatch "${common_sbatch_args[@]}" "${REPO_ROOT}/model/scripts/convert_so101_smoke.sbatch"
  )"
  action_job_id="$(tail -1 <<<"${action_job_id}")"
  if [[ -n "${action_job_id}" && "${DRY_RUN:-0}" != "1" ]]; then
    action_t5_dependency=(--dependency="afterok:${action_job_id}")
  fi

  export DATA_DIR="${DATA_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
  run_sbatch_step \
    "SO-101 action T5 precompute" \
    sbatch "${action_t5_dependency[@]}" "${common_sbatch_args[@]}" "${REPO_ROOT}/model/scripts/precompute_so101_t5_smoke.sbatch"
fi
