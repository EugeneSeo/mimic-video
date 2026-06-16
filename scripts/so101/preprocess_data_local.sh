#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/so101/preprocess_data_local.sh config.env [--target video|action|all] [--skip-t5]

Runs SO-101 preprocessing on a local/non-Slurm machine. This is the local
equivalent of scripts/so101/preprocess_data.sh, but it executes the steps
sequentially instead of submitting sbatch jobs.

Examples:
  DRY_RUN=1 MIMIC_VIDEO_ROOT=/workspace/mimic_video \
    scripts/so101/preprocess_data_local.sh scripts/so101/experiments/so101-multi-object-delta30.env --target all

  MIMIC_VIDEO_ROOT=/workspace/mimic_video \
    scripts/so101/preprocess_data_local.sh scripts/so101/experiments/so101-multi-object-delta30.env --target action --skip-t5
EOF
}

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-bottle-delta30.env}"
target="${PREPROCESS_TARGET:-all}"
skip_t5="${SKIP_T5:-0}"

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
    --skip-t5)
      skip_t5=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unexpected argument: $1" >&2
      usage >&2
      exit 1
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

# Non-cluster GPU rentals usually do not have /cluster/scratch.
export SCRATCH="${SCRATCH:-${HOME}}"

mimic_video_load_paths "${config_env}"
mimic_video_create_layout
mimic_video_prepare_runtime_env

MODEL_DIR="${REPO_ROOT}/model"
LOG_DIR="${LOG_DIR:-${MIMIC_VIDEO_LOG_DIR}}"
RUN_TMP_ROOT="${RUN_TMP_ROOT:-/tmp/mimic_video_preprocess_local}"

mkdir -p \
  "${LOG_DIR}" \
  "${RUN_TMP_ROOT}/uv-cache" \
  "${RUN_TMP_ROOT}/torch-cache"

export TMPDIR="${RUN_TMP_ROOT}"
export UV_CACHE_DIR="${RUN_TMP_ROOT}/uv-cache"
export TORCH_HOME="${RUN_TMP_ROOT}/torch-cache"
export PYTHONUNBUFFERED=1
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}}"

run_cmd() {
  local label="$1"
  shift

  echo
  echo "===== ${label} ====="
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; not running."
    return 0
  fi

  "$@"
}

run_python() {
  local label="$1"
  shift

  if [[ -x "${MODEL_DIR}/.venv/bin/python" ]]; then
    (
      # shellcheck disable=SC1091
      source "${MODEL_DIR}/.venv/bin/activate"
      run_cmd "${label}" "$@"
    )
  elif command -v uv >/dev/null 2>&1; then
    (
      cd "${MODEL_DIR}"
      run_cmd "${label}" uv run --extra cu126 "$@"
    )
  else
    echo "ERROR: neither model/.venv nor uv is available." >&2
    echo "Set up the environment first:" >&2
    echo "  cd ${MODEL_DIR} && uv sync --extra cu126" >&2
    exit 1
  fi
}

echo "===== SO-101 local preprocessing ====="
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Model dir: ${MODEL_DIR}"
echo "Config: ${MIMIC_VIDEO_SO101_CONFIG_FILE}"
echo "Target: ${target}"
echo "Skip T5: ${skip_t5}"
echo "MIMIC_VIDEO_ROOT: ${MIMIC_VIDEO_ROOT}"
echo "Video repo: ${SO101_DATASET_REPO}"
echo "Action repo: ${SO101_ACTION_DATASET_REPO}"
echo "Video output: ${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}"
echo "Action output: ${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "COSMOS_PREDICT2_ARGS: ${COSMOS_PREDICT2_ARGS}"
echo "======================================"
nvidia-smi || true

if [[ "${target}" == "video" || "${target}" == "all" ]]; then
  export REPO_ID="${REPO_ID:-${SO101_DATASET_REPO}}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
  export MAX_EPISODES="${MAX_EPISODES-}"
  export SOURCE_FPS="${SOURCE_FPS:-${SO101_VIDEO_SOURCE_FPS}}"
  export TARGET_FPS="${TARGET_FPS:-${SO101_VIDEO_TARGET_FPS}}"
  export VIEW_LAYOUT="${VIEW_LAYOUT:-${SO101_VIEW_LAYOUT}}"
  export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
  export OVERWRITE="${OVERWRITE:-0}"

  if [[ -e "${OUTPUT_DIR}" && "${OVERWRITE}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Removing existing video OUTPUT_DIR because OVERWRITE=1: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  fi
  if [[ -e "${OUTPUT_DIR}" && "${OVERWRITE}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "ERROR: video OUTPUT_DIR exists: ${OUTPUT_DIR}" >&2
    echo "Set OVERWRITE=1 or choose a new OUTPUT_DIR." >&2
    exit 1
  fi

  video_cmd=(
    python "${REPO_ROOT}/data_preprocessing/video/process_so101_two_camera_video.py"
    --repo-id "${REPO_ID}"
    --output-dir "${OUTPUT_DIR}"
    --video-backend "${VIDEO_BACKEND}"
    --source-fps "${SOURCE_FPS}"
    --target-fps "${TARGET_FPS}"
    --view-layout "${VIEW_LAYOUT}"
    --overwrite
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    video_cmd+=(--max-episodes "${MAX_EPISODES}")
  fi
  run_python "SO-101 two-camera hstack video conversion" "${video_cmd[@]}"

  export DATASET_PATH="${DATASET_PATH:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if ! find "${DATASET_PATH}/video" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
      echo "ERROR: no mp4 videos found under ${DATASET_PATH}/video" >&2
      exit 1
    fi
  fi

  if [[ "${skip_t5}" == "1" ]]; then
    echo
    echo "Skipping SO-101 video T5 precompute because --skip-t5/SKIP_T5=1."
  else
    run_python \
      "SO-101 video T5 precompute" \
      python "${REPO_ROOT}/data_preprocessing/video/get_t5_embeddings.py" \
        --dataset_path "${DATASET_PATH}"

    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      find "${DATASET_PATH}/t5_xxl" -maxdepth 1 -name '*.pickle' -type f | wc -l
    fi
  fi
fi

if [[ "${target}" == "action" || "${target}" == "all" ]]; then
  export REPO_ID="${ACTION_REPO_ID:-${SO101_ACTION_DATASET_REPO}}"
  export OUTPUT_DIR="${ACTION_OUTPUT_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
  export MAX_EPISODES="${ACTION_MAX_EPISODES-}"
  export SKIP_VIDEO="${SKIP_VIDEO:-1}"
  export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
  export OVERWRITE="${OVERWRITE:-0}"
  export ACTION_NUM_WORKERS="${ACTION_NUM_WORKERS:-1}"

  if [[ -e "${OUTPUT_DIR}" && "${OVERWRITE}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Removing existing action OUTPUT_DIR because OVERWRITE=1: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  fi
  if [[ -e "${OUTPUT_DIR}" && "${OVERWRITE}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "ERROR: action OUTPUT_DIR exists: ${OUTPUT_DIR}" >&2
    echo "Set OVERWRITE=1 or choose a new OUTPUT_DIR." >&2
    exit 1
  fi

  action_converter="${REPO_ROOT}/data_preprocessing/action/process_so101_lerobot.py"
  if (( ACTION_NUM_WORKERS > 1 )); then
    action_converter="${REPO_ROOT}/data_preprocessing/action/process_so101_lerobot_parallel.py"
  fi

  action_cmd=(
    python "${action_converter}"
    --repo-id "${REPO_ID}"
    --output-dir "${OUTPUT_DIR}"
    --video-backend "${VIDEO_BACKEND}"
  )
  if (( ACTION_NUM_WORKERS > 1 )); then
    action_cmd+=(--num-workers "${ACTION_NUM_WORKERS}")
  fi
  if [[ "${SKIP_VIDEO}" == "1" ]]; then
    action_cmd+=(--skip-video)
  fi
  if [[ -n "${MAX_EPISODES}" ]]; then
    action_cmd+=(--max-episodes "${MAX_EPISODES}")
  fi
  run_python "SO-101 action zarr conversion" "${action_cmd[@]}"

  if [[ "${skip_t5}" == "1" ]]; then
    echo
    echo "Skipping SO-101 action T5 precompute because --skip-t5/SKIP_T5=1."
  else
    export DATA_DIR="${DATA_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      if ! find "${DATA_DIR}" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
        echo "ERROR: DATA_DIR contains no .zarr episodes: ${DATA_DIR}" >&2
        exit 1
      fi
      t5_dir="${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}/text_encoder/t5-11b"
      if [[ ! -d "${t5_dir}" ]]; then
        echo "ERROR: T5 checkpoint directory is missing: ${t5_dir}" >&2
        echo "Download/symlink checkpoints before running action T5 precompute." >&2
        exit 1
      fi
    fi

    run_python \
      "SO-101 action T5 precompute" \
      python "${REPO_ROOT}/data_preprocessing/action/precompute_t5.py" \
        --dataset-path "${DATA_DIR}"
  fi
fi

echo
echo "Finished SO-101 local preprocessing target=${target}"
