#!/usr/bin/env bash

# Shared path/config helpers for SO-101 Mimic Video experiments.
# Source this file from scripts under scripts/so101/.

mimic_video_so101_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

mimic_video_repo_root() {
  local so101_dir
  so101_dir="$(mimic_video_so101_dir)"
  cd "${so101_dir}/../.." && pwd
}

mimic_video_latest_checkpoint() {
  local root
  for root in "$@"; do
    [[ -n "${root}" && -d "${root}" ]] || continue
    find "${root}" -type f -path '*/checkpoints/model/iter_*.pt' -print 2>/dev/null
  done | sed -n 's#.*iter_0*\([0-9][0-9]*\)\.pt$#\1\t&#p' | sort -n | tail -n 1 | cut -f 2-
}

mimic_video_resolve_checkpoint_path() {
  local file="$1"
  local artifact_dir="$2"
  shift 2

  case "${file}" in
    ""|latest|auto)
      local latest
      latest="$(mimic_video_latest_checkpoint "$@")"
      if [[ -n "${latest}" ]]; then
        printf '%s\n' "${latest}"
      else
        printf '%s\n' "${artifact_dir}/${file:-latest}"
      fi
      ;;
    /*)
      printf '%s\n' "${file}"
      ;;
    *)
      printf '%s\n' "${artifact_dir}/${file}"
      ;;
  esac
}

mimic_video_load_paths() {
  local config_ref="${1:-}"
  if [[ -z "${config_ref}" ]]; then
    config_ref="${MIMIC_VIDEO_SO101_CONFIG_FILE:-$(mimic_video_so101_dir)/experiments/so101-bottle-delta30.env}"
  fi

  local experiment
  local config_file
  if [[ -f "${config_ref}" || "${config_ref}" == */* || "${config_ref}" == *.env ]]; then
    config_file="${config_ref}"
    experiment="$(basename "${config_file}" .env)"
  else
    experiment="${config_ref}"
    config_file="$(mimic_video_so101_dir)/experiments/${experiment}.env"
  fi
  export MIMIC_VIDEO_EXPERIMENT="${experiment}"

  export REPO_ROOT="${REPO_ROOT:-$(mimic_video_repo_root)}"
  export SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
  export MIMIC_VIDEO_ROOT="${MIMIC_VIDEO_ROOT:-${SCRATCH}/mimic_video}"
  export MIMIC_VIDEO_SHARED_ROOT="${MIMIC_VIDEO_SHARED_ROOT:-${MIMIC_VIDEO_ROOT}/shared}"
  export MIMIC_VIDEO_EXPERIMENT_ROOT="${MIMIC_VIDEO_ROOT}/experiments/${MIMIC_VIDEO_EXPERIMENT}"

  local legacy_checkpoint_root="${SCRATCH}/mimic_video_checkpoints"
  if [[ -z "${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT:-}" && -d "${legacy_checkpoint_root}" ]]; then
    export MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT="${legacy_checkpoint_root}"
  else
    export MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT="${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT:-${MIMIC_VIDEO_SHARED_ROOT}/checkpoints}"
  fi
  export MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR="${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR:-${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}/video_backbone}"
  export MIMIC_VIDEO_SHARED_DATA_ROOT="${MIMIC_VIDEO_SHARED_DATA_ROOT:-${MIMIC_VIDEO_SHARED_ROOT}/data}"
  export MIMIC_VIDEO_SHARED_VIDEO_DATA_ROOT="${MIMIC_VIDEO_SHARED_VIDEO_DATA_ROOT:-${MIMIC_VIDEO_SHARED_ROOT}/video_data}"
  export MIMIC_VIDEO_SHARED_ACTION_DATA_DIR="${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR:-${MIMIC_VIDEO_SHARED_DATA_ROOT}/so101_bottle_action_only_full}"
  export MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR="${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_ROOT}/so101_bottle_front_wrist_hstack_5fps_full}"

  export MIMIC_VIDEO_EXPERIMENT_CHECKPOINT_ROOT="${MIMIC_VIDEO_EXPERIMENT_ROOT}/checkpoints"
  export MIMIC_VIDEO_EXPERIMENT_VIDEO_LORA_DIR="${MIMIC_VIDEO_EXPERIMENT_CHECKPOINT_ROOT}/video_lora"
  export MIMIC_VIDEO_DELTA_ACTION_DECODER_DIR="${MIMIC_VIDEO_EXPERIMENT_CHECKPOINT_ROOT}/action_decoder"
  export MIMIC_VIDEO_ABSOLUTE_ACTION_DECODER_DIR="${MIMIC_VIDEO_EXPERIMENT_CHECKPOINT_ROOT}/action_decoder_absolute"
  export MIMIC_VIDEO_EXPERIMENT_ACTION_DECODER_DIR="${MIMIC_VIDEO_DELTA_ACTION_DECODER_DIR}"
  export MIMIC_VIDEO_EXPERIMENT_ABSOLUTE_ACTION_DECODER_DIR="${MIMIC_VIDEO_ABSOLUTE_ACTION_DECODER_DIR}"
  export MIMIC_VIDEO_EVAL_OUTPUT_ROOT="${MIMIC_VIDEO_EXPERIMENT_ROOT}/eval_outputs"
  export MIMIC_VIDEO_LOG_DIR="${MIMIC_VIDEO_EXPERIMENT_ROOT}/logs"
  export MIMIC_VIDEO_RUNS_DIR="${MIMIC_VIDEO_EXPERIMENT_ROOT}/runs"
  export MIMIC_VIDEO_VIDEO_LORA_RUN_DIR="${MIMIC_VIDEO_RUNS_DIR}/video_lora"
  export MIMIC_VIDEO_DELTA_ACTION_DECODER_RUN_DIR="${MIMIC_VIDEO_RUNS_DIR}/action_decoder"
  export MIMIC_VIDEO_ABSOLUTE_ACTION_DECODER_RUN_DIR="${MIMIC_VIDEO_RUNS_DIR}/action_decoder_absolute"
  export MIMIC_VIDEO_ACTION_DECODER_RUN_DIR="${MIMIC_VIDEO_DELTA_ACTION_DECODER_RUN_DIR}"

  export UV_CACHE_DIR="${UV_CACHE_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-cache}"
  export UV_TOOL_DIR="${UV_TOOL_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-tools}"
  export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${MIMIC_VIDEO_SHARED_ROOT}/uv-bin}"
  export HF_HOME="${HF_HOME:-${MIMIC_VIDEO_SHARED_ROOT}/hf-home}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MIMIC_VIDEO_SHARED_ROOT}/hf-datasets-cache}"
  export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

  if [[ ! -f "${config_file}" ]]; then
    echo "ERROR: SO-101 config env not found: ${config_file}" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${config_file}"
  export MIMIC_VIDEO_SO101_CONFIG_FILE="${config_file}"

  local artifact_file="${MIMIC_VIDEO_SO101_ARTIFACT_FILE:-$(mimic_video_so101_dir)/artifacts/${MIMIC_VIDEO_EXPERIMENT}.env}"
  if [[ -f "${artifact_file}" ]]; then
    # shellcheck disable=SC1090
    source "${artifact_file}"
  fi
  export MIMIC_VIDEO_SO101_ARTIFACT_FILE="${artifact_file}"

  export ACTION_TRANSFORM="${ACTION_TRANSFORM:-delta}"
  export VIDEO_LORA_REPO="${VIDEO_LORA_REPO:-}"
  export VIDEO_LORA_FILE="${VIDEO_LORA_FILE:-latest}"
  export MIMIC_VIDEO_ACTION_DECODER_REPO="${MIMIC_VIDEO_ACTION_DECODER_REPO:-${ACTION_REPO:-}}"
  export MIMIC_VIDEO_ACTION_DECODER_FILE="${MIMIC_VIDEO_ACTION_DECODER_FILE:-${ACTION_FILE:-latest}}"
  export MIMIC_VIDEO_ACTION_DECODER_STATS_FILE="${MIMIC_VIDEO_ACTION_DECODER_STATS_FILE:-${ACTION_STATS_FILE:-}}"
  export BASE_VIDEO_REPO="${BASE_VIDEO_REPO:-}"
  export BASE_VIDEO_FILE="${BASE_VIDEO_FILE:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"
  export ACTION_ZARR_REPO="${ACTION_ZARR_REPO:-dreamdifferent/mimic-video-so101-bottle-action-zarr}"
  export SO101_DATASET_REPO="${SO101_DATASET_REPO:-dreamdifferent/so101_bottle}"
  export SO101_ACTION_DATASET_REPO="${SO101_ACTION_DATASET_REPO:-${SO101_DATASET_REPO}}"
  export VIDEO_DATA_DIR_NAME="${VIDEO_DATA_DIR_NAME:-so101_bottle_front_wrist_hstack_5fps_full}"
  export ACTION_ZARR_DIR_NAME="${ACTION_ZARR_DIR_NAME:-so101_bottle_action_only_full}"
  export SO101_VIDEO_SOURCE_FPS="${SO101_VIDEO_SOURCE_FPS:-30}"
  export SO101_VIDEO_TARGET_FPS="${SO101_VIDEO_TARGET_FPS:-5}"
  export SO101_VIEW_LAYOUT="${SO101_VIEW_LAYOUT:-hstack}"
  export SO101_ACTION_TARGET_FREQUENCY="${SO101_ACTION_TARGET_FREQUENCY:-30}"
  export SO101_ACTION_HORIZON="${SO101_ACTION_HORIZON:-90}"
  export SO101_W2A_MAX_HORIZON="${SO101_W2A_MAX_HORIZON:-91}"
  if [[ -z "${DATA_CONFIG:-}" ]]; then
    case "${ACTION_TRANSFORM}:${SO101_ACTION_TARGET_FREQUENCY}" in
      absolute:*)
        export DATA_CONFIG="so101"
        ;;
      delta:30)
        export DATA_CONFIG="so101_delta_30hz"
        ;;
      delta_gripper_absolute:30)
        export DATA_CONFIG="so101_delta_30hz_gripper_absolute"
        ;;
      delta:5)
        export DATA_CONFIG="so101_delta"
        ;;
      *)
        echo "ERROR: no SO-101 DATA_CONFIG mapping for ACTION_TRANSFORM=${ACTION_TRANSFORM} SO101_ACTION_TARGET_FREQUENCY=${SO101_ACTION_TARGET_FREQUENCY}" >&2
        return 1
        ;;
    esac
  fi
  if [[ -z "${MIMIC_VIDEO_ACTION_DECODER_STATS_FILE:-}" ]]; then
    export MIMIC_VIDEO_ACTION_DECODER_STATS_FILE="stats/${DATA_CONFIG}_stats.json"
  fi
  export ACTION_REPO="${MIMIC_VIDEO_ACTION_DECODER_REPO}"
  export ACTION_FILE="${MIMIC_VIDEO_ACTION_DECODER_FILE}"
  export ACTION_STATS_FILE="${MIMIC_VIDEO_ACTION_DECODER_STATS_FILE}"
  export ACTION_ASSETS_ENABLED="${ACTION_ASSETS_ENABLED:-1}"
  export MIMIC_VIDEO_NUM_SAMPLING_STEPS="${MIMIC_VIDEO_NUM_SAMPLING_STEPS:-35}"
  export MIMIC_VIDEO_STOP_VIDEO_DENOISING_STEP="${MIMIC_VIDEO_STOP_VIDEO_DENOISING_STEP:-10}"
  export MIMIC_VIDEO_NUM_VAL_EPISODES="${MIMIC_VIDEO_NUM_VAL_EPISODES:-10}"

  export MIMIC_VIDEO_SHARED_ACTION_DATA_DIR="${MIMIC_VIDEO_SHARED_DATA_ROOT}/${ACTION_ZARR_DIR_NAME}"
  export MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR="${MIMIC_VIDEO_SHARED_VIDEO_DATA_ROOT}/${VIDEO_DATA_DIR_NAME}"
  export MIMIC_VIDEO_BACKBONE_PATH="${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}/${BASE_VIDEO_FILE}"
  export MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH
  MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH="$(mimic_video_resolve_checkpoint_path \
    "${VIDEO_LORA_FILE}" \
    "${MIMIC_VIDEO_EXPERIMENT_VIDEO_LORA_DIR}" \
    "${MIMIC_VIDEO_VIDEO_LORA_RUN_DIR}/${VIDEO_LORA_EXPERIMENT_NAME:-}" \
    "${MIMIC_VIDEO_VIDEO_LORA_RUN_DIR}" \
    "${MIMIC_VIDEO_EXPERIMENT_VIDEO_LORA_DIR}")"
  export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}}"
  case "${ACTION_TRANSFORM}" in
    delta|delta_gripper_absolute)
      export MIMIC_VIDEO_ACTION_DECODER_DIR="${MIMIC_VIDEO_DELTA_ACTION_DECODER_DIR}"
      export MIMIC_VIDEO_ACTION_DECODER_RUN_DIR="${MIMIC_VIDEO_DELTA_ACTION_DECODER_RUN_DIR}"
      ;;
    absolute)
      export MIMIC_VIDEO_ACTION_DECODER_DIR="${MIMIC_VIDEO_ABSOLUTE_ACTION_DECODER_DIR}"
      export MIMIC_VIDEO_ACTION_DECODER_RUN_DIR="${MIMIC_VIDEO_ABSOLUTE_ACTION_DECODER_RUN_DIR}"
      ;;
    *)
      echo "ERROR: unsupported ACTION_TRANSFORM=${ACTION_TRANSFORM}. Supported: delta, delta_gripper_absolute, absolute" >&2
      return 1
      ;;
  esac
  export MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH
  MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH="$(mimic_video_resolve_checkpoint_path \
    "${MIMIC_VIDEO_ACTION_DECODER_FILE}" \
    "${MIMIC_VIDEO_ACTION_DECODER_DIR}" \
    "${MIMIC_VIDEO_ACTION_DECODER_RUN_DIR}/${POLICY_EXPERIMENT_NAME:-}" \
    "${MIMIC_VIDEO_ACTION_DECODER_RUN_DIR}" \
    "${MIMIC_VIDEO_ACTION_DECODER_DIR}")"
  export MIMIC_VIDEO_ACTION_DECODER_STATS_PATH="${MIMIC_VIDEO_ACTION_DECODER_DIR}/${MIMIC_VIDEO_ACTION_DECODER_STATS_FILE}"
  if [[ "${ACTION_ASSETS_ENABLED}" != "1" ]]; then
    export MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH=""
    export MIMIC_VIDEO_ACTION_DECODER_STATS_PATH=""
  fi
}

mimic_video_create_layout() {
  mkdir -p \
    "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}" \
    "${MIMIC_VIDEO_SHARED_DATA_ROOT}" \
    "${MIMIC_VIDEO_SHARED_VIDEO_DATA_ROOT}" \
    "${MIMIC_VIDEO_EXPERIMENT_VIDEO_LORA_DIR}" \
    "${MIMIC_VIDEO_EXPERIMENT_ACTION_DECODER_DIR}" \
    "${MIMIC_VIDEO_EXPERIMENT_ABSOLUTE_ACTION_DECODER_DIR}" \
    "${MIMIC_VIDEO_EVAL_OUTPUT_ROOT}" \
    "${MIMIC_VIDEO_LOG_DIR}" \
    "${MIMIC_VIDEO_RUNS_DIR}" \
    "${UV_CACHE_DIR}" \
    "${UV_TOOL_DIR}" \
    "${UV_TOOL_BIN_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}"
}

mimic_video_prepare_runtime_env() {
  export PATH="/usr/sbin:/sbin:${UV_TOOL_BIN_DIR}:${PATH}"
  export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
  export MIMIC_DATASET_STATS_NUM_WORKERS="${MIMIC_DATASET_STATS_NUM_WORKERS:-0}"
  export MIMIC_DATASET_STATS_BATCH_SIZE="${MIMIC_DATASET_STATS_BATCH_SIZE:-4}"
  export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
  export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-${SO101_VIDEO_TARGET_FPS}}"

  local nvidia_dir="${REPO_ROOT}/model/.venv/lib/python3.10/site-packages/nvidia"
  if [[ -d "${nvidia_dir}" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${nvidia_dir}}"
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    export LD_LIBRARY_PATH="${nvidia_dir}/cuda_runtime/lib:${nvidia_dir}/cuda_nvrtc/lib:${nvidia_dir}/cudnn/lib:${nvidia_dir}/nccl/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mimic_video_activate_model_venv_if_present() {
  if [[ -f "${REPO_ROOT}/model/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/model/.venv/bin/activate"
  fi
}

mimic_video_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
}

mimic_video_require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
}
