#!/usr/bin/env bash

# Shared path/config helpers for SO-101 Mimic Video experiments.
# Source this file from scripts under scripts/so101/.

mvs_so101_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

mvs_repo_root() {
  local so101_dir
  so101_dir="$(mvs_so101_dir)"
  cd "${so101_dir}/../.." && pwd
}

mvs_load_paths() {
  local experiment="${1:-${MVS_EXPERIMENT:-so101-homogeneous-rel}}"
  export MVS_EXPERIMENT="${experiment}"

  export REPO_ROOT="${REPO_ROOT:-$(mvs_repo_root)}"
  export SCRATCH="${SCRATCH:-/cluster/scratch/${USER}}"
  export MVS_ROOT="${MVS_ROOT:-${SCRATCH}/mimic_video}"
  export MVS_SHARED_ROOT="${MVS_SHARED_ROOT:-${MVS_ROOT}/shared}"
  export MVS_EXP_ROOT="${MVS_EXP_ROOT:-${MVS_ROOT}/experiments/${MVS_EXPERIMENT}}"

  export MVS_SHARED_CHECKPOINT_ROOT="${MVS_SHARED_CHECKPOINT_ROOT:-${MVS_SHARED_ROOT}/checkpoints}"
  export MVS_SHARED_VIDEO_BACKBONE_DIR="${MVS_SHARED_VIDEO_BACKBONE_DIR:-${MVS_SHARED_CHECKPOINT_ROOT}/video_backbone}"
  export MVS_SHARED_DATA_ROOT="${MVS_SHARED_DATA_ROOT:-${MVS_SHARED_ROOT}/data}"
  export MVS_SHARED_VIDEO_DATA_ROOT="${MVS_SHARED_VIDEO_DATA_ROOT:-${MVS_SHARED_ROOT}/video_data}"
  export MVS_SHARED_ACTION_DATA_DIR="${MVS_SHARED_ACTION_DATA_DIR:-${MVS_SHARED_DATA_ROOT}/so101_bottle_action_only_full}"
  export MVS_SHARED_VIDEO_DATA_DIR="${MVS_SHARED_VIDEO_DATA_DIR:-${MVS_SHARED_VIDEO_DATA_ROOT}/so101_bottle_front_wrist_hstack_5fps_full}"

  export MVS_EXP_CHECKPOINT_ROOT="${MVS_EXP_CHECKPOINT_ROOT:-${MVS_EXP_ROOT}/checkpoints}"
  export MVS_EXP_VIDEO_LORA_DIR="${MVS_EXP_VIDEO_LORA_DIR:-${MVS_EXP_CHECKPOINT_ROOT}/video_lora}"
  export MVS_EXP_ACTION_RELATIVE_DIR="${MVS_EXP_ACTION_RELATIVE_DIR:-${MVS_EXP_CHECKPOINT_ROOT}/action_decoder_relative}"
  export MVS_EXP_ACTION_ABSOLUTE_DIR="${MVS_EXP_ACTION_ABSOLUTE_DIR:-${MVS_EXP_CHECKPOINT_ROOT}/action_decoder_absolute}"
  export MVS_EVAL_OUTPUT_ROOT="${MVS_EVAL_OUTPUT_ROOT:-${MVS_EXP_ROOT}/eval_outputs}"
  export MVS_LOG_DIR="${MVS_LOG_DIR:-${MVS_EXP_ROOT}/logs}"
  export MVS_RUNS_DIR="${MVS_RUNS_DIR:-${MVS_EXP_ROOT}/runs}"

  export UV_CACHE_DIR="${UV_CACHE_DIR:-${MVS_SHARED_ROOT}/uv-cache}"
  export UV_TOOL_DIR="${UV_TOOL_DIR:-${MVS_SHARED_ROOT}/uv-tools}"
  export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${MVS_SHARED_ROOT}/uv-bin}"
  export HF_HOME="${HF_HOME:-${MVS_SHARED_ROOT}/hf-home}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MVS_SHARED_ROOT}/hf-datasets-cache}"
  export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

  local config_file="${MVS_SO101_CONFIG_FILE:-$(mvs_so101_dir)/experiments/${MVS_EXPERIMENT}.env}"
  if [[ ! -f "${config_file}" ]]; then
    echo "ERROR: SO-101 experiment config not found: ${config_file}" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${config_file}"
  export MVS_SO101_CONFIG_FILE="${config_file}"

  export ACTION_DECODER_KIND="${ACTION_DECODER_KIND:-relative}"
  export VIDEO_LORA_REPO="${VIDEO_LORA_REPO:-dreamdifferent/mimic-video-so101-2cam-hstack-5fps-v2w-lora}"
  export ACTION_RELATIVE_REPO="${ACTION_RELATIVE_REPO:-dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-relative-action-decoder}"
  export ACTION_ABSOLUTE_REPO="${ACTION_ABSOLUTE_REPO:-dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-action-decoder}"
  export VIDEO_LORA_FILE="${VIDEO_LORA_FILE:-checkpoints/model/iter_000001000.pt}"
  export ACTION_RELATIVE_FILE="${ACTION_RELATIVE_FILE:-checkpoints/model/iter_000002500.pt}"
  export ACTION_ABSOLUTE_FILE="${ACTION_ABSOLUTE_FILE:-checkpoints/model/iter_000002000.pt}"
  export ACTION_RELATIVE_STATS_FILE="${ACTION_RELATIVE_STATS_FILE:-stats/so101_relative_stats.json}"
  export ACTION_ABSOLUTE_STATS_FILE="${ACTION_ABSOLUTE_STATS_FILE:-stats/so101_stats.json}"
  export BASE_VIDEO_REPO="${BASE_VIDEO_REPO:-}"
  export BASE_VIDEO_FILE="${BASE_VIDEO_FILE:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"
  export ACTION_ZARR_REPO="${ACTION_ZARR_REPO:-dreamdifferent/mimic-video-so101-bottle-action-zarr}"
  export ACTION_ZARR_DIR_NAME="${ACTION_ZARR_DIR_NAME:-so101_bottle_action_only_full}"
  export DOWNLOAD_ABSOLUTE="${DOWNLOAD_ABSOLUTE:-0}"
  export MVS_NUM_SAMPLING_STEPS="${MVS_NUM_SAMPLING_STEPS:-35}"
  export MVS_STOP_VIDEO_DENOISING_STEP="${MVS_STOP_VIDEO_DENOISING_STEP:-10}"
  export MVS_NUM_VAL_EPISODES="${MVS_NUM_VAL_EPISODES:-10}"

  export MVS_SHARED_ACTION_DATA_DIR="${MVS_SHARED_DATA_ROOT}/${ACTION_ZARR_DIR_NAME}"
  export MVS_VIDEO_BACKBONE_PATH="${MVS_SHARED_VIDEO_BACKBONE_DIR}/${BASE_VIDEO_FILE}"
  export MVS_VIDEO_LORA_PATH="${MVS_EXP_VIDEO_LORA_DIR}/${VIDEO_LORA_FILE}"
  export MVS_ACTION_RELATIVE_PATH="${MVS_EXP_ACTION_RELATIVE_DIR}/${ACTION_RELATIVE_FILE}"
  export MVS_ACTION_ABSOLUTE_PATH="${MVS_EXP_ACTION_ABSOLUTE_DIR}/${ACTION_ABSOLUTE_FILE}"
  export MVS_ACTION_RELATIVE_STATS_PATH="${MVS_EXP_ACTION_RELATIVE_DIR}/${ACTION_RELATIVE_STATS_FILE}"
  export MVS_ACTION_ABSOLUTE_STATS_PATH="${MVS_EXP_ACTION_ABSOLUTE_DIR}/${ACTION_ABSOLUTE_STATS_FILE}"
  export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${MVS_SHARED_CHECKPOINT_ROOT}}"
  if [[ "${ACTION_DECODER_KIND}" == "absolute" ]]; then
    export MVS_ACTION_MODEL_PATH="${MVS_ACTION_ABSOLUTE_PATH}"
    export MVS_ACTION_STATS_PATH="${MVS_ACTION_ABSOLUTE_STATS_PATH}"
  else
    export MVS_ACTION_MODEL_PATH="${MVS_ACTION_RELATIVE_PATH}"
    export MVS_ACTION_STATS_PATH="${MVS_ACTION_RELATIVE_STATS_PATH}"
  fi
}

mvs_create_layout() {
  mkdir -p \
    "${MVS_SHARED_VIDEO_BACKBONE_DIR}" \
    "${MVS_SHARED_DATA_ROOT}" \
    "${MVS_SHARED_VIDEO_DATA_ROOT}" \
    "${MVS_EXP_VIDEO_LORA_DIR}" \
    "${MVS_EXP_ACTION_RELATIVE_DIR}" \
    "${MVS_EXP_ACTION_ABSOLUTE_DIR}" \
    "${MVS_EVAL_OUTPUT_ROOT}" \
    "${MVS_LOG_DIR}" \
    "${MVS_RUNS_DIR}" \
    "${UV_CACHE_DIR}" \
    "${UV_TOOL_DIR}" \
    "${UV_TOOL_BIN_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}"
}

mvs_prepare_runtime_env() {
  export PATH="/usr/sbin:/sbin:${UV_TOOL_BIN_DIR}:${PATH}"
  export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
  export MIMIC_DATASET_STATS_NUM_WORKERS="${MIMIC_DATASET_STATS_NUM_WORKERS:-0}"
  export MIMIC_DATASET_STATS_BATCH_SIZE="${MIMIC_DATASET_STATS_BATCH_SIZE:-4}"
  export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MVS_SHARED_VIDEO_DATA_DIR}}"
  export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-5}"

  local nvidia_dir="${REPO_ROOT}/model/.venv/lib/python3.10/site-packages/nvidia"
  if [[ -d "${nvidia_dir}" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${nvidia_dir}}"
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    export LD_LIBRARY_PATH="${nvidia_dir}/cuda_runtime/lib:${nvidia_dir}/cuda_nvrtc/lib:${nvidia_dir}/cudnn/lib:${nvidia_dir}/nccl/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mvs_activate_model_venv_if_present() {
  if [[ -f "${REPO_ROOT}/model/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/model/.venv/bin/activate"
  fi
}

mvs_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
}

mvs_require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
}
