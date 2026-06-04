#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-homogeneous-rel}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mvs_load_paths "${experiment}"
mvs_create_layout

mvs_require_file "${MVS_VIDEO_LORA_PATH}" "video LoRA"
mvs_require_dir "${MVS_SHARED_ACTION_DATA_DIR}" "action zarr"
mvs_require_dir "${MVS_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATA_DIR="${DATA_DIR:-${MVS_SHARED_ACTION_DATA_DIR}}"
export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MVS_SHARED_VIDEO_DATA_DIR}}"
export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-5}"
export VIDEO_LORA_CKPT="${VIDEO_LORA_CKPT:-${MVS_VIDEO_LORA_PATH}}"
if [[ "${ACTION_DECODER_KIND}" == "absolute" ]]; then
  default_data_config="so101"
  default_wandb_project="mimic-video-so101"
else
  default_data_config="so101_relative"
  default_wandb_project="mimic-video-so101-relative"
fi
export DATA_CONFIG="${DATA_CONFIG:-${default_data_config}}"
export INIT_NAME="${INIT_NAME:-partial_bridge_init}"
export VIDEO_CKPT="${VIDEO_CKPT:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused}"
export RUNS_DIR="${RUNS_DIR:-${MVS_RUNS_DIR}/action_decoder_${ACTION_DECODER_KIND}}"
export LOG_DIR="${LOG_DIR:-${MVS_LOG_DIR}}"
export WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MVS_EXP_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-${default_wandb_project}}"

mkdir -p "${MVS_LOG_DIR}" "${RUNS_DIR}" "${WANDB_ROOT_DIR}"

sbatch \
  --output="${MVS_LOG_DIR}/%x-%j.out" \
  --error="${MVS_LOG_DIR}/%x-%j.err" \
  "$@" \
  "${REPO_ROOT}/model/scripts/train_so101_smoke.sbatch"
