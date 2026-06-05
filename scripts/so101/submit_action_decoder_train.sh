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

mvs_require_file "${MVS_VIDEO_LORA_PATH}" "video LoRA"
mvs_require_dir "${MVS_SHARED_ACTION_DATA_DIR}" "action zarr"
mvs_require_dir "${MVS_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATA_DIR="${DATA_DIR:-${MVS_SHARED_ACTION_DATA_DIR}}"
export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MVS_SHARED_VIDEO_DATA_DIR}}"
export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-${SO101_VIDEO_TARGET_FPS}}"
export VIDEO_LORA_CKPT="${VIDEO_LORA_CKPT:-${MVS_VIDEO_LORA_PATH}}"
case "${ACTION_TRANSFORM}" in
  absolute)
    default_wandb_project="mimic-video-so101"
    ;;
  delta)
    default_wandb_project="mimic-video-so101-w2a"
    ;;
  *)
    echo "ERROR: unsupported ACTION_TRANSFORM=${ACTION_TRANSFORM}" >&2
    exit 1
    ;;
esac
echo "SO-101 action config: data_config=${DATA_CONFIG} transform=${ACTION_TRANSFORM} predicted_actions=${SO101_ACTION_HORIZON} action_hz=${SO101_ACTION_TARGET_FREQUENCY} w2a_max_horizon=${SO101_W2A_MAX_HORIZON}"
export INIT_NAME="${INIT_NAME:-partial_bridge_init}"
export VIDEO_CKPT="${VIDEO_CKPT:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused}"
export RUNS_DIR="${RUNS_DIR:-${MVS_ACTION_RUN_DIR}}"
export LOG_DIR="${LOG_DIR:-${MVS_LOG_DIR}}"
export WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MVS_EXP_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-${default_wandb_project}}"

mkdir -p "${MVS_LOG_DIR}" "${RUNS_DIR}" "${WANDB_ROOT_DIR}"

cmd=(
  sbatch
  --output="${MVS_LOG_DIR}/%x-%j.out" \
  --error="${MVS_LOG_DIR}/%x-%j.err" \
  "$@" \
  "${REPO_ROOT}/model/scripts/train_so101_smoke.sbatch"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

"${cmd[@]}"
