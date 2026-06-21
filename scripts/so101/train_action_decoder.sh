#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-bottle-delta30.env}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  config_env="$1"
  shift
fi
mimic_video_load_paths "${config_env}"
mimic_video_create_layout

if [[ "${VIDEO_LORA_FILE}" != "none" ]]; then
  mimic_video_require_file "${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}" "video LoRA"
fi
mimic_video_require_file "${MIMIC_VIDEO_BACKBONE_PATH}" "base video backbone"
mimic_video_require_file "${MIMIC_VIDEO_SHARED_VIDEO_BACKBONE_DIR}/tokenizer/tokenizer.pth" "base video tokenizer"
mimic_video_require_dir "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" "action zarr"
mimic_video_require_dir "${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}" "5fps hstack video dataset"
if ! find "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "ERROR: action zarr contains no .zarr episodes: ${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" >&2
  exit 1
fi
if ! find "${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" -maxdepth 3 -name language_embedding -type d -print -quit | grep -q .; then
  echo "ERROR: action zarr contains no language_embedding dirs: ${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}" >&2
  exit 1
fi
video_check_dir="${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}"
if [[ -d "${video_check_dir}/video" ]]; then
  video_check_dir="${video_check_dir}/video"
fi
if ! find "${video_check_dir}" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
  echo "ERROR: 5fps hstack video dataset contains no mp4 files: ${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}" >&2
  exit 1
fi

export REPO_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export DATA_DIR="${DATA_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-${SO101_VIDEO_TARGET_FPS}}"
if [[ "${VIDEO_LORA_FILE}" == "none" ]]; then
  export VIDEO_LORA_CKPT=""
else
  export VIDEO_LORA_CKPT="${VIDEO_LORA_CKPT:-${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}}"
fi
case "${ACTION_TRANSFORM}" in
  absolute)
    default_wandb_project="mimic-video-so101"
    ;;
  delta|delta_gripper_absolute)
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
model_checkpoint_root="${REPO_ROOT}/model/checkpoints"
mimic_video_require_file "${model_checkpoint_root}/video_backbone/${VIDEO_CKPT}.pt" "model video backbone"
if [[ "${INIT_NAME}" == "partial_bridge_init" ]]; then
  case "${VIDEO_CKPT}" in
    v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused)
      partial_bridge_ckpt="${model_checkpoint_root}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
      ;;
    v2w_pretrained_cosmos)
      partial_bridge_ckpt="${model_checkpoint_root}/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
      ;;
    *)
      echo "ERROR: no partial Bridge-init checkpoint is configured for VIDEO_CKPT=${VIDEO_CKPT}" >&2
      exit 1
      ;;
  esac
  mimic_video_require_file "${partial_bridge_ckpt}" "partial Bridge-init action decoder"
fi
export RUNS_DIR="${RUNS_DIR:-${MIMIC_VIDEO_ACTION_DECODER_RUN_DIR}}"
export LOG_DIR="${LOG_DIR:-${MIMIC_VIDEO_LOG_DIR}}"
export WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MIMIC_VIDEO_EXPERIMENT_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-${default_wandb_project}}"
export MAX_ITER="${MAX_ITER:-2000}"
export SAVE_ITER="${SAVE_ITER:-250}"
export KEEP_LATEST_ONLY="${KEEP_LATEST_ONLY:-false}"
export RUN_VALIDATION="${RUN_VALIDATION:-false}"
export LOGGING_ITER="${LOGGING_ITER:-10}"
export VALIDATION_ITER="${VALIDATION_ITER:-50}"
export GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"

mkdir -p "${MIMIC_VIDEO_LOG_DIR}" "${RUNS_DIR}" "${WANDB_ROOT_DIR}"

echo "SO-101 action submit: max_iter=${MAX_ITER} save_iter=${SAVE_ITER} keep_latest_only=${KEEP_LATEST_ONLY} run_validation=${RUN_VALIDATION}"
echo "SO-101 action submit: runs_dir=${RUNS_DIR}"
echo "SO-101 action submit: video_lora=${VIDEO_LORA_CKPT:-<none>}"

cmd=(
  sbatch
  --export=ALL
  --output="${MIMIC_VIDEO_LOG_DIR}/%x-%j.out"
  --error="${MIMIC_VIDEO_LOG_DIR}/%x-%j.err"
  "$@"
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
