#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

config_env="${MIMIC_VIDEO_SO101_CONFIG_FILE:-${SCRIPT_DIR}/experiments/so101-multi-object-delta30.env}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  config_env="$1"
  shift
fi

# Non-cluster GPU rentals usually do not have /cluster/scratch.
export SCRATCH="${SCRATCH:-${HOME}}"

mimic_video_load_paths "${config_env}"
mimic_video_create_layout
mimic_video_prepare_runtime_env

MODEL_DIR="${REPO_ROOT}/model"
CHECKPOINT_ROOT="${MIMIC_VIDEO_SHARED_CHECKPOINT_ROOT}"
RUNS_DIR="${RUNS_DIR:-${MIMIC_VIDEO_ACTION_DECODER_RUN_DIR}}"
WANDB_ROOT_DIR="${WANDB_ROOT_DIR:-${MIMIC_VIDEO_EXPERIMENT_ROOT}/wandb}"
RUN_TMP_DIR="${RUN_TMP_DIR:-/tmp/mimic_video_w2a_local}"

export DATA_DIR="${DATA_DIR:-${MIMIC_VIDEO_SHARED_ACTION_DATA_DIR}}"
export SO101_VIDEO_DIR="${SO101_VIDEO_DIR:-${MIMIC_VIDEO_SHARED_VIDEO_DATA_DIR}}"
export SO101_VIDEO_FPS="${SO101_VIDEO_FPS:-${SO101_VIDEO_TARGET_FPS}}"
export VIDEO_LORA_CKPT="${VIDEO_LORA_CKPT:-${MIMIC_VIDEO_VIDEO_LORA_CKPT_PATH}}"
export INIT_NAME="${INIT_NAME:-partial_bridge_init}"
export VIDEO_CKPT="${VIDEO_CKPT:-v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused}"
export WANDB_PROJECT="${WANDB_PROJECT:-mimic-video-so101-w2a}"
export WANDB_ROOT_DIR

NUM_GPUS="${NUM_GPUS:-2}"
LR_TAG="${LR_TAG:-1.000e-04}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-8}"
MAX_ITER="${MAX_ITER:-30000}"
LOGGING_ITER="${LOGGING_ITER:-100}"
SAVE_ITER="${SAVE_ITER:-1000}"
KEEP_LATEST_ONLY="${KEEP_LATEST_ONLY:-false}"
VALIDATION_ITER="${VALIDATION_ITER:-1000}"
RUN_VALIDATION="${RUN_VALIDATION:-false}"
NUM_WORKERS="${NUM_WORKERS:-1}"
VIDEO_LORA_RANK="${VIDEO_LORA_RANK:-256}"
VIDEO_LORA_ALPHA="${VIDEO_LORA_ALPHA:-32}"
VIDEO_LORA_TARGET_MODULES="${VIDEO_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2}"

EXPERIMENT="${EXPERIMENT:-w2a_${DATA_CONFIG}_${INIT_NAME}_${VIDEO_CKPT}_lr${LR_TAG}_layer20_bsz${GLOBAL_BATCH_SIZE}}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT}_local}"

mimic_video_require_dir "${DATA_DIR}" "action zarr"
mimic_video_require_dir "${SO101_VIDEO_DIR}" "5fps hstack video dataset"
mimic_video_require_file "${VIDEO_LORA_CKPT}" "video LoRA"

if ! find "${DATA_DIR}" -maxdepth 2 -name '*.zarr' -type d -print -quit | grep -q .; then
  echo "ERROR: DATA_DIR contains no .zarr episodes: ${DATA_DIR}" >&2
  exit 1
fi

SO101_VIDEO_CHECK_DIR="${SO101_VIDEO_DIR}"
if [[ -d "${SO101_VIDEO_DIR}/video" ]]; then
  SO101_VIDEO_CHECK_DIR="${SO101_VIDEO_DIR}/video"
fi
if ! find "${SO101_VIDEO_CHECK_DIR}" -maxdepth 1 -name '*.mp4' -type f -print -quit | grep -q .; then
  echo "ERROR: SO101_VIDEO_DIR must contain mp4 files, or a video/ subdir with mp4 files:" >&2
  echo "  ${SO101_VIDEO_DIR}" >&2
  exit 1
fi

if [[ "${INIT_NAME}" == "partial_bridge_init" ]]; then
  case "${VIDEO_CKPT}" in
    v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused)
      PARTIAL_CKPT="${CHECKPOINT_ROOT}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
      ;;
    v2w_pretrained_cosmos)
      PARTIAL_CKPT="${CHECKPOINT_ROOT}/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
      ;;
    *)
      echo "ERROR: no partial Bridge-init checkpoint is configured for VIDEO_CKPT=${VIDEO_CKPT}" >&2
      exit 1
      ;;
  esac
  mimic_video_require_file "${PARTIAL_CKPT}" "partial Bridge-init action decoder"
fi

VIDEO_CKPT_PATH="${CHECKPOINT_ROOT}/video_backbone/${VIDEO_CKPT}.pt"
mimic_video_require_file "${VIDEO_CKPT_PATH}" "frozen video backbone"

if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
  exit 1
fi
PER_GPU_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

mkdir -p \
  "${RUNS_DIR}" \
  "${WANDB_ROOT_DIR}/cache" \
  "${RUN_TMP_DIR}/uv-cache" \
  "${RUN_TMP_DIR}/torch-cache" \
  "${RUN_TMP_DIR}/triton-cache"

export TMPDIR="${RUN_TMP_DIR}"
export UV_CACHE_DIR="${RUN_TMP_DIR}/uv-cache"
export TORCH_HOME="${RUN_TMP_DIR}/torch-cache"
export TRITON_CACHE_DIR="${RUN_TMP_DIR}/triton-cache"
export IMAGINAIRE_OUTPUT_ROOT="${RUNS_DIR}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-dreamdifferent}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_ROOT_DIR}"
export WANDB_CACHE_DIR="${WANDB_ROOT_DIR}/cache"
export WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export PYTHONUNBUFFERED=1
export SO101_DATA_DIR="${DATA_DIR}"
export SO101_NUM_VAL_EPISODES="${SO101_NUM_VAL_EPISODES:-0}"
export MIMIC_DATASET_STATS_NUM_WORKERS="${MIMIC_DATASET_STATS_NUM_WORKERS:-0}"
export MIMIC_DATASET_STATS_BATCH_SIZE="${MIMIC_DATASET_STATS_BATCH_SIZE:-4}"

MASTER_PORT="${MASTER_PORT:-12345}"

echo "===== SO-101 local action-decoder training ====="
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Model dir: ${MODEL_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Data dir: ${DATA_DIR}"
echo "Video dir: ${SO101_VIDEO_DIR}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Video checkpoint: ${VIDEO_CKPT_PATH}"
echo "Video LoRA checkpoint: ${VIDEO_LORA_CKPT}"
echo "Partial action init: ${PARTIAL_CKPT:-<none>}"
echo "Output root: ${RUNS_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Per-GPU batch: ${PER_GPU_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM_ITER} effective_batch=$((GLOBAL_BATCH_SIZE * GRAD_ACCUM_ITER))"
echo "max_iter=${MAX_ITER} save_iter=${SAVE_ITER} keep_latest_only=${KEEP_LATEST_ONLY}"
echo "W&B: enabled=${WANDB_ENABLED} entity=${WANDB_ENTITY} project=${WANDB_PROJECT} mode=${WANDB_MODE}"
echo "Extra train overrides: ${*:-<none>}"
echo "==============================================="
nvidia-smi || true

TRAIN_CMD=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  -m scripts.train
  --config=cosmos_predict2/configs/config.py
  --
  "experiment=${EXPERIMENT}"
  "trainer.max_iter=${MAX_ITER}"
  "trainer.logging_iter=${LOGGING_ITER}"
  "trainer.run_validation=${RUN_VALIDATION}"
  "trainer.validation_iter=${VALIDATION_ITER}"
  "checkpoint.save_iter=${SAVE_ITER}"
  "checkpoint.keep_latest_only=${KEEP_LATEST_ONLY}"
  "dataloader_train.num_workers=${NUM_WORKERS}"
  "trainer.grad_accum_iter=${GRAD_ACCUM_ITER}"
  "model.config.video_dit_path=${VIDEO_CKPT_PATH}"
  "model.config.video_lora_dit_path=${VIDEO_LORA_CKPT}"
  "model.config.video_lora_rank=${VIDEO_LORA_RANK}"
  "model.config.video_lora_alpha=${VIDEO_LORA_ALPHA}"
  "model.config.video_lora_target_modules='${VIDEO_LORA_TARGET_MODULES}'"
)

if [[ "${INIT_NAME}" == "partial_bridge_init" ]]; then
  TRAIN_CMD+=(
    "model.config.action_dit_path=${PARTIAL_CKPT}"
    "model.config.allow_partial_action_dit_load=true"
  )
fi

TRAIN_CMD+=("$@")

cd "${MODEL_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${TRAIN_CMD[@]}"
  printf '\n'
  echo "DRY_RUN=1; not running."
  exit 0
fi

if [[ -x .venv/bin/python && -x .venv/bin/torchrun ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  "${TRAIN_CMD[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run --extra cu126 "${TRAIN_CMD[@]}"
else
  echo "ERROR: neither model/.venv nor uv is available." >&2
  echo "Set up the environment first:" >&2
  echo "  cd ${MODEL_DIR} && uv sync --extra cu126" >&2
  exit 1
fi

echo "Finished SO-101 local action run: ${RUN_NAME}"
