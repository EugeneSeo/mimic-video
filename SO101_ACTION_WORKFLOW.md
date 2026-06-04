# SO-101 Action Decoder Workflow

This document covers SO-101 action zarr conversion, absolute action-decoder
training, relative action-decoder training, and checkpoint upload conventions.
Video backbone data preparation is documented in `SO101_VIDEO_WORKFLOW.md`.

## Path Convention

Use user-independent scratch paths for new runs:

```bash
export REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/$USER/workspace/mimic-video}"
export SCRATCH="${SCRATCH:-/cluster/scratch/$USER}"
export MVS_ROOT="${MVS_ROOT:-$SCRATCH/mimic_video}"
export MVS_EXPERIMENT="${MVS_EXPERIMENT:-so101-homogeneous-rel}"
```

Shared data and the base video backbone live under `$MVS_ROOT/shared`. Per-run
video LoRA/action-decoder checkpoints and logs live under
`$MVS_ROOT/experiments/$MVS_EXPERIMENT`.

## Representations

Two SO-101 action representations are available:

```text
DATA_CONFIG=so101
  action target: future absolute joint target
  deployment: send predicted action as the target joint position

DATA_CONFIG=so101_relative
  action target: future_joint - current_joint
  deployment: absolute_target = current_joint + predicted_delta
```

Both use:

```text
obs video:       5 frames at 5 Hz
future video:   56 frames at 5 Hz
action target:  15 actions at 5 Hz
```

The action chunk covers 3 seconds. For guarded robot rollout, execute only the
first 1 to 5 actions before replanning.

## Checkpoints And Repos

Required video checkpoints:

```text
Base video backbone:
$MVS_ROOT/shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt

SO-101 bottle video LoRA:
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt
```

Download the base Bridge-finetuned Mimic Video backbone with:

```bash
cd "$REPO_ROOT"
bash scripts/so101/download_base_backbone.sh "$MVS_EXPERIMENT"
```

Hugging Face model repos:

```text
Video LoRA:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-v2w-lora

Absolute action decoder:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-action-decoder

Relative action decoder:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-relative-action-decoder
```

## Convert Action Zarr

The action zarr can omit image frames. During training, the loader reads hstack
5 fps frames from `SO101_VIDEO_DIR`.

If the temporary HF dataset is available, download it into the shared data root:

```bash
cd "$REPO_ROOT"
bash scripts/so101/download_action_zarr.sh "$MVS_EXPERIMENT"
```

```bash
cd $REPO_ROOT

OUTPUT_DIR=$MVS_ROOT/shared/data/so101_bottle_action_only_full \
MAX_EPISODES= \
SKIP_VIDEO=1 \
VIDEO_BACKEND=pyav \
OVERWRITE=1 \
sbatch model/scripts/convert_so101_smoke.sbatch
```

Expected count for `so101_bottle`:

```bash
find $MVS_ROOT/shared/data/so101_bottle_action_only_full -maxdepth 2 -name '*.zarr' -type d | wc -l
```

## Precompute Action T5

If the zarr does not already contain `language_embedding`, run:

```bash
cd $REPO_ROOT

DATA_DIR=$MVS_ROOT/shared/data/so101_bottle_action_only_full \
sbatch model/scripts/precompute_so101_t5_smoke.sbatch
```

Quick check:

```bash
python - <<'PY'
import os, pathlib, zarr
root = pathlib.Path(os.environ["MVS_ROOT"]) / "shared/data/so101_bottle_action_only_full"
episodes = sorted(root.glob("*.zarr"))
count = 0
for ep in episodes:
    z = zarr.open(ep, mode="r")
    if "language_embedding" in z:
        count += 1
print(f"language_embedding: {count}/{len(episodes)}")
PY
```

## Train Absolute Action Decoder

For the current experiment-aware scratch layout, prefer the wrapper and override
only the representation-specific knobs:

```bash
cd "$REPO_ROOT"
DATA_CONFIG=so101 \
WANDB_PROJECT=mimic-video-so101 \
bash scripts/so101/submit_action_decoder_train.sh "$MVS_EXPERIMENT"
```

It fills `REPO_ROOT`, `DATA_DIR`, `SO101_VIDEO_DIR`, `VIDEO_LORA_CKPT`,
`RUNS_DIR`, Hugging Face cache paths, and Slurm log paths from
`scripts/so101/lib/paths.sh`.

```bash
cd $REPO_ROOT

DATA_DIR=$MVS_ROOT/shared/data/so101_bottle_action_only_full \
SO101_VIDEO_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
SO101_VIDEO_FPS=5 \
DATA_CONFIG=so101 \
INIT_NAME=partial_bridge_init \
VIDEO_CKPT=v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused \
VIDEO_LORA_CKPT=$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt \
LR_TAG=1.000e-04 \
GLOBAL_BATCH_SIZE=4 \
GRAD_ACCUM_ITER=8 \
MAX_ITER=3000 \
SAVE_ITER=250 \
KEEP_LATEST_ONLY=true \
RUN_VALIDATION=false \
SO101_NUM_VAL_EPISODES=0 \
WANDB_MODE=online \
WANDB_PROJECT=mimic-video-so101 \
RUNS_DIR=$MVS_ROOT/experiments/so101-homogeneous-rel/runs/action_decoder_absolute \
sbatch --time=24:00:00 model/scripts/train_so101_smoke.sbatch
```

For action-decoder training, `GLOBAL_BATCH_SIZE` is global across GPUs. With 2
GPUs, `GLOBAL_BATCH_SIZE=4`, and `GRAD_ACCUM_ITER=8`:

```text
2 GPUs * 2 samples/GPU * 8 grad accumulation = 32
```

## Train Relative Action Decoder

For `so101-homogeneous-rel`, the wrapper defaults to `DATA_CONFIG=so101_relative`
and `INIT_NAME=partial_bridge_init`:

```bash
cd "$REPO_ROOT"
bash scripts/so101/submit_action_decoder_train.sh "$MVS_EXPERIMENT"
```

```bash
cd $REPO_ROOT

DATA_DIR=$MVS_ROOT/shared/data/so101_bottle_action_only_full \
SO101_VIDEO_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
SO101_VIDEO_FPS=5 \
DATA_CONFIG=so101_relative \
INIT_NAME=partial_bridge_init \
VIDEO_CKPT=v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused \
VIDEO_LORA_CKPT=$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt \
LR_TAG=1.000e-04 \
GLOBAL_BATCH_SIZE=4 \
GRAD_ACCUM_ITER=8 \
MAX_ITER=3000 \
SAVE_ITER=500 \
KEEP_LATEST_ONLY=true \
RUN_VALIDATION=false \
SO101_NUM_VAL_EPISODES=0 \
WANDB_MODE=online \
WANDB_PROJECT=mimic-video-so101-relative \
RUNS_DIR=$MVS_ROOT/experiments/so101-homogeneous-rel/runs/action_decoder_relative \
sbatch --time=24:00:00 model/scripts/train_so101_smoke.sbatch
```

The first relative run saved:

```text
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/checkpoints/model/iter_000002500.pt
```

## Upload Action Decoder To Hugging Face

Example for the relative decoder:

```bash
cd $REPO_ROOT/model
source .venv/bin/activate

REPO_ID=dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-relative-action-decoder
RUN_DIR=$MVS_ROOT/experiments/so101-homogeneous-rel/runs/action_decoder_relative/vam/so101_relative/w2a_so101_relative_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4

hf repo create "$REPO_ID" --repo-type model --private || true

hf upload "$REPO_ID" \
  "$RUN_DIR/checkpoints/model/iter_000002500.pt" \
  checkpoints/model/iter_000002500.pt \
  --repo-type model

hf upload "$REPO_ID" \
  "$RUN_DIR/config.yaml" \
  configs/config_iter_000002500.yaml \
  --repo-type model
```

Only upload optimizer/scheduler/trainer checkpoints if another run needs to
resume training exactly from that checkpoint.
