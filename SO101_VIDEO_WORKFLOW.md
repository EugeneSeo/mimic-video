# SO-101 Video Backbone Workflow

This document covers SO-101 video2world data preparation, T5 precompute, video
LoRA training, and preview rendering. The current baseline visual convention is
two-camera horizontal stacking:

```text
480x640 RGB frame
left half:  front camera, resized/padded into 480x320
right half: wrist camera, resized/padded into 480x320
```

The exported mp4s are real 5 fps subsampled videos. We do not only change mp4
metadata. For SO-101 30 fps sources, conversion keeps every 6th frame.

## Path Convention

Use user-independent scratch paths for new runs:

```bash
export REPO_ROOT="${REPO_ROOT:-/cluster/project/cvg/students/$USER/workspace/mimic-video}"
export SCRATCH="${SCRATCH:-/cluster/scratch/$USER}"
export MVS_ROOT="${MVS_ROOT:-$SCRATCH/mimic_video}"
export MVS_EXPERIMENT="${MVS_EXPERIMENT:-so101-homogeneous-rel}"
```

Shared datasets and base checkpoints live under `$MVS_ROOT/shared`. Per-run
video LoRA checkpoints, logs, and previews live under
`$MVS_ROOT/experiments/$MVS_EXPERIMENT`.

## Environment

Run commands from the repository root on Euler:

```bash
cd $REPO_ROOT
cd model
source .venv/bin/activate
cd ..
```

The sbatch scripts set the Euler-specific CUDA, Hugging Face cache, W&B cache,
and `eth_proxy` environment variables internally.

## Checkpoints And Repos

Required local checkpoints:

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

Hugging Face model repos used so far:

```text
Video LoRA:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-v2w-lora

Absolute action decoder:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-action-decoder

Relative action decoder:
dreamdifferent/mimic-video-so101-2cam-hstack-5fps-w2a-relative-action-decoder
```

Dataset repos:

```text
Bottle-only:   dreamdifferent/so101_bottle
Multi-object:  dreamdifferent/so101_multi_object_new
```

## Convert Video Data

Bottle hstack dataset:

```bash
cd $REPO_ROOT

REPO_ID=dreamdifferent/so101_bottle \
OUTPUT_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
MAX_EPISODES= \
OVERWRITE=1 \
SOURCE_FPS=30 \
TARGET_FPS=5 \
VIEW_LAYOUT=hstack \
VIDEO_BACKEND=pyav \
sbatch model/scripts/convert_so101_two_camera_video.sbatch
```

Multi-object hstack dataset:

```bash
cd $REPO_ROOT

REPO_ID=dreamdifferent/so101_multi_object_new \
OUTPUT_DIR=$MVS_ROOT/shared/video_data/so101_multi_object_front_wrist_hstack_5fps_full \
MAX_EPISODES= \
OVERWRITE=1 \
SOURCE_FPS=30 \
TARGET_FPS=5 \
VIEW_LAYOUT=hstack \
VIDEO_BACKEND=pyav \
sbatch model/scripts/convert_so101_two_camera_video.sbatch
```

Expected counts:

```bash
DATASET_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full

find "$DATASET_DIR/video" -maxdepth 1 -name '*.mp4' | wc -l
find "$DATASET_DIR/metas" -maxdepth 1 -name '*.txt' | wc -l
```

Known counts:

```text
so101_bottle:            110 videos / 110 metas
so101_multi_object_new:  153 videos / 153 metas
```

Duration sanity check from conversion logs:

```bash
awk '
/episode_/ && /kept/ {
  split($0,a,"kept "); split(a[2],b," frames"); split(b[1],c,"/");
  kept+=c[1]; src+=c[2]; n+=1;
  diff=((c[1]-1)/5)-((c[2]-1)/30); abs=diff<0?-diff:diff;
  if (abs>maxdiff) {maxdiff=abs; maxline=$0}
}
END {
  printf("episodes=%d source_duration=%.3f kept_duration=%.3f max_episode_span_diff=%.3f\n", n, src/30, kept/5, maxdiff);
  print maxline;
}' $MVS_ROOT/experiments/so101-homogeneous-rel/logs/<convert-job>.out
```

Episode-level span differences around one 5 fps frame are expected because each
episode keeps frames `0, 6, 12, ...`.

## T5 Precompute

Run after conversion. This creates `t5_xxl/episode_xxxxxx.pickle` from the task
text in `metas/*.txt`; the text encoder is not trained during video/action
training.

```bash
cd $REPO_ROOT

DATASET_PATH=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
sbatch model/scripts/precompute_so101_two_camera_video_t5.sbatch
```

Check:

```bash
find "$DATASET_PATH/t5_xxl" -maxdepth 1 -name '*.pickle' | wc -l
```

The T5 pickle count should match the mp4 count.

## Train Video LoRA

For the current experiment-aware scratch layout, prefer the wrapper:

```bash
cd "$REPO_ROOT"
bash scripts/so101/submit_video_lora_train.sh "$MVS_EXPERIMENT"
```

It fills `REPO_ROOT`, `DATASET_DIR`, `RUNS_DIR`,
`SO101_TWO_CAMERA_INIT_DIT_PATH`, Hugging Face cache paths, and Slurm log paths
from `scripts/so101/lib/paths.sh`.

The video backbone script uses `PER_GPU_BATCH_SIZE`. Older commands using
`GLOBAL_BATCH_SIZE` still work as a fallback, but new commands should use
`PER_GPU_BATCH_SIZE`.

Bottle run:

```bash
cd $REPO_ROOT

DATASET_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
RUNS_DIR=$MVS_ROOT/experiments/so101-homogeneous-rel/runs/video_lora \
MAX_ITER=3000 \
SAVE_ITER=100 \
KEEP_LATEST_ONLY=true \
PER_GPU_BATCH_SIZE=2 \
GRAD_ACCUM_ITER=8 \
WANDB_MODE=online \
WANDB_PROJECT=mimic-video-so101-v2w \
SO101_V2W_WANDB_VIDEO_EVAL=1 \
SO101_V2W_WANDB_VIDEO_EVERY_N=250 \
SO101_V2W_WANDB_VIDEO_START_AFTER=500 \
SO101_V2W_WANDB_VIDEO_NUM_SAMPLES=2 \
SO101_V2W_WANDB_VIDEO_STEPS=20 \
sbatch --time=24:00:00 model/scripts/train_so101_two_camera_video_backbone_2gpu.sbatch
```

Multi-object run:

```bash
cd $REPO_ROOT

DATASET_DIR=$MVS_ROOT/shared/video_data/so101_multi_object_front_wrist_hstack_5fps_full \
RUNS_DIR=$MVS_ROOT/experiments/so101-multi-object/runs/video_lora \
MAX_ITER=3000 \
SAVE_ITER=100 \
KEEP_LATEST_ONLY=true \
PER_GPU_BATCH_SIZE=2 \
GRAD_ACCUM_ITER=8 \
WANDB_MODE=online \
WANDB_PROJECT=mimic-video-so101-multi-object-v2w \
SO101_V2W_WANDB_VIDEO_EVAL=1 \
SO101_V2W_WANDB_VIDEO_EVERY_N=250 \
SO101_V2W_WANDB_VIDEO_START_AFTER=500 \
SO101_V2W_WANDB_VIDEO_NUM_SAMPLES=2 \
SO101_V2W_WANDB_VIDEO_STEPS=20 \
sbatch --time=24:00:00 model/scripts/train_so101_two_camera_video_backbone_2gpu.sbatch
```

With `2` GPUs, `PER_GPU_BATCH_SIZE=2`, and `GRAD_ACCUM_ITER=8`, the effective
batch size is:

```text
2 GPUs * 2 samples/GPU * 8 grad accumulation = 32
```

`KEEP_LATEST_ONLY=true` keeps only the latest checkpoint while saving every
`SAVE_ITER`.

## W&B Video Preview

The callback logs video predictions when:

```text
SO101_V2W_WANDB_VIDEO_EVAL=1
SO101_V2W_WANDB_VIDEO_START_AFTER=500
SO101_V2W_WANDB_VIDEO_EVERY_N=250
SO101_V2W_WANDB_VIDEO_STEPS=20
```

Expected preview iterations with the command above:

```text
500, 750, 1000, 1250, ...
```

These previews are a cheap video-backbone monitor. Action eval uses the Mimic
Video convention `num_sampling_steps=35` and `stop_video_denoising_step=10`.

## Render Preview From Checkpoint

```bash
cd $REPO_ROOT

DATASET_DIR=$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full \
MODEL_CHECKPOINT=/path/to/iter_00000XXXX.pt \
OUTPUT_DIR=$MVS_ROOT/experiments/so101-homogeneous-rel/eval_outputs/v2w_preview \
NUM_SAMPLES=4 \
NUM_SAMPLING_STEPS=20 \
NUM_CONDITIONAL_FRAMES=5 \
SEED=0 \
sbatch model/scripts/render_so101_two_camera_video_preview.sbatch
```

The preview renderer writes side-by-side ground-truth and predicted videos.
