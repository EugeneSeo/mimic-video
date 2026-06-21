# SO-101 Workflow

This repo uses script-relative paths and defaults scratch storage to:

```bash
SCRATCH="${SCRATCH:-/cluster/scratch/$USER}"
```

Config env files live in `scripts/so101/experiments/`. Each wrapper takes one
config env file as its first argument and forwards the resolved values to sbatch.
HF artifact env files live in `scripts/so101/artifacts/`.

## Configs

```text
scripts/so101/experiments/so101-bottle-absolute5.env       bottle dataset, absolute 5Hz action decoder
scripts/so101/experiments/so101-bottle-delta30.env         bottle dataset, 30Hz incremental-delta action decoder
scripts/so101/experiments/so101-multi-object-delta30.env   multi-object dataset, 30Hz incremental-delta action decoder
scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env
                                                               multi-object dataset, 30Hz delta joints 0..4 + absolute gripper
```

## Setup

```bash
cd /cluster/project/cvg/students/$USER/workspace/mimic-video

bash scripts/so101/setup_env.sh
export PATH="$SCRATCH/mimic_video/shared/uv-bin:$PATH"
hf auth login
```

Setup is shared and does not take a config env file. Choose the config only when
running a workflow command:

```bash
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-bottle-delta30.env --target all
```

For a new dataset or embodiment, copy one env file, change the dataset/HF/checkpoint
values, and pass that env file to the wrappers.

```bash
cp scripts/so101/experiments/so101-bottle-delta30.env /path/to/my_dataset.env
# Edit /path/to/my_dataset.env.
bash scripts/so101/preprocess_data.sh /path/to/my_dataset.env --target all
```

Use `DRY_RUN=1` on wrappers to print the `sbatch` command without submitting.

## Change Dataset

To run the same workflow on another SO-101 dataset, copy an experiment env and
change only the dataset/output names first.

For example, to switch the multi-object delta30 gripper-absolute workflow to the
bottle dataset:

```bash
cp scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env \
  /tmp/so101-bottle-delta30-gripper-absolute.env
```

Edit the copied env:

```bash
SO101_DATASET_REPO="dreamdifferent/so101_bottle"
SO101_ACTION_DATASET_REPO="dreamdifferent/so101_bottle"
VIDEO_DATA_DIR_NAME="so101_bottle_front_wrist_hstack_5fps_full"
ACTION_ZARR_DIR_NAME="so101_bottle_action_only_full"

# Use a bottle video LoRA checkpoint or set VIDEO_LORA_FILE="none" to train the
# action decoder from the base video backbone only.
VIDEO_LORA_FILE="/path/to/bottle/video_lora/checkpoints/model/iter_000001000.pt"

# Keep a separate output root so different datasets do not resume each other.
RUNS_DIR="/cluster/scratch/${USER}/mimic_video_runs_so101_bottle_w2a_delta_30hz_gripper_absolute"
```

Then run the normal commands with the copied env:

```bash
bash scripts/so101/preprocess_data.sh /tmp/so101-bottle-delta30-gripper-absolute.env --target all
bash scripts/so101/train_action_decoder.sh /tmp/so101-bottle-delta30-gripper-absolute.env --partition=gpupr.24h --gres=gpumem:40G --time=24:00:00
```

If you want the copied env to be shared with others, place it under
`scripts/so101/experiments/` and add a matching artifact env under
`scripts/so101/artifacts/` once checkpoints are published.

## Shared Backbone

Run once per scratch workspace. The base video backbone is shared by all SO-101 experiments.

```bash
bash scripts/so101/download_base_backbone.sh
```

## Assets

Download published video LoRA/action decoder artifacts when evaluating or resuming
from released checkpoints:

```bash
bash scripts/so101/download_assets.sh scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env
bash scripts/so101/check_assets.sh scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env
```

Artifact locations are kept in `scripts/so101/artifacts/`. The gripper-absolute
action decoder currently points to:

```text
dreamdifferent/mimic-video-so101-multi-object-delta30-gripper-absolute-action-decoder
```

## Preprocess

```bash
# Video + action zarr + T5 embeddings.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-bottle-delta30.env --target all

# Only video side.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-multi-object-delta30.env --target video

# Only action side.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-multi-object-delta30.env --target action

# Multi-object delta30 gripper-absolute action decoder data.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env --target all
```

## Video LoRA

```bash
bash scripts/so101/train_video_backbone.sh scripts/so101/experiments/so101-bottle-delta30.env
bash scripts/so101/render_video_preview.sh scripts/so101/experiments/so101-bottle-delta30.env
```

For multi-object video LoRA:

```bash
bash scripts/so101/train_video_backbone.sh scripts/so101/experiments/so101-multi-object-delta30.env
bash scripts/so101/render_video_preview.sh scripts/so101/experiments/so101-multi-object-delta30.env
```

## Action Decoder

```bash
bash scripts/so101/train_action_decoder.sh scripts/so101/experiments/so101-bottle-delta30.env
bash scripts/so101/train_action_decoder.sh scripts/so101/experiments/so101-multi-object-delta30.env
```

For the 30Hz delta action decoder with absolute gripper:

```bash
bash scripts/so101/train_action_decoder.sh \
  scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env \
  --partition=gpupr.24h \
  --gres=gpumem:40G \
  --time=24:00:00
```

For a larger 80GB run:

```bash
bash scripts/so101/train_action_decoder.sh \
  scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env \
  --partition=gpupr.120h \
  --gres=gpumem:80G \
  --time=48:00:00
```

On Euler, do not pass `--gres=gpu:2` together with `--gres=gpumem:*` for this
wrapper. The sbatch already requests GPUs; override only GPU memory unless you
are editing the sbatch itself. Continuing the same experiment with the same
`RUNS_DIR` resumes from the latest checkpoint.

Absolute 5Hz assets are mainly for old bottle eval/download:

```bash
bash scripts/so101/download_assets.sh scripts/so101/experiments/so101-bottle-absolute5.env
```

## Eval

Keep eval checkpoint and stats paths flowing from the SO-101 env loader:

```bash
bash scripts/so101/check_assets.sh scripts/so101/experiments/so101-bottle-absolute5.env
bash scripts/so101/run_policy_server.sh scripts/so101/experiments/so101-bottle-absolute5.env
```

For delta checkpoints after training:

```bash
bash scripts/so101/check_assets.sh scripts/so101/experiments/so101-bottle-delta30.env
bash scripts/so101/run_policy_server.sh scripts/so101/experiments/so101-bottle-delta30.env
```

For multi-object delta30 gripper-absolute checkpoints:

```bash
bash scripts/so101/check_assets.sh scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env
bash scripts/so101/run_policy_server.sh scripts/so101/experiments/so101-multi-object-delta30-gripper-absolute.env
```
