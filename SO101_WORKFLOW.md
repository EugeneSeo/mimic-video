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

## Shared Backbone

Run once per scratch workspace. The base video backbone is shared by all SO-101 experiments.

```bash
bash scripts/so101/download_base_backbone.sh
```

## Preprocess

```bash
# Video + action zarr + T5 embeddings.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-bottle-delta30.env --target all

# Only video side.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-multi-object-delta30.env --target video

# Only action side.
bash scripts/so101/preprocess_data.sh scripts/so101/experiments/so101-multi-object-delta30.env --target action
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

Pass extra sbatch options after the config env. For a longer 24 hour action
decoder run:

```bash
bash scripts/so101/train_action_decoder.sh scripts/so101/experiments/so101-multi-object-delta30.env --time=24:00:00
```

On a non-Slurm GPU server, set the data/checkpoint root explicitly and use the
local torchrun wrapper:

```bash
export MIMIC_VIDEO_ROOT=/workspace/mimic_video
export NUM_GPUS=2
bash scripts/so101/train_action_decoder_local.sh scripts/so101/experiments/so101-multi-object-delta30.env
```

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
