# SO-101 Workflow

This repo uses script-relative paths and defaults scratch storage to:

```bash
SCRATCH="${SCRATCH:-/cluster/scratch/$USER}"
```

Experiment configs live in `scripts/so101/experiments/`.
HF artifact configs live in `scripts/so101/artifacts/`.

## Experiments

```text
so101-bottle-absolute5       bottle dataset, absolute 5Hz action decoder
so101-bottle-delta30         bottle dataset, 30Hz incremental-delta action decoder
so101-multi-object-delta30   multi-object dataset, 30Hz incremental-delta action decoder
```

## Setup

```bash
cd /cluster/project/cvg/students/$USER/workspace/mimic-video

bash scripts/so101/setup_env.sh
export PATH="$SCRATCH/mimic_video/shared/uv-bin:$PATH"
hf auth login
```

Setup is shared across SO-101 experiments and does not take an experiment name.
Choose the experiment only when running a workflow command, either as the first
argument:

```bash
bash scripts/so101/preprocess_data.sh so101-bottle-delta30 --target all
```

or as a shell default:

```bash
export MIMIC_VIDEO_EXPERIMENT=so101-bottle-delta30
bash scripts/so101/preprocess_data.sh --target all
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
bash scripts/so101/preprocess_data.sh so101-bottle-delta30 --target all

# Only video side.
bash scripts/so101/preprocess_data.sh so101-multi-object-delta30 --target video

# Only action side.
bash scripts/so101/preprocess_data.sh so101-multi-object-delta30 --target action
```

## Video LoRA

```bash
bash scripts/so101/train_video_backbone.sh so101-bottle-delta30
bash scripts/so101/render_video_preview.sh so101-bottle-delta30
```

For multi-object video LoRA:

```bash
bash scripts/so101/train_video_backbone.sh so101-multi-object-delta30
bash scripts/so101/render_video_preview.sh so101-multi-object-delta30
```

## Action Decoder

```bash
bash scripts/so101/train_action_decoder.sh so101-bottle-delta30
bash scripts/so101/train_action_decoder.sh so101-multi-object-delta30
```

Absolute 5Hz assets are mainly for old bottle eval/download:

```bash
bash scripts/so101/download_assets.sh so101-bottle-absolute5
```

## Eval

Keep eval checkpoint and stats paths flowing from the SO-101 env loader:

```bash
bash scripts/so101/check_assets.sh so101-bottle-absolute5
bash scripts/so101/run_policy_server.sh so101-bottle-absolute5
```

For delta checkpoints after training:

```bash
bash scripts/so101/check_assets.sh so101-bottle-delta30
bash scripts/so101/run_policy_server.sh so101-bottle-delta30
```
