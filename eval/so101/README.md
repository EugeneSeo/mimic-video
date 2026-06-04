# SO-101 Mimic Video Eval

This directory contains the SO-101 evaluation utilities:

```text
offline_eval.py              dataset sample eval, generated/oracle modes
policy_server.py             TCP/websocket prediction server
synthetic_policy_client.py   random-image server smoke test
run_server.sh                thin wrapper around policy_server.py
```

The preferred entrypoints are now the experiment-aware wrappers under
`scripts/so101/`. They keep checkpoints, logs, and eval outputs separated by
experiment name.

## Experiment Layout

Default setup:

```bash
export SCRATCH="${SCRATCH:-/cluster/scratch/$USER}"
export MVS_EXPERIMENT="${MVS_EXPERIMENT:-so101-homogeneous-rel}"
export MVS_ROOT="${MVS_ROOT:-$SCRATCH/mimic_video}"
```

The default `so101-homogeneous-rel` experiment uses:

```text
shared base video backbone:
$MVS_ROOT/shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt

experiment video LoRA:
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt

experiment relative action decoder:
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/checkpoints/model/iter_000002500.pt

experiment relative normalizer stats:
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/stats/so101_relative_stats.json

shared action zarr:
$MVS_ROOT/shared/data/so101_bottle_action_only_full

recommended shared 5fps hstack video dataset:
$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full
```

The SO-101 wrappers set `COSMOS_PREDICT2_ARGS="--checkpoints
$MVS_ROOT/shared/checkpoints"` so Cosmos also loads tokenizer/text-encoder
assets from the shared checkpoint directory.

Online requests still use the compatibility key
`observation/images/front`, but the image value should be the 480x640 two-camera
hstack frame:

```text
left half:  front camera
right half: wrist camera
```

## Setup And Asset Checks

From the repository root:

```bash
bash scripts/so101/setup_hf_cli.sh so101-homogeneous-rel
export PATH="$MVS_ROOT/shared/uv-bin:$PATH"
export HF_HOME="$MVS_ROOT/shared/hf-home"

hf auth login
bash scripts/so101/download_base_backbone.sh so101-homogeneous-rel
bash scripts/so101/download_assets.sh so101-homogeneous-rel
bash scripts/so101/download_action_zarr.sh so101-homogeneous-rel
bash scripts/so101/check_assets.sh so101-homogeneous-rel
```

`download_base_backbone.sh` downloads the released Bridge-finetuned Mimic Video
backbone from `jonpai/mimic-video` into the shared checkpoint directory.
`download_assets.sh` downloads the private SO-101 HF video LoRA and relative
action decoder, including the dataset-free serving stats JSON, into the
experiment directory. `download_action_zarr.sh` downloads the temporary
action-only zarr dataset from
`dreamdifferent/mimic-video-so101-bottle-action-zarr` into shared data and
rewrites `paths.pkl` so the dataloader uses the local episode paths. If the zarr
was downloaded before this wrapper existed, repair it with:

```bash
bash scripts/so101/repair_action_zarr_paths.sh so101-homogeneous-rel
```

## Eval Settings

Use Mimic Video's action-eval convention:

```text
num_sampling_steps = 35
stop_video_denoising_step = 10
```

Training-time W&B video previews may use 20 video sampling steps because they
are only lightweight visual progress checks.

## Policy Server

On a GPU node, terminal 1:

```bash
bash scripts/so101/run_policy_server.sh so101-homogeneous-rel
```

By default the server uses:

```text
$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/stats/so101_relative_stats.json
```

so serving does not require the action zarr. To force dataset-derived stats
instead, run with `MVS_SERVER_USE_STATS=0` and make sure the action zarr exists.

Terminal 2 synthetic smoke:

```bash
bash scripts/so101/run_synthetic_client.sh so101-homogeneous-rel
```

Expected smoke result:

```text
actions.shape=(15, 6)
finite values
no server error
```

Synthetic images are random, so action values are not meaningful.

## Offline Eval

Generated mode, real SO-101 zarr samples:

```bash
MVS_EVAL_MODE=generated \
MVS_EVAL_SPLIT=val \
MVS_EVAL_NUM_SAMPLES=4 \
bash scripts/so101/run_offline_eval.sh so101-homogeneous-rel
```

Outputs go under:

```text
$MVS_ROOT/experiments/so101-homogeneous-rel/eval_outputs/generated_val_4_35_10
```

Expected files include:

```text
summary.json
policy_metadata.json
generated_per_sample.csv
generated_trajectories.npz
generated_sample0_trajectory.png
```

For video dumps:

```bash
MVS_EVAL_MODE=generated \
MVS_EVAL_SPLIT=val \
MVS_EVAL_NUM_SAMPLES=2 \
MVS_DUMP_VIDEOS=1 \
bash scripts/so101/run_offline_eval.sh so101-homogeneous-rel
```

## Request Schema

```python
{
    "observation/images/front": hstack_image,  # uint8 RGB, HWC or CHW
    "observation/state": joint_state,          # shape (6,) or (1, 6)
    "prompt": "pick up the bottle",
    "observation/prompt_embedding": embedding, # optional, shape (512, 1024)
    "reset_history": False,
    "seed": 0,
}
```

If the server is not started with `--load-text-encoder`, the client must send
`observation/prompt_embedding`.

## Response Schema

```python
{
    "actions": np.ndarray,  # shape (15, 6)
    "server_timing": {...},
    "metadata": {"action_horizon": 15, "action_dim": 6},
}
```

For `DATA_CONFIG=so101`, actions are absolute joint targets. For
`DATA_CONFIG=so101_relative`, actions are joint deltas and deployment should use
`absolute_target = current_joint + predicted_delta`.
