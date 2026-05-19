# SO-101 mimic-video Evaluation Plan

## Current status

- Branch: `so101-finetuning`
- Dataset: `dreamdifferent/so101_bottle`
- Converted full front-camera data:
  - `/cluster/scratch/eugseo/mimic_video_data/so101_bottle_front_full`
- Current trained random-init checkpoint:
  - `/cluster/scratch/eugseo/mimic_video_runs_so101_full_bsz8_20260515_022343/vam/so101/w2a_so101_random_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz8/checkpoints/model/iter_000000250.pt`
- Frozen video backbone:
  - `/cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt`
- SO-101 dataset statistics cache:
  - `/cluster/scratch/eugseo/mimic_video_data/so101_bottle_front_full/.statistics_cache/c408a5f5319d71d04529bb825f448a210cac8585a62a00123e3b3459ae28b350`

## Important interpretation

- The current random-init SO-101 run trained:
  - frozen Bridge-finetuned video backbone,
  - randomly initialized SO-101 `world2action` action decoder,
  - full action-decoder finetuning, not LoRA.
- The `lora_rank256` string in experiment/checkpoint names refers to the frozen Bridge video backbone checkpoint, not the SO-101 action decoder training method.
- The next more meaningful baseline is `partial_bridge_init`:
  - load compatible Bridge action-decoder body weights,
  - randomly initialize shape-incompatible SO-101 6D input/output/action-specific layers,
  - finetune the whole action decoder.

## Recommended next command: partial Bridge-init training

This uses existing Bridge action-decoder information while keeping the same SO-101 6D action setup.

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

DATA_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_front_full \
RUNS_DIR=/cluster/scratch/eugseo/mimic_video_runs_so101_partial_bridge_bsz8 \
INIT_NAME=partial_bridge_init \
VIDEO_CKPT=v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused \
GLOBAL_BATCH_SIZE=8 \
GRAD_ACCUM_ITER=8 \
MAX_ITER=3000 \
SAVE_ITER=100 \
LOGGING_ITER=10 \
WANDB_LOG_EVERY_N=1 \
RUN_VALIDATION=false \
SO101_NUM_VAL_EPISODES=10 \
WANDB_ENABLED=1 \
WANDB_ENTITY=dreamdifferent \
WANDB_PROJECT=mimic-video-so101 \
  sbatch --time=24:00:00 model/scripts/train_so101_smoke.sbatch
```

Notes:

- `RUNS_DIR` is intentionally fixed so future jobs can resume from the latest checkpoint.
- `GLOBAL_BATCH_SIZE=8`, `GRAD_ACCUM_ITER=8` gives effective batch 64.
- `MAX_ITER=3000` is a target; the 24h job may timeout before that, but checkpoints should save and the same command can resume.
- `WANDB_LOG_EVERY_N=1` logs scalar train loss every optimizer step, while `LOGGING_ITER=10` keeps terminal/device-monitor logs less noisy.

## Existing eval code inventory

Current checkout does not contain a ready SO-101 mimic-video eval wrapper.

Relevant mimic-video references:

- Bridge/SimplerEnv VAM policy:
  - `eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py`
- LIBERO eval:
  - `eval/libero/run.py`

Relevant DreamZero infrastructure:

- Generic websocket server/client:
  - `/cluster/project/cvg/students/eugseo/workspace/dreamzero/eval_utils/policy_server.py`
  - `/cluster/project/cvg/students/eugseo/workspace/dreamzero/eval_utils/policy_client.py`
- Current DreamZero sim eval client is DROID-style and assumes 8D actions, so it is not directly compatible with SO-101 6D joint actions.

If a repo update adds SO-101 eval support, prefer that implementation. Otherwise, implement a thin SO-101 policy adapter using the existing mimic-video pipeline loading logic.

## Evaluation strategy

### Phase 1: offline heldout sanity eval

Before robot/sim closed-loop evaluation, evaluate on heldout SO-101 zarr episodes.

Goals:

- Load SO-101 checkpoint and frozen video backbone.
- Load SO-101 normalizer statistics.
- Feed heldout front-camera images, 6D joint state, and task prompt.
- Confirm model output shape is `(H, 6)`.
- Compute action MSE/MAE against heldout future 6D joint actions.
- Save a small qualitative report with:
  - predicted vs. target joint trajectories,
  - action magnitude sanity checks,
  - optional predicted future video preview if useful.

This phase validates I/O and normalization before any online rollout.

### Phase 2: SO-101 policy adapter

Implement a `MimicVideoSO101Policy` wrapper with:

- `make_config()` + `experiment=...` override,
- `Video2WorldPipeline.from_config(...)`,
- `World2ActionPipeline.from_config(...)`,
- SO-101 stats loaded into `world2action_pipe.normalizer`,
- image history of 5 front frames,
- current 6D joint-state input,
- action chunk output `(15, 6)`,
- configurable number of actions to execute per policy query.

### Phase 3: camera mapping

SO-101 provides two images in the LeRobot dataset:

- `observation.images.front`
- `observation.images.wrist`

But the current mimic-video SO-101 checkpoint was trained on only:

- `observation.images.front -> workspace_rgb`

Therefore first eval should use:

- external/front image only,
- wrist image ignored.

Do not use 2-view stitching/grid for the first eval because it is out of training distribution.

### Phase 4: online/sim eval interface

If using DreamZero websocket infra, configure the server/client around:

- one external/front camera,
- no wrist camera required for mimic-video SO-101 v1,
- current 6D joint state,
- output action chunk shape `(N, 6)`,
- action space: joint-position target.

The existing DROID-style client expects 8D actions, so it needs a SO-101-specific client or adapter.

## Acceptance checks

- Offline eval loads random-init `iter_250` checkpoint without shape/stat errors.
- Offline eval loads partial-Bridge-init checkpoint when available.
- Predicted action shape is `(H, 6)`.
- Normalizer uses SO-101 statistics, not Bridge statistics.
- Front-camera image preprocessing matches training: RGB, uint8 input, resized/normalized to mimic-video convention.
- Eval wrapper ignores wrist image for the first baseline.
- Closed-loop interface sends only valid 6D SO-101 joint-position commands.
