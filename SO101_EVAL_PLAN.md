# SO-101 mimic-video Evaluation Plan

## Current status

- Branch: `so101-finetuning`
- Active eval experiment: `so101-homogeneous-rel`
- Dataset source: `dreamdifferent/so101_bottle`
- Frozen video backbone:
  - `$MVS_ROOT/shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt`
- SO-101 two-camera video LoRA:
  - `$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt`
- SO-101 relative action decoder:
  - `$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/checkpoints/model/iter_000002500.pt`
- SO-101 relative normalizer stats for dataset-free serving:
  - `$MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/stats/so101_relative_stats.json`
- Optional action zarr for offline dataset eval:
  - `$MVS_ROOT/shared/data/so101_bottle_action_only_full`
- Optional hstack 5 fps video data for offline/generated eval:
  - `$MVS_ROOT/shared/video_data/so101_bottle_front_wrist_hstack_5fps_full`

## Important interpretation

- The active `so101-homogeneous-rel` eval stack uses:
  - frozen Bridge-finetuned video backbone,
  - SO-101 two-camera video LoRA,
  - SO-101 relative `world2action` action decoder,
  - stats loaded from JSON so the policy server does not need zarr at startup.
- The `lora_rank256` string in experiment/checkpoint names refers to the frozen Bridge video backbone checkpoint, not the SO-101 action decoder training method.
- For `DATA_CONFIG=so101_relative`, actions are joint deltas. Deployment should apply `absolute_target = current_joint + predicted_delta`.

## Recommended current command: policy server

The wrapper should follow the experiment layout above and use the stats JSON by
default, avoiding zarr access for online serving:

```bash
cd $REPO_ROOT

bash scripts/so101/run_policy_server.sh so101-homogeneous-rel
```

The server must use `msgpack-numpy` payloads for MacBook/robot clients. Do not
serve robot clients with the pickle fallback because NumPy 2.x clients and NumPy
1.x servers can fail during `pickle.loads()`. If the wrapper reports missing
`msgpack`, install it in the server venv and restart:

```bash
cd $REPO_ROOT/model
source .venv/bin/activate
pip install msgpack
```

Equivalent explicit command:

```bash
python eval/so101/policy_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --experiment-name w2a_so101_relative_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
  --video-model-path $MVS_ROOT/shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  --video-lora-path $MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/video_lora/checkpoints/model/iter_000001000.pt \
  --action-model-path $MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/checkpoints/model/iter_000002500.pt \
  --stats-path $MVS_ROOT/experiments/so101-homogeneous-rel/checkpoints/action_decoder_relative/stats/so101_relative_stats.json \
  --num-sampling-steps 35 \
  --stop-video-denoising-step 10
```

The server usually should not load the T5 text encoder on smaller GPUs. For the
fixed SO-101 bottle task, precompute the prompt embedding once:

```bash
cd $REPO_ROOT
bash scripts/so101/export_bottle_prompt_embedding.sh so101-homogeneous-rel
```

This writes:

```text
$MVS_ROOT/experiments/so101-homogeneous-rel/prompt_embeddings/so101_bottle_container.npy
$MVS_ROOT/experiments/so101-homogeneous-rel/prompt_embeddings/manifest.json
```

for the prompt:

```text
pick up the bottle and place it into the container
```

The robot client can then send only the prompt string. The server resolves that
prompt through the manifest and loads the cached embedding.

## Existing eval code inventory

Current checkout contains SO-101 mimic-video eval wrappers:

- Policy wrapper:
  - `eval/so101/mimic_video_so101_policy.py`
- Dataset/offline evaluator:
  - `eval/so101/offline_eval.py`
- TCP/websocket server:
  - `eval/so101/policy_server.py`
- Synthetic smoke client:
  - `eval/so101/synthetic_policy_client.py`
- Experiment-aware shell wrappers:
  - `scripts/so101/run_policy_server.sh`
  - `scripts/so101/run_synthetic_client.sh`
  - `scripts/so101/run_offline_eval.sh`

Relevant mimic-video references:

- Bridge/SimplerEnv VAM policy:
  - `eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py`
- LIBERO eval:
  - `eval/libero/run.py`

Relevant DreamZero infrastructure:

- Generic websocket server/client:
  - `/cluster/project/cvg/students/$USER/workspace/dreamzero/eval_utils/policy_server.py`
  - `/cluster/project/cvg/students/$USER/workspace/dreamzero/eval_utils/policy_client.py`
- Current DreamZero sim eval client is DROID-style and assumes 8D actions, so it is not directly compatible with SO-101 6D joint actions.

If DreamZero closed-loop eval is used, keep a thin adapter around this policy
server rather than changing the mimic-video loading path.

## Evaluation strategy

### Phase 1: dataset-free server smoke

Before robot/sim closed-loop evaluation, start the policy server from the released
checkpoints and stats JSON, then run the synthetic client.

Goals:

- Load frozen video backbone, video LoRA, action decoder, and SO-101 stats JSON.
- Avoid zarr access during server startup.
- Feed hstack two-camera images, 6D joint state, and prompt string resolved by the server-side embedding cache.
- Confirm model output shape is `(H, 6)`.
- Confirm outputs are finite and metadata reports action horizon 15 and action dim 6.

This phase validates checkpoint loading, normalization, and request/response I/O
before any online rollout.

### Phase 2: offline heldout sanity eval

Use `scripts/so101/run_offline_eval.sh` only after the optional action zarr and
hstack video dataset are present.

Goals:

- Sample heldout SO-101 zarr episodes.
- Run generated/oracle modes.
- Compute action MSE/MAE against heldout future 6D joint actions.
- Save predicted vs. target joint trajectories and action magnitude checks.

### Phase 3: SO-101 policy adapter

The `MimicVideoSO101Policy` wrapper should provide:

- `make_config()` + `experiment=...` override,
- `Video2WorldPipeline.from_config(...)`,
- `World2ActionPipeline.from_config(...)`,
- SO-101 stats loaded into `world2action_pipe.normalizer` from JSON or zarr,
- image history of 5 hstack frames,
- current 6D joint-state input,
- action chunk output `(15, 6)`,
- configurable number of actions to execute per policy query.

### Phase 4: camera mapping

SO-101 provides two images in the LeRobot dataset:

- `observation.images.front`
- `observation.images.wrist`

The active `so101-homogeneous-rel` checkpoint expects the two-camera hstack
convention:

- left half: front camera,
- right half: wrist camera,
- full frame passed through compatibility key `observation/images/front`.

Do not replace this with front-only input for this experiment; that belongs to
the older single-camera baseline.

### Phase 5: online/sim eval interface

If using DreamZero websocket infra, configure the server/client around:

- one hstack RGB image containing front and wrist cameras,
- current 6D joint state,
- output action chunk shape `(N, 6)`,
- action space: relative 6D joint delta for `so101_relative`.

The existing DROID-style client expects 8D actions, so it needs a SO-101-specific client or adapter.

## Acceptance checks

- Policy server starts from checkpoint files plus `so101_relative_stats.json` without zarr.
- Synthetic client receives finite `(15, 6)` action chunks.
- Offline eval loads the relative action decoder when optional zarr/video data are present.
- Predicted action shape is `(H, 6)`.
- Normalizer uses SO-101 statistics, not Bridge statistics.
- Hstack image preprocessing matches training: RGB, uint8 input, front-left/wrist-right.
- Closed-loop interface sends only valid 6D SO-101 joint commands after converting relative deltas to targets.
