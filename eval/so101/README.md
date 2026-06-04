# SO-101 Mimic Video Eval

This directory contains SO-101 evaluation utilities for Mimic Video:

```text
offline_eval.py              dataset sample eval, generated/oracle modes
policy_server.py             TCP/websocket prediction server
synthetic_policy_client.py   random-image server smoke test
run_server.sh                thin wrapper around policy_server.py
```

The current baseline uses the two-camera hstack frame described in
`SO101_VIDEO_WORKFLOW.md`. Online requests still use the key
`observation/images/front` for compatibility, but the image value should be the
480x640 hstack frame:

```text
left half:  front camera
right half: wrist camera
```

## Default Eval Settings

Use Mimic Video's action-eval convention:

```text
num_sampling_steps = 35
stop_video_denoising_step = 10
```

Training-time W&B video previews may use 20 video sampling steps because they
are only a lightweight visual progress check.

## Checkpoints

Common paths:

```text
Base video backbone:
/cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt

SO-101 bottle video LoRA:
/cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt

Absolute action decoder:
/cluster/scratch/eugseo/mimic_video_runs_so101_w2a_bridge_init_so101_v2w_lora_hstack_video_5hz_action_resume_to_2000/vam/so101/w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002000.pt

Relative action decoder:
/cluster/scratch/eugseo/mimic_video_runs_so101_w2a_relative_bridge_init_so101_v2w_lora_hstack_video_5hz_action_24h_part1/vam/so101_relative/w2a_so101_relative_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002500.pt

Action zarr:
/cluster/scratch/eugseo/mimic_video_data/so101_bottle_action_only_full

External hstack video dataset:
/cluster/scratch/eugseo/mimic_video_video_data/so101_bottle_front_wrist_hstack_5fps_full
```

## Environment

From the repository root:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video
cd model
source .venv/bin/activate
cd ..

export PATH="/usr/sbin:/sbin:$PATH"
export CUDA_HOME="$PWD/model/.venv/lib/python3.10/site-packages/nvidia"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/cuda_runtime/lib:$CUDA_HOME/cuda_nvrtc/lib:$CUDA_HOME/cudnn/lib:$CUDA_HOME/nccl/lib:${LD_LIBRARY_PATH:-}"

export SO101_VIDEO_DIR=/cluster/scratch/eugseo/mimic_video_video_data/so101_bottle_front_wrist_hstack_5fps_full
export SO101_VIDEO_FPS=5
export MIMIC_DATASET_STATS_NUM_WORKERS=0
export MIMIC_DATASET_STATS_BATCH_SIZE=4
```

## Offline Generated Eval

Generated mode matches the intended online path: video2world predicts future
context, then world2action predicts the action chunk.

```bash
python eval/so101/offline_eval.py \
  --split val \
  --mode generated \
  --num-samples 4 \
  --num-val-episodes 10 \
  --experiment-name w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
  --video-model-path /cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  --video-lora-path /cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt \
  --action-model-path /cluster/scratch/eugseo/mimic_video_runs_so101_w2a_bridge_init_so101_v2w_lora_hstack_video_5hz_action_resume_to_2000/vam/so101/w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002000.pt \
  --data-dir /cluster/scratch/eugseo/mimic_video_data/so101_bottle_action_only_full \
  --num-sampling-steps 35 \
  --stop-video-denoising-step 10 \
  --output-dir /cluster/scratch/eugseo/mvs_so101_offline_generated_val_4_35_10
```

Outputs:

```text
summary.json
policy_metadata.json
generated_per_sample.csv
generated_trajectories.npz
generated_sample0_trajectory.png
```

## Offline Oracle Eval

Oracle mode feeds GT future video features to isolate action-decoder quality:

```bash
python eval/so101/offline_eval.py \
  --split val \
  --mode oracle \
  --num-samples 4 \
  --num-val-episodes 10 \
  --seed 0 \
  --experiment-name w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
  --video-model-path /cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  --video-lora-path /cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt \
  --action-model-path /cluster/scratch/eugseo/mimic_video_runs_so101_w2a_bridge_init_so101_v2w_lora_hstack_video_5hz_action_resume_to_2000/vam/so101/w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002000.pt \
  --data-dir /cluster/scratch/eugseo/mimic_video_data/so101_bottle_action_only_full \
  --num-sampling-steps 35 \
  --stop-video-denoising-step 10 \
  --output-dir /cluster/scratch/eugseo/mvs_so101_offline_oracle_val_4_35_10
```

## Video Dump

Use a small sample count because this decodes generated videos:

```bash
python eval/so101/offline_eval.py \
  --split val \
  --mode generated \
  --num-samples 2 \
  --num-val-episodes 10 \
  --experiment-name w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
  --video-model-path /cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  --video-lora-path /cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt \
  --action-model-path /cluster/scratch/eugseo/mimic_video_runs_so101_w2a_bridge_init_so101_v2w_lora_hstack_video_5hz_action_resume_to_2000/vam/so101/w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002000.pt \
  --data-dir /cluster/scratch/eugseo/mimic_video_data/so101_bottle_action_only_full \
  --num-sampling-steps 35 \
  --stop-video-denoising-step 10 \
  --dump-videos \
  --output-dir /cluster/scratch/eugseo/mvs_so101_offline_generated_val_2_video_35_10
```

This writes `videos/*_input_context.mp4`, `videos/*_gt_future.mp4`,
`videos/*_generated_full.mp4`, and `videos/*_generated_future.mp4`.

## Prediction Server

Terminal 1:

```bash
python eval/so101/policy_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --experiment-name w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
  --video-model-path /cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  --video-lora-path /cluster/scratch/eugseo/mimic_video_runs_so101_2cam_v2w_full_5fps_24h_2gpu_bsz2_acc8/posttraining/video2world_so101_two_camera/v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4/checkpoints/model/iter_000001000.pt \
  --action-model-path /cluster/scratch/eugseo/mimic_video_runs_so101_w2a_bridge_init_so101_v2w_lora_hstack_video_5hz_action_resume_to_2000/vam/so101/w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4/checkpoints/model/iter_000002000.pt \
  --data-dir /cluster/scratch/eugseo/mimic_video_data/so101_bottle_action_only_full \
  --num-val-episodes 0 \
  --num-sampling-steps 35 \
  --stop-video-denoising-step 10
```

Terminal 2 synthetic smoke:

```bash
python eval/so101/synthetic_policy_client.py \
  --host 127.0.0.1 \
  --port 8000 \
  --use-prompt-embedding \
  --num-requests 3 \
  --image-height 480 \
  --image-width 640
```

The synthetic client sends random images and zero prompt embeddings, so action
values are not meaningful. It only validates model loading, LoRA injection,
transport, output shape, finite values, and latency.

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

