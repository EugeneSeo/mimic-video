# SO-101 Action-Decoder Finetuning

This note documents the first SO-101 baseline path for finetuning only the mimic-video `world2action` action decoder on the LeRobot/Hugging Face dataset `dreamdifferent/so101_bottle`.

The video backbone stays frozen. The SO-101 action decoder uses native 6D joint-position state/action vectors:

- `obs/lowdim_concat`: current 6D `observation.state`
- `action/lowdim_concat`: future 6D `action`
- `workspace_rgb`: one camera stream, defaulting to `observation.images.front`

## 1. Convert LeRobot dataset to mimic zarr

Run from the repository root:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

python data_preprocessing/action/process_so101_lerobot.py \
  --repo-id dreamdifferent/so101_bottle \
  --output-dir /path/to/data/so101_bottle \
  --camera-key observation.images.front \
  --video-backend pyav
```

For Euler, submit the smoke conversion as a CPU Slurm job instead of using an interactive session:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

sbatch model/scripts/convert_so101_smoke.sbatch
```

This defaults to:

```text
OUTPUT_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke
MAX_EPISODES=2
CAMERA_KEY=observation.images.front
VIDEO_BACKEND=pyav
```

Useful overrides:

```bash
OUTPUT_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_front_full \
MAX_EPISODES= \
CAMERA_KEY=observation.images.front \
  sbatch model/scripts/convert_so101_smoke.sbatch
```

Note: `MAX_EPISODES=` intentionally means “no episode limit.” If `MAX_EPISODES` is unset, the sbatch defaults to the 2-episode smoke conversion.

For a quick smoke conversion without Slurm:

```bash
python data_preprocessing/action/process_so101_lerobot.py \
  --repo-id dreamdifferent/so101_bottle \
  --output-dir /path/to/data/so101_bottle_smoke \
  --camera-key observation.images.front \
  --video-backend pyav \
  --max-episodes 2
```

To use the wrist camera instead:

```bash
python data_preprocessing/action/process_so101_lerobot.py \
  --repo-id dreamdifferent/so101_bottle \
  --output-dir /path/to/data/so101_bottle_wrist \
  --camera-key observation.images.wrist \
  --video-backend pyav
```

The converter requires a LeRobot version with `LeRobotDataset` v3 support, plus the existing action-preprocessing dependencies such as `zarr` and `numcodecs`.

The converter defaults to `--video-backend pyav`. This avoids the `torchcodec` path, which can fail on Euler if system FFmpeg shared libraries such as `libavutil.so.*` are not visible. If you explicitly want to test TorchCodec, submit with `VIDEO_BACKEND=torchcodec`.

For training, the SO-101 sbatch also prepends `/usr/sbin:/sbin` to `PATH`, because `transformer_engine` may call `ldconfig` during import on Euler. It also points `CUDA_HOME`/`CUDA_PATH` at the venv's NVIDIA package directory and extends `LD_LIBRARY_PATH` so `transformer_engine` and Imaginaire can find `libnvrtc`, `libcudnn`, `libnccl`, and `libcudart.so`. If you run training manually and see an `ldconfig`, `libnvrtc`, or `libcudart.so` error, use:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video/model
export PATH="/usr/sbin:/sbin:$PATH"
export CUDA_HOME="$PWD/.venv/lib/python3.10/site-packages/nvidia"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/cuda_runtime/lib:$CUDA_HOME/cuda_nvrtc/lib:$CUDA_HOME/cudnn/lib:$CUDA_HOME/nccl/lib:${LD_LIBRARY_PATH:-}"
```

## 2. Precompute language embeddings

Run T5 precompute on the converted zarr directory:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

python data_preprocessing/action/precompute_t5.py \
  --dataset-path /path/to/data/so101_bottle
```

For Euler, submit the smoke T5 precompute as a GPU Slurm job because the repository T5 encoder runs on CUDA by default:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

sbatch model/scripts/precompute_so101_t5_smoke.sbatch
```

This defaults to:

```text
DATA_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke
GPU=nvidia_a100_80gb_pcie:1
```

Useful override:

```bash
DATA_DIR=/cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke \
PROMPT="pick up the bottle and place it into the container" \
  sbatch model/scripts/precompute_so101_t5_smoke.sbatch
```

For a direct shell run on an interactive GPU allocation:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video/model

python ../data_preprocessing/action/precompute_t5.py \
  --dataset-path /cluster/scratch/eugseo/mimic_video_data/so101_bottle_smoke
```

## 3. Training dry run

The SO-101 config is registered as `data_config=so101`. The converted data directory is read from `SO101_DATA_DIR`:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video/model

SO101_DATA_DIR=/path/to/data/so101_bottle \
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  --dryrun \
  -- experiment=w2a_so101_random_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz16
```

Partial Bridge-initialized decoder dry run:

```bash
SO101_DATA_DIR=/path/to/data/so101_bottle \
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  --dryrun \
  -- experiment=w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz16
```

## 4. Finetuning commands

### Slurm smoke test on Euler

For Euler, prefer submitting even smoke tests through Slurm:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video

DATA_DIR=/path/to/data/so101_bottle_smoke \
  sbatch model/scripts/train_so101_smoke.sbatch
```

The smoke sbatch sets `SO101_NUM_VAL_EPISODES=0` by default because the smoke conversion contains only two episodes. For full-dataset training, use e.g. `SO101_NUM_VAL_EPISODES=10`.

The smoke sbatch requests one GPU plus `gpumem:80G` by default and uses `GLOBAL_BATCH_SIZE=4`. On Euler, this is more reliable than typed GPU GRES for selecting an 80GB A100-class node. A 24GB RTX 4090 can load the model but OOMs at `bsz16`; an A100 40GB also OOMed at `bsz16` while encoding the video batch through the frozen tokenizer/VAE.

For smoke robustness on Euler, the sbatch also defaults to:

```text
NUM_WORKERS=1
MIMIC_DATASET_STATS_NUM_WORKERS=0
MIMIC_DATASET_STATS_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=4
RUN_TMP_DIR=/tmp/mvs_${SLURM_JOB_ID}
```

The short `/tmp` path avoids Python multiprocessing `AF_UNIX path too long` failures that can happen if `TMPDIR` includes the long experiment name. `NUM_WORKERS=1` keeps PyTorch DataLoader multiprocessing enabled because the upstream dataloader config sets `prefetch_factor`; set it higher for larger full-dataset runs after the smoke path is stable.

Useful overrides:

```bash
DATA_DIR=/path/to/data/so101_bottle \
INIT_NAME=random \
VIDEO_CKPT=v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused \
LR_TAG=1.000e-04 \
GLOBAL_BATCH_SIZE=4 \
MAX_ITER=100 \
RUN_VALIDATION=false \
SO101_NUM_VAL_EPISODES=0 \
  sbatch model/scripts/train_so101_smoke.sbatch
```

The default smoke run uses `MAX_ITER=50`, which completed far enough to save checkpoints within a 2h allocation on an 80GB A100-class GPU. Use `MAX_ITER=100` only with a longer wall-time request.

For partial Bridge initialization:

```bash
DATA_DIR=/path/to/data/so101_bottle \
INIT_NAME=partial_bridge_init \
  sbatch model/scripts/train_so101_smoke.sbatch
```

Logs are written to:

```text
/cluster/scratch/eugseo/mimic_video_logs/%x-%j.out
/cluster/scratch/eugseo/mimic_video_logs/%x-%j.err
```

Training checkpoints and local trainer output are redirected to scratch by default:

```text
IMAGINAIRE_OUTPUT_ROOT=/cluster/scratch/eugseo/mimic_video_runs
```

W&B logging is enabled by default in the SO-101 smoke sbatch and points to the team entity:

```text
WANDB_ENTITY=dreamdifferent
WANDB_PROJECT=mimic-video-so101
WANDB_MODE=online
WANDB_DIR=/cluster/scratch/eugseo/mimic_video_wandb
```

Before submitting online W&B jobs, authenticate once in an interactive shell:

```bash
wandb login
```

To disable networked W&B for a smoke run:

```bash
WANDB_ENABLED=0 DATA_DIR=/path/to/data/so101_bottle_smoke   sbatch model/scripts/train_so101_smoke.sbatch
```

Or keep local W&B files for later sync:

```bash
WANDB_MODE=offline DATA_DIR=/path/to/data/so101_bottle_smoke   sbatch model/scripts/train_so101_smoke.sbatch
```

### Direct shell commands

For direct commands, set `SO101_DATA_DIR` because the SO-101 Hydra data config resolves `dataset.dataset.data_dir` from this environment variable. The Slurm smoke script sets this automatically from `DATA_DIR`.


Recommended first smoke run on a single GPU:

```bash
cd /cluster/project/cvg/students/eugseo/workspace/mimic-video/model

SO101_DATA_DIR=/path/to/data/so101_bottle \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_so101_random_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4 \
     trainer.max_iter=500 \
     trainer.validation_iter=100 \
     checkpoint.save_iter=100
```

Main random-init run:

```bash
SO101_DATA_DIR=/path/to/data/so101_bottle \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_so101_random_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4
```

Main partial Bridge-init run:

```bash
SO101_DATA_DIR=/path/to/data/so101_bottle \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_so101_partial_bridge_init_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz4
```

Available SO-101 presets:

- init: `random`, `partial_bridge_init`
- video backbone: `v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused` (recommended first), `v2w_pretrained_cosmos`
- LR: `1.000e-04`, `3.000e-04`
- global batch: `bsz4`, `bsz8`, `bsz16`, `bsz32`, `bsz64`

Example larger-batch A100-style preset:

```bash
SO101_DATA_DIR=/path/to/data/so101_bottle \
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_so101_random_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz64 \
```

## 5. Checkpoints needed

At minimum, the random-init run needs the selected frozen video backbone and text/tokenizer assets expected by mimic-video. The Slurm smoke script defaults to the released Bridge-finetuned video backbone because SO-101 is also single-arm manipulation.

The partial Bridge-init run additionally expects the action decoder matching the selected video backbone. For the recommended Bridge-finetuned video backbone:

```text
model/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt
```

For the base Cosmos video backbone:

```text
model/checkpoints/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt
```

If this file is missing, either download the pretrained Cosmos Bridge checkpoint bundle or use the `random` SO-101 experiment first.

## 6. Quick sanity checks

After conversion, each zarr episode should contain:

```text
workspace_rgb
workspace_rgb_timestamps
joint_state_lowdim
joint_state_lowdim_timestamps
joint_action_lowdim
joint_action_lowdim_timestamps
language_instruction
language_instruction_timestamps
language_embedding
language_embedding_timestamps
```

Expected model batch dimensions after dataloading:

- `obs/lowdim_concat.shape[-1] == 6`
- `action/lowdim_concat.shape[-1] == 6`
- `obs/workspace_rgb` and `action/workspace_rgb` are single-view video tensors.
