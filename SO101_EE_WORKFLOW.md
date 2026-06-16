# SO101 EE-Space Data Workflow

This workflow converts the LeRobot v3 `dreamdifferent/so101_bottle` style data
into the mimic-video action format used by the Bridge-style 10D action head.

## Representation

The source dataset is 30Hz SO101 joint-space data:

- `observation.state`: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`
- `action`: the same six joint dimensions
- `observation.images.front` and `observation.images.wrist`

The action converter runs SO101 forward kinematics on the five arm joints and
writes 30Hz absolute EE pose matrices:

- `eef_state_pose_lowdim`
- `eef_action_pose_lowdim`
- `gripper_state_lowdim`
- `gripper_action_lowdim`

The dataloader config `so101_ee` samples those 30Hz poses at 5Hz, converts
future action EE poses relative to the current observed EE pose, converts
rotation matrices to 6D, and builds the usual 10D action vector:

```text
[relative_ee_xyz(3), relative_ee_rot6d(6), gripper(1)]
```

## Video

SO101 has two cameras, but this repo's Bridge-style video/action path expects
one `workspace_rgb` stream. The video converter composes a single 480x640 frame:

- front camera: left 480x320 half
- wrist camera: right 480x320 half

It also performs real temporal downsampling from 30Hz to 5Hz by keeping every
sixth frame. The action Zarr can omit images; `MimicDataset` reads
`workspace_rgb` from the external hstack MP4 directory via `SO101_VIDEO_DIR`.

## Required Local Inputs

Do not run full dataset downloads or conversion inside an interactive agent
session. Materialize these paths first, preferably through a Slurm job:

- `LEROBOT_ROOT`: local LeRobot v3 dataset root containing `meta/`, `data/`, and `videos/`.
- `URDF_PATH`: local SO101 URDF, usually `SO-ARM100/Simulation/SO101/so101_new_calib.urdf`.
- `OUTPUT_DIR`: conversion output for each job.

The default conda environment is:

```bash
/cluster/home/dohkim/miniforge3/envs/lerobot
```

The action EE conversion environment must contain:

```text
lerobot, pandas, zarr, numcodecs, placo
```

At implementation time, the default `lerobot` conda environment was missing
`zarr`, `numcodecs`, and `placo`, while `model/.venv` was also missing `placo`.
Do not install packages from an interactive agent session. Prepare an
environment with the required packages first, then point the Slurm script to it
with either:

```bash
CONDA_ENV_PATH=/path/to/env
```

or:

```bash
PYTHON_BIN=/path/to/env/bin/python
```

## Smoke Conversion

Video smoke conversion:

```bash
LEROBOT_ROOT=/path/to/so101_bottle \
OUTPUT_DIR=/cluster/scratch/$USER/mimic_video/shared/video_data/so101_bottle_front_wrist_hstack_5fps_smoke \
MAX_EPISODES=1 \
OVERWRITE=1 \
sbatch model/scripts/convert_so101_two_camera_video.sbatch
```

Action smoke conversion:

```bash
LEROBOT_ROOT=/path/to/so101_bottle \
URDF_PATH=/path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
OUTPUT_DIR=/cluster/scratch/$USER/mimic_video/shared/data/so101_bottle_ee_action_smoke \
MAX_EPISODES=1 \
OVERWRITE=1 \
sbatch model/scripts/convert_so101_ee_action.sbatch
```

Kinematics-only dry run:

```bash
LEROBOT_ROOT=/path/to/so101_bottle \
URDF_PATH=/path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
OUTPUT_DIR=/tmp/unused_so101_ee \
DRY_RUN=1 \
sbatch model/scripts/convert_so101_ee_action.sbatch
```

## Training Config

After conversion and T5 precompute, set:

```bash
export SO101_EE_DATA_DIR=/path/to/so101_ee_action_zarr
export SO101_VIDEO_DIR=/path/to/so101_hstack_5fps_video_dataset
export SO101_VIDEO_FPS=5
```

Then use `data_config=so101_ee`. It uses `world2action_pipe=so101_ee`,
which is a Bridge-compatible 10D action head with `max_horizon=16`.

The 30Hz EE Zarr can be reused for later 30Hz EE experiments. For the first
implementation, keep action training at 5Hz through `policy_io/so101_ee.yaml`
instead of materializing a separate 5Hz action Zarr.
