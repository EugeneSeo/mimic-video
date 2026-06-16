"""Convert local LeRobot v3 SO-101 joint data to EE-pose action zarrs.

This converter is intentionally local-only: it reads an already materialized
LeRobot v3 dataset root and never downloads from the Hub. The output is a
Bridge-style action dataset with absolute end-effector pose matrices stored at
the source timestamps. MimicDataset can later sample those poses at 5Hz and
convert action poses to relative EE space.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tqdm

S_TO_NS = 1_000_000_000
DEFAULT_MOTOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass(frozen=True)
class FeatureLayout:
    action_names: list[str]
    state_names: list[str]
    arm_action_indices: list[int]
    arm_state_indices: list[int]
    gripper_action_index: int
    gripper_state_index: int


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_zarr() -> tuple[Any, Any]:
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError(
            "SO101 EE conversion requires `zarr` and `numcodecs` in the active Python environment. "
            "Use the model environment or install those packages into the LeRobot environment before conversion."
        ) from exc
    return zarr, Blosc


def _load_tasks(root: pathlib.Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {0: ""}

    tasks = pd.read_parquet(tasks_path)
    if "task_index" in tasks.columns:
        index_values = tasks["task_index"].to_numpy()
    else:
        index_values = np.arange(len(tasks))

    if "task" in tasks.columns:
        task_values = tasks["task"].astype(str).to_numpy()
    else:
        task_values = tasks.index.astype(str).to_numpy()

    return {int(idx): str(task) for idx, task in zip(index_values, task_values, strict=False)}


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    try:
        names = info["features"][key]["names"]
    except KeyError as exc:
        available = ", ".join(sorted(info.get("features", {})))
        raise KeyError(f"Feature {key!r} not found. Available features: {available}") from exc
    if not isinstance(names, list):
        raise ValueError(f"Feature {key!r} does not expose named dimensions.")
    return [str(name) for name in names]


def _build_feature_layout(
    info: dict[str, Any],
    *,
    action_key: str,
    state_key: str,
    motor_names: tuple[str, ...],
    gripper_name: str,
) -> FeatureLayout:
    action_names = _feature_names(info, action_key)
    state_names = _feature_names(info, state_key)

    def index(names: list[str], dim_name: str) -> int:
        try:
            return names.index(dim_name)
        except ValueError as exc:
            raise ValueError(f"Dimension {dim_name!r} not found in {names}") from exc

    arm_dim_names = [f"{name}.pos" for name in motor_names]
    gripper_dim_name = f"{gripper_name}.pos"
    return FeatureLayout(
        action_names=action_names,
        state_names=state_names,
        arm_action_indices=[index(action_names, name) for name in arm_dim_names],
        arm_state_indices=[index(state_names, name) for name in arm_dim_names],
        gripper_action_index=index(action_names, gripper_dim_name),
        gripper_state_index=index(state_names, gripper_dim_name),
    )


def _stack_column(series: pd.Series, key: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in series]
    if not values:
        raise ValueError(f"No rows found for {key!r}.")
    return np.stack(values, axis=0)


def _timestamps_ns(frame: pd.DataFrame, fps: float) -> np.ndarray:
    if "timestamp" in frame.columns:
        timestamps_s = frame["timestamp"].to_numpy(dtype=np.float64)
    else:
        timestamps_s = np.arange(len(frame), dtype=np.float64) / fps
    return np.rint(timestamps_s * S_TO_NS).astype(np.uint64)


def _poses_from_joints(kinematics: Any, joints_deg: np.ndarray) -> np.ndarray:
    poses = np.empty((len(joints_deg), 4, 4), dtype=np.float32)
    for idx, joint_pos in enumerate(joints_deg):
        poses[idx] = kinematics.forward_kinematics(joint_pos).astype(np.float32)
    return poses


def _write_array(root: Any, name: str, array: np.ndarray, *, chunk_t: int) -> None:
    _, Blosc = _require_zarr()
    root.create_dataset(
        name,
        shape=array.shape,
        dtype=array.dtype,
        chunks=(min(chunk_t, len(array)), *array.shape[1:]),
        compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
    )
    root[name][...] = array


def _write_timestamps(root: Any, name: str, timestamps_ns: np.ndarray) -> None:
    _, Blosc = _require_zarr()
    root.create_dataset(
        f"{name}_timestamps",
        shape=timestamps_ns.shape,
        dtype="uint64",
        chunks=(len(timestamps_ns),),
        compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
    )
    root[f"{name}_timestamps"][...] = timestamps_ns


def _write_episode(
    episode: pd.DataFrame,
    *,
    output_dir: pathlib.Path,
    action_key: str,
    state_key: str,
    layout: FeatureLayout,
    kinematics: Any,
    tasks: dict[int, str],
    default_task: str,
    fps: float,
    overwrite: bool,
) -> pathlib.Path:
    episode = episode.sort_values("frame_index") if "frame_index" in episode.columns else episode
    episode_index = int(np.asarray(episode["episode_index"].iloc[0]).reshape(-1)[0])
    task_index = int(np.asarray(episode["task_index"].iloc[0]).reshape(-1)[0]) if "task_index" in episode else 0

    states = _stack_column(episode[state_key], state_key)
    actions = _stack_column(episode[action_key], action_key)
    state_joints = states[:, layout.arm_state_indices]
    action_joints = actions[:, layout.arm_action_indices]
    gripper_state = states[:, layout.gripper_state_index : layout.gripper_state_index + 1].astype(np.float32)
    gripper_action = actions[:, layout.gripper_action_index : layout.gripper_action_index + 1].astype(np.float32)
    timestamps_ns = _timestamps_ns(episode, fps=fps)

    if not (len(state_joints) == len(action_joints) == len(gripper_state) == len(timestamps_ns)):
        raise ValueError(f"Episode {episode_index} has inconsistent lengths.")

    state_poses = _poses_from_joints(kinematics, state_joints)
    action_poses = _poses_from_joints(kinematics, action_joints)

    out_path = output_dir / f"episode_{episode_index:06d}.zarr"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output episode exists: {out_path}. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zarr, Blosc = _require_zarr()
    with zarr.open(str(out_path), "w") as root:
        _write_array(root, "eef_state_pose_lowdim", state_poses, chunk_t=1024)
        _write_timestamps(root, "eef_state_pose_lowdim", timestamps_ns)
        _write_array(root, "eef_action_pose_lowdim", action_poses, chunk_t=1024)
        _write_timestamps(root, "eef_action_pose_lowdim", timestamps_ns)
        _write_array(root, "gripper_state_lowdim", gripper_state, chunk_t=1024)
        _write_timestamps(root, "gripper_state_lowdim", timestamps_ns)
        _write_array(root, "gripper_action_lowdim", gripper_action, chunk_t=1024)
        _write_timestamps(root, "gripper_action_lowdim", timestamps_ns)

        language = tasks.get(task_index) or default_task
        root.create_dataset(
            "language_instruction",
            shape=(1,),
            dtype=bytes,
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction"][...] = np.array([language.encode("utf-8")])
        root.create_dataset(
            "language_instruction_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction_timestamps"][...] = np.array([0], dtype=np.uint64)

    return out_path


def _iter_dataframes(root: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True, help="Local LeRobot v3 dataset root.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write .zarr episodes.")
    parser.add_argument("--urdf-path", type=pathlib.Path, required=True, help="Local SO101 URDF path.")
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--motor-names", default=",".join(DEFAULT_MOTOR_NAMES))
    parser.add_argument("--gripper-name", default="gripper")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode indices to convert.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional first-N episode limit for smoke tests.")
    parser.add_argument("--default-task", default="pick up the bottle")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect metadata and run FK on the first row only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = _load_json(args.root / "meta" / "info.json")
    tasks = _load_tasks(args.root)
    motor_names = tuple(name.strip() for name in args.motor_names.split(",") if name.strip())
    layout = _build_feature_layout(
        info,
        action_key=args.action_key,
        state_key=args.state_key,
        motor_names=motor_names,
        gripper_name=args.gripper_name,
    )

    from lerobot.model.kinematics import RobotKinematics

    kinematics = RobotKinematics(
        urdf_path=str(args.urdf_path),
        target_frame_name=args.target_frame_name,
        joint_names=list(motor_names),
    )

    data_paths = _iter_dataframes(args.root)
    selected = set(args.episodes) if args.episodes is not None else None
    written = 0

    print(
        "SO101 EE conversion: "
        f"root={args.root}, fps={args.fps:g}, target_frame={args.target_frame_name}, "
        f"motors={list(motor_names)}, selected={sorted(selected) if selected is not None else 'all'}"
    )

    for parquet_path in tqdm.tqdm(data_paths, desc="Reading LeRobot parquet files"):
        frame = pd.read_parquet(parquet_path)
        if selected is not None:
            frame = frame[frame["episode_index"].isin(selected)]
        if frame.empty:
            continue

        for episode_index, episode in frame.groupby("episode_index", sort=True):
            if args.max_episodes is not None and written >= args.max_episodes:
                return
            if args.dry_run:
                states = _stack_column(episode[args.state_key].iloc[:1], args.state_key)
                pose = kinematics.forward_kinematics(states[0, layout.arm_state_indices])
                print(f"episode={int(episode_index)} rows={len(episode)} first_ee_pose=\n{pose}")
                return

            out_path = _write_episode(
                episode,
                output_dir=args.output_dir,
                action_key=args.action_key,
                state_key=args.state_key,
                layout=layout,
                kinematics=kinematics,
                tasks=tasks,
                default_task=args.default_task,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            print(f"wrote {out_path}", flush=True)
            written += 1

    print(f"Wrote {written} SO101 EE episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
