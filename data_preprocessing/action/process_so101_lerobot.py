"""Convert a LeRobot v3 SO-101 dataset to mimic-video action zarr format.

Example:
    python data_preprocessing/action/process_so101_lerobot.py \
        --repo-id dreamdifferent/so101_bottle \
        --output-dir /path/to/data/so101_bottle \
        --camera-key observation.images.front \
        --video-backend pyav \
        --max-episodes 2

The resulting zarr episodes are compatible with
``cosmos_predict2/configs/dataloading/so101.yaml``. Run
``precompute_t5.py`` on the output directory before training.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

S_TO_NS = 1_000_000_000


@dataclass
class EpisodeBuffer:
    episode_index: int
    task: str
    images: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    timestamps_ns: list[int] = field(default_factory=list)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_uint8_hwc(image: Any) -> np.ndarray:
    image_np = _to_numpy(image)
    if image_np.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image_np.shape}.")

    # LeRobot returns images as either CHW tensors or HWC arrays depending on
    # version/transform configuration.
    if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
        image_np = np.moveaxis(image_np, 0, -1)

    if image_np.dtype != np.uint8:
        # Common case: float tensor in [0, 1].
        if np.nanmax(image_np) <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    if image_np.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image_np.shape}.")

    return np.ascontiguousarray(image_np)


def _scalar_int(value: Any) -> int:
    value_np = _to_numpy(value)
    return int(value_np.reshape(-1)[0])


def _scalar_float(value: Any) -> float:
    value_np = _to_numpy(value)
    return float(value_np.reshape(-1)[0])


def _get_task(dataset: Any, sample: dict[str, Any], default_task: str) -> str:
    if "task" in sample and sample["task"]:
        return str(sample["task"])

    task_index = _scalar_int(sample.get("task_index", 0))
    meta = getattr(dataset, "meta", None)
    tasks = getattr(meta, "tasks", None)
    if tasks is None and meta is not None:
        tasks = getattr(meta, "tasks_table", None)

    if isinstance(tasks, dict):
        task = tasks.get(task_index) or tasks.get(str(task_index))
        if task is not None:
            return str(task)

    try:
        task = tasks[task_index]
        if isinstance(task, dict):
            return str(task.get("task", task.get("name", default_task)))
        return str(task)
    except Exception:
        return default_task


def _iter_episode_buffers(
    dataset: Any,
    *,
    camera_key: str,
    max_episodes: int | None,
    default_task: str,
    fps: float,
    skip_video: bool,
) -> Iterable[EpisodeBuffer]:
    current: EpisodeBuffer | None = None
    emitted = 0

    for i in tqdm.trange(len(dataset), desc="Reading LeRobot frames"):
        sample = dataset[i]
        episode_index = _scalar_int(sample["episode_index"])

        if current is None or episode_index != current.episode_index:
            if current is not None:
                yield current
                emitted += 1
                if max_episodes is not None and emitted >= max_episodes:
                    return
            current = EpisodeBuffer(
                episode_index=episode_index,
                task=_get_task(dataset, sample, default_task=default_task),
            )

        if not skip_video and camera_key not in sample:
            available = ", ".join(sorted(k for k in sample if k.startswith("observation.images.")))
            raise KeyError(f"Camera key {camera_key!r} not found. Available image keys: {available}")

        if not skip_video:
            current.images.append(_to_uint8_hwc(sample[camera_key]))
        current.states.append(_to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1))
        current.actions.append(_to_numpy(sample["action"]).astype(np.float32).reshape(-1))

        if "timestamp" in sample:
            timestamp_s = _scalar_float(sample["timestamp"])
        else:
            timestamp_s = len(current.timestamps_ns) / fps
        current.timestamps_ns.append(int(round(timestamp_s * S_TO_NS)))

    if current is not None and (max_episodes is None or emitted < max_episodes):
        yield current


def _write_array(root: zarr.Group, name: str, array: np.ndarray, *, chunk_t: int) -> None:
    root.create_dataset(
        name,
        shape=array.shape,
        dtype=array.dtype,
        chunks=(min(chunk_t, len(array)), *array.shape[1:]),
        compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
    )
    root[name][...] = array


def _write_timestamps(root: zarr.Group, name: str, timestamps_ns: np.ndarray) -> None:
    root.create_dataset(
        f"{name}_timestamps",
        shape=timestamps_ns.shape,
        dtype="uint64",
        chunks=(len(timestamps_ns),),
        compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
    )
    root[f"{name}_timestamps"][...] = timestamps_ns


def write_episode(buffer: EpisodeBuffer, output_dir: pathlib.Path, *, skip_video: bool) -> None:
    images = None if skip_video else np.stack(buffer.images, axis=0).astype(np.uint8)
    states = np.stack(buffer.states, axis=0).astype(np.float32)
    actions = np.stack(buffer.actions, axis=0).astype(np.float32)
    timestamps_ns = np.asarray(buffer.timestamps_ns, dtype=np.uint64)

    image_len = len(timestamps_ns) if images is None else len(images)
    if not (image_len == len(states) == len(actions) == len(timestamps_ns)):
        raise ValueError(
            "Episode arrays have inconsistent lengths: "
            f"images={image_len}, states={len(states)}, actions={len(actions)}, timestamps={len(timestamps_ns)}"
        )
    if states.shape[-1] != 6 or actions.shape[-1] != 6:
        raise ValueError(f"Expected SO-101 6D state/action, got state={states.shape}, action={actions.shape}.")

    out_path = output_dir / f"episode_{buffer.episode_index:06d}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zarr.open(str(out_path), "w") as root:
        if images is not None:
            _write_array(root, "workspace_rgb", images, chunk_t=65)
            _write_timestamps(root, "workspace_rgb", timestamps_ns)

        _write_array(root, "joint_state_lowdim", states, chunk_t=1024)
        _write_timestamps(root, "joint_state_lowdim", timestamps_ns)

        _write_array(root, "joint_action_lowdim", actions, chunk_t=1024)
        _write_timestamps(root, "joint_action_lowdim", timestamps_ns)

        root.create_dataset(
            "language_instruction",
            shape=(1,),
            dtype=bytes,
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction"][...] = np.array([buffer.task.encode("utf-8")])
        root.create_dataset(
            "language_instruction_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction_timestamps"][...] = np.array([0], dtype=np.uint64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="dreamdifferent/so101_bottle", help="HF/LeRobot dataset repo id.")
    parser.add_argument("--root", type=pathlib.Path, default=None, help="Optional local LeRobot cache/root directory.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write .zarr episodes.")
    parser.add_argument(
        "--camera-key",
        default="observation.images.front",
        choices=("observation.images.front", "observation.images.wrist"),
        help="Single camera stream to map to mimic-video workspace_rgb.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for smoke conversion.")
    parser.add_argument(
        "--default-task",
        default="pick up the bottle",
        help="Fallback language instruction if the dataset API does not expose task text.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS when samples do not include timestamps.")
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Write only state/action/language data. Use SO101_VIDEO_DIR at training time for workspace_rgb.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "torchcodec", "video_reader"),
        help=(
            "LeRobot video decoder backend. Default to pyav because torchcodec can require "
            "system FFmpeg shared libraries that are often unavailable on clusters."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "This converter requires LeRobot. Install a version with LeRobotDataset v3 support, "
            "for example in your training env: `pip install lerobot`."
        ) from exc

    dataset_kwargs = {"repo_id": args.repo_id, "video_backend": args.video_backend}
    if args.root is not None:
        dataset_kwargs["root"] = args.root
    dataset = LeRobotDataset(**dataset_kwargs)

    num_written = 0
    for buffer in _iter_episode_buffers(
        dataset,
        camera_key=args.camera_key,
        max_episodes=args.max_episodes,
        default_task=args.default_task,
        fps=args.fps,
        skip_video=args.skip_video,
    ):
        write_episode(buffer, args.output_dir, skip_video=args.skip_video)
        num_written += 1

    print(f"Wrote {num_written} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
