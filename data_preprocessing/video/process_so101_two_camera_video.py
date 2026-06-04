"""Convert LeRobot SO-101 videos into a Cosmos video2world dataset.

The output format is the standard video2world folder layout:

    output_dir/
    ├── video/episode_000000.mp4
    └── metas/episode_000000.txt

Each output frame is a fixed 480x640 RGB canvas by default. The default layout
keeps backward-compatible two-view frames: front on the left, wrist on the right,
with each view resized/padded into a 480x320 half. Single-view front/wrist
layouts are available for ablations without changing the downstream V2W training
code.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Any

import imageio.v2 as iio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class EpisodeVideoBuffer:
    episode_index: int
    task: str
    source_frames: int = 0
    frames: list[np.ndarray] = field(default_factory=list)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_uint8_hwc(image: Any) -> np.ndarray:
    image_np = _to_numpy(image)
    if image_np.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image_np.shape}.")
    if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
        image_np = np.moveaxis(image_np, 0, -1)
    if image_np.dtype != np.uint8:
        if np.nanmax(image_np) <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    if image_np.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image_np.shape}.")
    return np.ascontiguousarray(image_np)


def _scalar_int(value: Any) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _resize_pad(image: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(out_w / w, out_h / h)
    resized_w = max(int(round(w * scale)), 1)
    resized_h = max(int(round(h * scale)), 1)
    resized = Image.fromarray(image, "RGB").resize((resized_w, resized_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), 0)
    canvas.paste(resized, ((out_w - resized_w) // 2, (out_h - resized_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def compose_frame(
    front: np.ndarray,
    wrist: np.ndarray,
    *,
    height: int,
    width: int,
    view_layout: str,
) -> np.ndarray:
    if view_layout == "front":
        return _resize_pad(front, out_h=height, out_w=width)
    if view_layout == "wrist":
        return _resize_pad(wrist, out_h=height, out_w=width)
    if view_layout == "hstack":
        return compose_two_view_frame(front, wrist, height=height, width=width)
    raise ValueError(f"Unknown view layout: {view_layout}")


def compose_two_view_frame(
    front: np.ndarray,
    wrist: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if width % 2 != 0:
        raise ValueError(f"Two-camera horizontal canvas requires even width, got {width}")
    half_w = width // 2
    front_half = _resize_pad(front, out_h=height, out_w=half_w)
    wrist_half = _resize_pad(wrist, out_h=height, out_w=half_w)
    return np.concatenate([front_half, wrist_half], axis=1)


def _get_task(dataset: Any, sample: dict[str, Any], default_task: str) -> str:
    if "task" in sample and sample["task"]:
        return str(sample["task"])
    task_index = _scalar_int(sample.get("task_index", 0))
    meta = getattr(dataset, "meta", None)
    tasks = getattr(meta, "tasks", None) if meta is not None else None
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


def iter_episode_buffers(
    dataset: Any,
    *,
    front_key: str,
    wrist_key: str,
    max_episodes: int | None,
    default_task: str,
    height: int,
    width: int,
    source_fps: float,
    target_fps: float,
    view_layout: str,
):
    ratio = source_fps / target_fps
    stride = int(round(ratio))
    if stride <= 0 or not np.isclose(ratio, stride, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Expected source_fps/target_fps to be a positive integer ratio, got "
            f"{source_fps}/{target_fps}={ratio}."
        )

    current: EpisodeVideoBuffer | None = None
    emitted = 0
    for idx in tqdm(range(len(dataset)), desc="Reading SO-101 frames"):
        sample = dataset[idx]
        episode_index = _scalar_int(sample["episode_index"])
        if current is None or episode_index != current.episode_index:
            if current is not None:
                yield current
                emitted += 1
                if max_episodes is not None and emitted >= max_episodes:
                    return
            current = EpisodeVideoBuffer(
                episode_index=episode_index,
                task=_get_task(dataset, sample, default_task=default_task),
            )

        required_keys = [front_key] if view_layout == "front" else [wrist_key] if view_layout == "wrist" else [front_key, wrist_key]
        missing = [key for key in required_keys if key not in sample]
        if missing:
            available = ", ".join(sorted(k for k in sample if k.startswith("observation.images.")))
            raise KeyError(f"Missing camera keys {missing}. Available image keys: {available}")

        source_frame_index = current.source_frames
        current.source_frames += 1
        if source_frame_index % stride == 0:
            current.frames.append(
                compose_frame(
                    _to_uint8_hwc(sample[front_key]) if front_key in sample else np.zeros((height, width, 3), dtype=np.uint8),
                    _to_uint8_hwc(sample[wrist_key]) if wrist_key in sample else np.zeros((height, width, 3), dtype=np.uint8),
                    height=height,
                    width=width,
                    view_layout=view_layout,
                )
            )

    if current is not None and (max_episodes is None or emitted < max_episodes):
        yield current


def write_episode(buffer: EpisodeVideoBuffer, output_dir: pathlib.Path, *, fps: float, overwrite: bool) -> None:
    stem = f"episode_{buffer.episode_index:06d}"
    video_path = output_dir / "video" / f"{stem}.mp4"
    meta_path = output_dir / "metas" / f"{stem}.txt"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists() and not overwrite:
        raise FileExistsError(f"Output video exists: {video_path}. Use --overwrite to replace outputs.")

    with iio.get_writer(
        video_path,
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18"],
    ) as writer:
        for frame in buffer.frames:
            writer.append_data(frame)
    meta_path.write_text(buffer.task.strip() + "\n", encoding="utf-8")
    print(
        f"{stem}: kept {len(buffer.frames)}/{buffer.source_frames} frames "
        f"at {fps:g} fps ({len(buffer.frames) / fps:.2f}s)",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="dreamdifferent/so101_bottle")
    parser.add_argument("--root", type=pathlib.Path, default=None, help="Optional local LeRobot cache/root directory.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--front-key", default="observation.images.front")
    parser.add_argument("--wrist-key", default="observation.images.wrist")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--view-layout", default="hstack", choices=("hstack", "front", "wrist"))
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", "--fps", dest="target_fps", type=float, default=5.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--default-task", default="pick up the bottle")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("This converter requires LeRobotDataset v3 support.") from exc

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"OUTPUT_DIR is not empty: {args.output_dir}. Use --overwrite or choose a new path.")

    dataset_kwargs = {"repo_id": args.repo_id, "video_backend": args.video_backend}
    if args.root is not None:
        dataset_kwargs["root"] = args.root
    dataset = LeRobotDataset(**dataset_kwargs)

    num_written = 0
    for buffer in iter_episode_buffers(
        dataset,
        front_key=args.front_key,
        wrist_key=args.wrist_key,
        max_episodes=args.max_episodes,
        default_task=args.default_task,
        height=args.height,
        width=args.width,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        view_layout=args.view_layout,
    ):
        write_episode(buffer, args.output_dir, fps=args.target_fps, overwrite=args.overwrite)
        num_written += 1

    print(
        f"Wrote {num_written} videos to {args.output_dir} using "
        f"view_layout={args.view_layout} "
        f"after downsampling {args.source_fps:g}fps -> {args.target_fps:g}fps"
    )


if __name__ == "__main__":
    main()
