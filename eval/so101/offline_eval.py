"""Offline trajectory evaluation for a mimic-video SO-101 checkpoint.

This Phase-1 script evaluates the model on converted SO-101 zarr chunks before
any real-robot rollout. It compares predicted 6D joint-action chunks against the
future action chunks returned by the same mimic-video dataloader used for
training.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pathlib
import sys
from typing import Literal

import hydra
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mimic_video_so101_policy import MimicVideoSO101Policy, SO101MimicVideoPolicyConfig  # noqa: E402


def _default_mimic_video_root() -> pathlib.Path:
    scratch = pathlib.Path(os.environ.get("SCRATCH", f"/cluster/scratch/{os.environ.get('USER', 'unknown')}"))
    return pathlib.Path(os.environ.get("MIMIC_VIDEO_ROOT", os.environ.get("MVS_ROOT", scratch / "mimic_video")))


def _env_path(*names: str, default: pathlib.Path) -> pathlib.Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return pathlib.Path(value)
    return default


_DEFAULT_ROOT = _default_mimic_video_root()
_DEFAULT_EXPERIMENT_ROOT = pathlib.Path(
    os.environ.get(
        "MIMIC_VIDEO_EXPERIMENT_ROOT",
        os.environ.get("MVS_EXP_ROOT", _DEFAULT_ROOT / "experiments" / os.environ.get("MVS_EXPERIMENT", "so101-homogeneous-rel")),
    )
)

DEFAULT_EXPERIMENT = os.environ.get(
    "POLICY_EXPERIMENT_NAME",
    "w2a_so101_relative_partial_bridge_init_"
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
    "lr1.000e-04_layer20_bsz4",
)
DEFAULT_VIDEO_CKPT = _env_path(
    "MVS_VIDEO_BACKBONE_PATH",
    "MIMIC_VIDEO_BACKBONE_PATH",
    default=_DEFAULT_ROOT
    / "shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt",
)
DEFAULT_ACTION_CKPT = _env_path(
    "MVS_ACTION_MODEL_PATH",
    "MIMIC_VIDEO_ACTION_MODEL_PATH",
    default=_DEFAULT_EXPERIMENT_ROOT / "checkpoints/action_decoder_relative/checkpoints/model/iter_000002500.pt",
)
DEFAULT_DATA_DIR = _env_path(
    "MVS_SHARED_ACTION_DATA_DIR",
    "SO101_DATA_DIR",
    default=_DEFAULT_ROOT / "shared/data/so101_bottle_action_only_full",
)
DEFAULT_OUTPUT_DIR = pathlib.Path("/tmp/mimic_video_so101_offline_eval")

EvalSplit = Literal["train", "val"]
EvalMode = Literal["generated", "oracle", "both"]


@dataclasses.dataclass(frozen=True)
class OfflineEvalArgs:
    experiment_name: str = DEFAULT_EXPERIMENT
    video_model_path: pathlib.Path = DEFAULT_VIDEO_CKPT
    video_lora_path: pathlib.Path | None = None
    action_model_path: pathlib.Path = DEFAULT_ACTION_CKPT
    data_dir: pathlib.Path = DEFAULT_DATA_DIR
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR
    split: EvalSplit = "val"
    mode: EvalMode = "both"
    num_samples: int = 8
    seed: int = 0
    num_val_episodes: int = 10
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 20
    use_cuda_graphs: bool = False
    dump_videos: bool = False


def parse_args() -> OfflineEvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--video-model-path", type=pathlib.Path, default=DEFAULT_VIDEO_CKPT)
    parser.add_argument("--video-lora-path", type=pathlib.Path, default=None)
    parser.add_argument("--action-model-path", type=pathlib.Path, default=DEFAULT_ACTION_CKPT)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--mode", choices=("generated", "oracle", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-val-episodes", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--stop-video-denoising-step", type=int, default=20)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--dump-videos", action="store_true", help="Save input, GT future, and generated videos for visual inspection.")
    return OfflineEvalArgs(**vars(parser.parse_args()))


class OfflineSO101Evaluator:
    def __init__(self, args: OfflineEvalArgs) -> None:
        if args.num_samples <= 0:
            raise ValueError("--num-samples must be positive")
        if args.stop_video_denoising_step < 0 or args.stop_video_denoising_step >= args.num_sampling_steps:
            raise ValueError("--stop-video-denoising-step must be in [0, --num-sampling-steps)")
        self.args = args
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = MimicVideoSO101Policy(
            SO101MimicVideoPolicyConfig(
                experiment_name=args.experiment_name,
                video_model_path=args.video_model_path,
                video_lora_path=args.video_lora_path,
                action_model_path=args.action_model_path,
                data_dir=args.data_dir,
                num_val_episodes=args.num_val_episodes,
                num_sampling_steps=args.num_sampling_steps,
                stop_video_denoising_step=args.stop_video_denoising_step,
                use_cuda_graphs=args.use_cuda_graphs,
            )
        )
        self.dataset = hydra.utils.instantiate(
            self.policy.data_config.dataset.dataset,
            train=args.split == "train",
            verbose=False,
        )

    def run(self) -> None:
        sample_indices = self._sample_indices()
        modes = ("generated", "oracle") if self.args.mode == "both" else (self.args.mode,)
        print(
            f"Running SO-101 offline eval: split={self.args.split}, "
            f"modes={','.join(modes)}, samples={len(sample_indices)}, "
            f"output_dir={self.args.output_dir}",
            flush=True,
        )
        all_results = {mode: self._evaluate_mode(mode, sample_indices) for mode in modes}
        self.policy.write_metadata(self.args.output_dir / "policy_metadata.json")
        self._write_summary(all_results, sample_indices)
        self._write_plots(all_results)
        print(f"Wrote SO-101 offline eval results to {self.args.output_dir}", flush=True)
        for mode, result in all_results.items():
            aggregate = result["aggregate"]
            print(
                f"{mode}: mae={aggregate['mae']:.6f}, "
                f"mse={aggregate['mse']:.6f}, "
                f"rmse={aggregate['rmse']:.6f}, "
                f"max_abs_error={aggregate['max_abs_error']:.6f}, "
                f"mean_elapsed_ms={aggregate['mean_elapsed_ms']:.1f}",
                flush=True,
            )

    def _sample_indices(self) -> np.ndarray:
        if len(self.dataset) == 0:
            raise RuntimeError(f"Dataset split {self.args.split!r} has no chunks")
        rng = np.random.default_rng(self.args.seed)
        replace = len(self.dataset) < self.args.num_samples
        return rng.choice(len(self.dataset), size=self.args.num_samples, replace=replace)

    def _evaluate_mode(self, mode: str, sample_indices: np.ndarray) -> dict[str, object]:
        rows = []
        predictions, targets = [], []
        for order, dataset_index in enumerate(sample_indices):
            print(f"[{mode}] sample {order + 1}/{len(sample_indices)} dataset_index={int(dataset_index)}", flush=True)
            sample = self.dataset[int(dataset_index)]
            sample_seed = self.args.seed + order
            result = self.policy.predict_from_dataset_sample(sample, mode=mode, seed=sample_seed)
            if self.args.dump_videos:
                self._dump_sample_videos(mode, order, int(dataset_index), sample, sample_seed)
            pred = np.asarray(result["actions"], dtype=np.float32)
            target = np.asarray(sample["action/lowdim_concat"], dtype=np.float32)
            if pred.shape != target.shape:
                raise RuntimeError(f"Prediction/target shape mismatch for index {dataset_index}: {pred.shape} vs {target.shape}")
            error = pred - target
            rows.append(
                {
                    "order": order,
                    "dataset_index": int(dataset_index),
                    "mode": mode,
                    "mae": float(np.mean(np.abs(error))),
                    "mse": float(np.mean(error ** 2)),
                    "max_abs_error": float(np.max(np.abs(error))),
                    "elapsed_ms": float(result["elapsed_ms"]),
                    "pred_min": float(np.min(pred)),
                    "pred_max": float(np.max(pred)),
                    "target_min": float(np.min(target)),
                    "target_max": float(np.max(target)),
                }
            )
            predictions.append(pred)
            targets.append(target)

        pred_arr = np.stack(predictions, axis=0)
        target_arr = np.stack(targets, axis=0)
        error_arr = pred_arr - target_arr
        self._write_mode_arrays(mode, pred_arr, target_arr, error_arr)
        self._write_mode_rows(mode, rows)
        return {
            "rows": rows,
            "aggregate": {
                "mode": mode,
                "num_samples": len(rows),
                "mae": float(np.mean(np.abs(error_arr))),
                "mse": float(np.mean(error_arr ** 2)),
                "rmse": float(np.sqrt(np.mean(error_arr ** 2))),
                "max_abs_error": float(np.max(np.abs(error_arr))),
                "per_joint_mae": np.mean(np.abs(error_arr), axis=(0, 1)).astype(float).tolist(),
                "per_joint_mse": np.mean(error_arr ** 2, axis=(0, 1)).astype(float).tolist(),
                "mean_elapsed_ms": float(np.mean([row["elapsed_ms"] for row in rows])),
            },
        }

    def _dump_sample_videos(
        self,
        mode: str,
        order: int,
        dataset_index: int,
        sample: dict[str, np.ndarray],
        seed: int,
    ) -> None:
        video_dir = self.args.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        prefix = video_dir / f"{mode}_sample{order:03d}_idx{dataset_index}"

        input_frames = self._video_array_to_uint8_frames(sample["obs/workspace_rgb"])
        gt_future_frames = self._video_array_to_uint8_frames(sample["action/workspace_rgb"])
        self._write_mp4(prefix.with_name(prefix.name + "_input_context.mp4"), input_frames, fps=10)
        self._write_mp4(prefix.with_name(prefix.name + "_gt_future.mp4"), gt_future_frames, fps=10)

        metadata = {
            "mode": mode,
            "order": order,
            "dataset_index": dataset_index,
            "seed": seed,
            "input_context_frames": int(input_frames.shape[0]),
            "gt_future_frames": int(gt_future_frames.shape[0]),
        }
        if mode == "generated":
            print(f"[{mode}] dumping decoded generated video for sample {order + 1}", flush=True)
            generated_video = self.policy.generate_video_from_dataset_sample(sample, seed=seed)
            generated_frames = self._video_array_to_uint8_frames(generated_video)
            generated_future_frames = self._select_future_frames(
                generated_frames,
                observed_frames=input_frames.shape[0],
                target_future_frames=gt_future_frames.shape[0],
            )
            self._write_mp4(prefix.with_name(prefix.name + "_generated_full.mp4"), generated_frames, fps=10)
            self._write_mp4(prefix.with_name(prefix.name + "_generated_future.mp4"), generated_future_frames, fps=10)
            metadata["generated_full_frames"] = int(generated_frames.shape[0])
            metadata["generated_future_frames"] = int(generated_future_frames.shape[0])

        prefix.with_name(prefix.name + "_video_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    def _video_array_to_uint8_frames(self, video: np.ndarray) -> np.ndarray:
        array = np.asarray(video)
        if array.ndim == 5:
            if array.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for video dump, got shape {array.shape}")
            array = array[0]
        if array.ndim != 4:
            raise ValueError(f"Expected video shape C,T,H,W or T,H,W,C, got {array.shape}")

        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 3, 0))
        elif array.shape[-1] not in (1, 3):
            raise ValueError(f"Could not infer channel axis for video shape {array.shape}")

        array = array.astype(np.float32)
        if float(np.nanmin(array)) < -0.05:
            array = (array + 1.0) / 2.0
        elif float(np.nanmax(array)) > 2.0:
            array = array / 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        array = np.clip(array, 0.0, 1.0)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return (array * 255.0).round().astype(np.uint8)

    def _select_future_frames(
        self,
        generated_frames: np.ndarray,
        *,
        observed_frames: int,
        target_future_frames: int,
    ) -> np.ndarray:
        if generated_frames.shape[0] >= observed_frames + target_future_frames:
            start = observed_frames
            end = observed_frames + target_future_frames
            return generated_frames[start:end]
        if generated_frames.shape[0] >= target_future_frames:
            return generated_frames[-target_future_frames:]
        return generated_frames

    def _write_mp4(self, path: pathlib.Path, frames: np.ndarray, *, fps: int) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("--dump-videos requires imageio with ffmpeg support in this environment.") from exc
        imageio.mimsave(path, list(frames), fps=fps, quality=8)

    def _write_mode_arrays(self, mode: str, predictions: np.ndarray, targets: np.ndarray, errors: np.ndarray) -> None:
        np.savez_compressed(
            self.args.output_dir / f"{mode}_trajectories.npz",
            predictions=predictions,
            targets=targets,
            errors=errors,
        )

    def _write_mode_rows(self, mode: str, rows: list[dict[str, object]]) -> None:
        with (self.args.output_dir / f"{mode}_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary(self, results: dict[str, dict[str, object]], sample_indices: np.ndarray) -> None:
        summary = {
            "args": dataclasses.asdict(self.args) | {
                "video_model_path": str(self.args.video_model_path),
                "action_model_path": str(self.args.action_model_path),
                "data_dir": str(self.args.data_dir),
                "output_dir": str(self.args.output_dir),
            },
            "dataset_length": len(self.dataset),
            "sample_indices": sample_indices.astype(int).tolist(),
            "policy_metadata": self.policy.metadata(),
            "aggregates": {mode: result["aggregate"] for mode, result in results.items()},
        }
        (self.args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _write_plots(self, results: dict[str, dict[str, object]]) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for mode in results:
            data = np.load(self.args.output_dir / f"{mode}_trajectories.npz")
            pred, target = data["predictions"], data["targets"]
            sample_id = 0
            fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
            timesteps = np.arange(pred.shape[1])
            for joint_idx, ax in enumerate(axes.reshape(-1)):
                ax.plot(timesteps, target[sample_id, :, joint_idx], label="gt", linewidth=2)
                ax.plot(timesteps, pred[sample_id, :, joint_idx], label="pred", linewidth=2, linestyle="--")
                ax.set_title(joint_names[joint_idx])
                ax.grid(True, alpha=0.3)
            axes[0, 0].legend(loc="best")
            fig.suptitle(f"SO-101 mimic-video offline trajectory check ({mode}, sample 0)")
            fig.tight_layout()
            fig.savefig(self.args.output_dir / f"{mode}_sample0_trajectory.png", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    OfflineSO101Evaluator(parse_args()).run()
