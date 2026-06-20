"""Offline trajectory evaluation for a mimic-video SO-101 EE checkpoint."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
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

DEFAULT_EXPERIMENT = (
    "w2a_so101_ee_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
    "lr1.000e-04_layer20_bsz1"
)
DEFAULT_VIDEO_CKPT = pathlib.Path(
    "/cluster/scratch/dohkim/mimic_video/shared/checkpoints/video_backbone/"
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
)
DEFAULT_VIDEO_LORA_CKPT = pathlib.Path(
    "/cluster/scratch/dohkim/mimic_video/shared/checkpoints/video_lora/"
    "so101_multi_object_2cam_hstack_5fps/checkpoints/model/iter_000001000.pt"
)
DEFAULT_ACTION_CKPT = pathlib.Path(
    "/cluster/scratch/dohkim/mimic_video/experiments/so101_hetero_ee/runs/action_decoder/vam/so101_ee/"
    "w2a_so101_ee_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
    "lr1.000e-04_layer20_bsz1/checkpoints/model/iter_000002750.pt"
)
DEFAULT_STATS_PATH = pathlib.Path(
    "/cluster/scratch/dohkim/mimic_video/shared/data/so101_hetero_ee_action_30hz_full/"
    ".statistics_cache/38f88b27628b4953d3ba0c99452d845ed566224ad6b6f39d06fe6e685dec7084"
)
DEFAULT_DATA_DIR = pathlib.Path(
    "/cluster/scratch/dohkim/mimic_video/shared/data/so101_hetero_ee_action_30hz_full"
)
DEFAULT_OUTPUT_DIR = pathlib.Path("/tmp/mimic_video_so101_ee_offline_eval")

EvalSplit = Literal["train", "val"]
EvalMode = Literal["generated", "oracle", "both"]


@dataclasses.dataclass(frozen=True)
class OfflineEvalArgs:
    experiment_name: str = DEFAULT_EXPERIMENT
    video_model_path: pathlib.Path = DEFAULT_VIDEO_CKPT
    video_lora_path: pathlib.Path | None = DEFAULT_VIDEO_LORA_CKPT
    action_model_path: pathlib.Path = DEFAULT_ACTION_CKPT
    stats_path: pathlib.Path | None = DEFAULT_STATS_PATH
    data_dir: pathlib.Path = DEFAULT_DATA_DIR
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR
    split: EvalSplit = "val"
    mode: EvalMode = "both"
    num_samples: int = 8
    seed: int = 0
    num_val_episodes: int = 10
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 10
    use_cuda_graphs: bool = False
    dump_videos: bool = False


def parse_args() -> OfflineEvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--video-model-path", type=pathlib.Path, default=DEFAULT_VIDEO_CKPT)
    parser.add_argument("--video-lora-path", type=pathlib.Path, default=DEFAULT_VIDEO_LORA_CKPT)
    parser.add_argument("--action-model-path", type=pathlib.Path, default=DEFAULT_ACTION_CKPT)
    parser.add_argument("--stats-path", type=pathlib.Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--mode", choices=("generated", "oracle", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-val-episodes", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--stop-video-denoising-step", type=int, default=10)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--dump-videos", action="store_true")
    return OfflineEvalArgs(**vars(parser.parse_args()))


def main(args: OfflineEvalArgs) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = MimicVideoSO101Policy(
        SO101MimicVideoPolicyConfig(
            experiment_name=args.experiment_name,
            video_model_path=args.video_model_path,
            video_lora_path=args.video_lora_path,
            action_model_path=args.action_model_path,
            data_dir=None if args.stats_path is not None else args.data_dir,
            stats_path=args.stats_path,
            num_val_episodes=args.num_val_episodes,
            num_sampling_steps=args.num_sampling_steps,
            stop_video_denoising_step=args.stop_video_denoising_step,
            use_cuda_graphs=args.use_cuda_graphs,
        )
    )
    policy.write_metadata(args.output_dir / "policy_metadata.json")

    dataset = hydra.utils.instantiate(
        policy.data_config.dataset.dataset,
        train=(args.split == "train"),
        verbose=False,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)

    modes: list[EvalMode]
    if args.mode == "both":
        modes = ["generated", "oracle"]
    else:
        modes = [args.mode]

    rows: list[dict[str, object]] = []
    for mode in modes:
        for sample_idx in indices:
            sample = dataset[int(sample_idx)]
            sample_np = {key: np.asarray(value) for key, value in sample.items()}
            result = policy.predict_from_dataset_sample(sample_np, mode=mode, seed=args.seed + int(sample_idx))
            pred = np.asarray(result["actions"])
            gt = np.asarray(sample_np["action/lowdim_concat"])
            mse = float(np.mean((pred - gt) ** 2))
            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "mode": mode,
                    "mse": mse,
                    "elapsed_ms": float(result["elapsed_ms"]),
                }
            )

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_idx", "mode", "mse", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_samples": len(indices),
        "modes": modes,
        "mean_mse_by_mode": {
            mode: float(np.mean([row["mse"] for row in rows if row["mode"] == mode])) for mode in modes
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
