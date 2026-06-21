"""Export SO-101 normalizer statistics for dataset-free policy serving."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import hydra
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from cosmos_predict2.configs.config import make_config  # noqa: E402
from imaginaire.lazy_config import instantiate  # noqa: E402
from imaginaire.utils.config_helper import override  # noqa: E402

DEFAULT_DELTA_30HZ_EXPERIMENT = (
    "w2a_so101_delta_30hz_partial_bridge_init_"
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
    "lr1.000e-04_layer20_bsz4"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULT_DELTA_30HZ_EXPERIMENT)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--num-val-episodes", type=int, default=10)
    parser.add_argument(
        "--video-dir",
        type=pathlib.Path,
        default=None,
        help="Optional external hstack video dir when exporting from an action-only zarr.",
    )
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Compute statistics without writing the dataset .statistics_cache under data-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["SO101_DATA_DIR"] = str(args.data_dir)
    os.environ["SO101_NUM_VAL_EPISODES"] = str(args.num_val_episodes)
    if args.video_dir is not None:
        os.environ["SO101_VIDEO_DIR"] = str(args.video_dir)
        os.environ["SO101_VIDEO_FPS"] = str(args.video_fps)

    config = make_config()
    config = override(config, ["--", f"experiment={args.experiment_name}"])
    data_config = instantiate(config.data_config)
    dataset = hydra.utils.instantiate(data_config.dataset.dataset, train=True, verbose=False)
    stats = dataset._compute_statistics() if args.skip_cache else dataset.get_statistics()

    payload = {
        "metadata": {
            "experiment_name": args.experiment_name,
            "data_dir": str(args.data_dir),
            "video_dir": None if args.video_dir is None else str(args.video_dir),
            "video_fps": args.video_fps,
            "num_val_episodes": args.num_val_episodes,
            "dataset_length": len(dataset),
            "note": "Normalizer statistics for SO-101 dataset-free policy serving.",
        },
        "stats": _jsonable(stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote stats to {args.output}")


if __name__ == "__main__":
    main()
