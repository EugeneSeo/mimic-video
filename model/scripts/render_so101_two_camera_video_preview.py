"""Render SO-101 two-camera V2W predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import pickle

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.lazy_config import instantiate
from imaginaire.utils import distributed, log
from imaginaire.utils.config_helper import get_config_module, override


def default_mimic_video_root() -> pathlib.Path:
    scratch = pathlib.Path(os.environ.get("SCRATCH", f"/cluster/scratch/{os.environ.get('USER', 'unknown')}"))
    return pathlib.Path(os.environ.get("MIMIC_VIDEO_ROOT", os.environ.get("MVS_ROOT", scratch / "mimic_video")))


def default_experiment_root() -> pathlib.Path:
    root = default_mimic_video_root()
    experiment = os.environ.get("MIMIC_VIDEO_EXPERIMENT", os.environ.get("MVS_EXPERIMENT", "so101-homogeneous-rel"))
    return pathlib.Path(os.environ.get("MIMIC_VIDEO_EXPERIMENT_ROOT", os.environ.get("MVS_EXP_ROOT", root / "experiments" / experiment)))


def parse_args() -> argparse.Namespace:
    root = default_mimic_video_root()
    exp_root = default_experiment_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cosmos_predict2/configs/config.py")
    parser.add_argument("--experiment", default="v2w_so101_two_camera_lora_rank256_lr1.778e-04_bsz4")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get(
            "SO101_VIDEO_DIR",
            str(root / "shared/video_data/so101_bottle_front_wrist_hstack_5fps_full"),
        ),
    )
    parser.add_argument(
        "--model-checkpoint",
        default=os.environ.get(
            "MVS_VIDEO_LORA_PATH",
            str(exp_root / "checkpoints/video_lora/checkpoints/model/iter_000001000.pt"),
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(exp_root / "previews/video_lora_iter1000"),
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-conditional-frames", type=int, default=5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def read_video_uint8(path: pathlib.Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = [frame[..., :3] for frame in reader]
    finally:
        reader.close()
    return np.asarray(frames, dtype=np.uint8)


def load_t5_embedding(path: pathlib.Path) -> torch.Tensor:
    with path.open("rb") as f:
        embedding_raw = pickle.load(f)
    if not isinstance(embedding_raw, list) or len(embedding_raw) != 1:
        raise ValueError(f"Expected one T5 embedding in {path}")
    embedding = embedding_raw[0]
    if not isinstance(embedding, np.ndarray) or embedding.ndim != 2:
        raise ValueError(f"Expected a 2D numpy T5 embedding in {path}")
    n_tokens = embedding.shape[0]
    if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
        embedding = np.concatenate(
            [
                embedding,
                np.zeros(
                    (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
    return torch.from_numpy(embedding).unsqueeze(0)


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    array = video.detach().float().cpu().numpy()
    if array.ndim != 4:
        raise ValueError(f"Expected generated video C,T,H,W, got {array.shape}")
    array = np.transpose(array, (1, 2, 3, 0))
    if float(np.nanmin(array)) < -0.05:
        array = (array + 1.0) / 2.0
    elif float(np.nanmax(array)) > 2.0:
        array = array / 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def side_by_side(gt: np.ndarray, generated: np.ndarray) -> np.ndarray:
    target_len = min(len(gt), len(generated))
    gt = gt[:target_len]
    generated = generated[:target_len]
    if gt.shape[1:3] != generated.shape[1:3]:
        generated_t = torch.from_numpy(generated).permute(0, 3, 1, 2).float()
        generated_t = F.interpolate(generated_t, size=gt.shape[1:3], mode="bilinear", align_corners=False)
        generated = generated_t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    separator = np.full((target_len, gt.shape[1], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([gt, separator, generated], axis=2)


def build_model(args: argparse.Namespace):
    os.environ["SO101_TWO_CAMERA_VIDEO_DIR"] = args.dataset_dir
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={args.experiment}"])
    model = instantiate(config.model)
    model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


@torch.no_grad()
def render_predictions(args: argparse.Namespace) -> None:
    distributed.init()
    if not distributed.is_rank0():
        return

    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    pipe = model.pipe
    video_paths = sorted((dataset_dir / "video").glob("*.mp4"))[: args.num_samples]
    manifest = []

    for sample_idx, video_path in enumerate(video_paths):
        frames = read_video_uint8(video_path)
        condition_frames = frames[: args.num_conditional_frames]
        if len(condition_frames) < args.num_conditional_frames:
            log.warning(f"Skipping short video: {video_path}")
            continue

        t5_embedding = load_t5_embedding(dataset_dir / "t5_xxl" / f"{video_path.stem}.pickle").cuda()
        video = torch.from_numpy(condition_frames).permute(3, 0, 1, 2).unsqueeze(0).cuda().to(dtype=torch.uint8)
        generated = pipe.generate_video(
            vid_input=video,
            num_latent_conditional_frames=pipe.tokenizer.get_latent_num_frames(args.num_conditional_frames),
            prompt_embedding=t5_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=args.num_sampling_steps,
            seed=args.seed + sample_idx,
            use_cuda_graphs=False,
            fps=float(args.fps),
        )
        generated_uint8 = tensor_video_to_uint8(generated[0])
        preview = side_by_side(frames, generated_uint8)

        pred_path = output_dir / f"{video_path.stem}_pred.mp4"
        preview_path = output_dir / f"{video_path.stem}_gt_vs_pred.mp4"
        imageio.mimsave(pred_path, list(generated_uint8), fps=args.fps, quality=8)
        imageio.mimsave(preview_path, list(preview), fps=args.fps, quality=8)
        manifest.append(
            {
                "sample_index": sample_idx,
                "video": str(video_path),
                "prediction": str(pred_path),
                "gt_vs_prediction": str(preview_path),
            }
        )
        log.info(f"Wrote {preview_path}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.success(f"Wrote {len(manifest)} SO-101 V2W previews to {output_dir}")


if __name__ == "__main__":
    render_predictions(parse_args())
