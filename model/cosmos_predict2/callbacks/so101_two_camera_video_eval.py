"""Lightweight W&B video previews for SO-101 two-camera V2W training."""

from __future__ import annotations

import os
import pathlib
import pickle
import tempfile
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


@dataclass(frozen=True)
class PreviewSpec:
    video_path: pathlib.Path
    meta_path: pathlib.Path
    t5_embedding_path: pathlib.Path


class SO101TwoCameraVideoEvalCallback(Callback):
    def __init__(
        self,
        dataset_dir: str,
        iteration_list: str = "",
        every_n: int = 50,
        start_after: int = 300,
        num_samples: int = 2,
        num_sampling_steps: int = 20,
        num_conditional_frames: int = 5,
        seed: int = 0,
        fps: int = 5,
        enabled_env: str = "SO101_V2W_WANDB_VIDEO_EVAL",
    ) -> None:
        self.dataset_dir = pathlib.Path(dataset_dir) if dataset_dir else pathlib.Path(".")
        self.iterations = self._parse_iteration_list(str(iteration_list))
        self.every_n = int(every_n)
        self.start_after = int(start_after)
        self.num_samples = int(num_samples)
        self.num_sampling_steps = int(num_sampling_steps)
        self.num_conditional_frames = int(num_conditional_frames)
        self.seed = int(seed)
        self.fps = int(fps)
        self.enabled_env = enabled_env
        self._preview_specs: list[PreviewSpec] | None = None

    @staticmethod
    def _parse_iteration_list(spec: str) -> set[int]:
        result = set()
        for item in spec.split(","):
            item = item.strip()
            if not item or item.lower() in {"none", "null"}:
                continue
            result.add(int(item))
        return result

    def _enabled(self) -> bool:
        return os.environ.get(self.enabled_env, "1").lower() not in {"0", "false", "no", "off"}

    def _should_log(self, iteration: int) -> bool:
        if iteration in self.iterations:
            return True
        if iteration <= self.start_after:
            return False
        return self.every_n > 0 and iteration % self.every_n == 0

    def _load_preview_specs(self) -> list[PreviewSpec]:
        if self._preview_specs is not None:
            return self._preview_specs
        video_dir = self.dataset_dir / "video"
        meta_dir = self.dataset_dir / "metas"
        t5_dir = self.dataset_dir / "t5_xxl"
        specs = []
        for video_path in sorted(video_dir.glob("*.mp4"))[: self.num_samples]:
            meta_path = meta_dir / f"{video_path.stem}.txt"
            t5_embedding_path = t5_dir / f"{video_path.stem}.pickle"
            if meta_path.exists() and t5_embedding_path.exists():
                specs.append(
                    PreviewSpec(
                        video_path=video_path,
                        meta_path=meta_path,
                        t5_embedding_path=t5_embedding_path,
                    )
                )
        self._preview_specs = specs
        if not specs:
            log.warning(f"SO101TwoCameraVideoEvalCallback found no preview videos in {video_dir}")
        return specs

    @distributed.rank0_only
    @torch.no_grad()
    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss
        if not self._enabled() or not self._should_log(iteration):
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        specs = self._load_preview_specs()
        if not specs:
            return

        was_training = model.training
        model.eval()
        try:
            log_payload = {"trainer/global_step": iteration}
            for sample_idx, spec in enumerate(specs):
                preview_path = self._render_preview(model, spec, iteration=iteration, sample_idx=sample_idx)
                log_payload[f"so101_v2w_preview/sample_{sample_idx:03d}"] = wandb.Video(
                    str(preview_path),
                    fps=self.fps,
                    format="mp4",
                    caption=f"iter={iteration} {spec.video_path.name}",
                )
            wandb.log(log_payload, step=iteration)
        except Exception as exc:
            log.warning(f"SO101 two-camera video preview failed at iter {iteration}: {exc}")
        finally:
            if was_training:
                model.train()
            torch.cuda.empty_cache()

    def _render_preview(self, model, spec: PreviewSpec, *, iteration: int, sample_idx: int) -> pathlib.Path:
        frames = self._read_video_uint8(spec.video_path)
        prompt = spec.meta_path.read_text(encoding="utf-8").strip()
        condition_frames = frames[: self.num_conditional_frames]
        if len(condition_frames) < self.num_conditional_frames:
            raise ValueError(f"Preview video too short: {spec.video_path}")

        generated = self._generate(model, spec, condition_frames, prompt=prompt, seed=self.seed + sample_idx)
        preview = self._side_by_side(frames, generated)
        out_dir = pathlib.Path(tempfile.gettempdir()) / "mvs_so101_v2w_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"iter_{iteration:09d}_sample_{sample_idx:03d}.mp4"
        imageio.mimsave(out_path, list(preview), fps=self.fps, quality=8)
        return out_path

    def _generate(
        self,
        model,
        spec: PreviewSpec,
        condition_frames: np.ndarray,
        *,
        prompt: str,
        seed: int,
    ) -> np.ndarray:
        pipe = model.pipe
        video = torch.from_numpy(condition_frames).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        video = video.to(dtype=torch.uint8)
        prompt_embedding = self._load_prompt_embedding(spec.t5_embedding_path).cuda()
        generated = pipe.generate_video(
            vid_input=video,
            num_latent_conditional_frames=pipe.tokenizer.get_latent_num_frames(self.num_conditional_frames),
            prompt=prompt,
            prompt_embedding=prompt_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=self.num_sampling_steps,
            seed=seed,
            use_cuda_graphs=False,
            fps=float(self.fps),
        )
        return self._tensor_video_to_uint8(generated[0])

    def _load_prompt_embedding(self, path: pathlib.Path) -> torch.Tensor:
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

    def _read_video_uint8(self, path: pathlib.Path) -> np.ndarray:
        reader = imageio.get_reader(path)
        try:
            frames = [frame[..., :3] for frame in reader]
        finally:
            reader.close()
        return np.asarray(frames, dtype=np.uint8)

    def _tensor_video_to_uint8(self, video: torch.Tensor) -> np.ndarray:
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

    def _side_by_side(self, gt: np.ndarray, generated: np.ndarray) -> np.ndarray:
        target_len = min(len(gt), len(generated))
        gt = gt[:target_len]
        generated = generated[:target_len]
        if gt.shape[1:3] != generated.shape[1:3]:
            generated_t = torch.from_numpy(generated).permute(0, 3, 1, 2).float()
            generated_t = F.interpolate(generated_t, size=gt.shape[1:3], mode="bilinear", align_corners=False)
            generated = generated_t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        separator = np.full((target_len, gt.shape[1], 4, 3), 255, dtype=np.uint8)
        return np.concatenate([gt, separator, generated], axis=2)
