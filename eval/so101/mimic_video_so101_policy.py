"""mimic-video SO-101 policy wrapper for offline and server-side evaluation.

The wrapper intentionally follows the SO-101 observation/action convention used by
DreamDifferent/OpenPI clients while keeping mimic-video's current front-camera-only
training setup:

    observation/images/front -> workspace_rgb
    observation/state        -> obs/lowdim_concat, shape (1, 6)
    actions                  -> action/lowdim_concat, shape (15, 6)

For Phase 1 offline evaluation, prompt embeddings are read from the converted zarr
dataset and passed directly to the video pipeline, so the heavyweight T5 text
encoder does not need to be loaded.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import pathlib
import sys
import time
from typing import Literal

import hydra
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from cosmos_predict2.configs.config import make_config  # noqa: E402
from cosmos_predict2.data.action.utils import extract_normalization_types  # noqa: E402
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # noqa: E402
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline  # noqa: E402
from cosmos_predict2.pipelines.world2action import World2ActionPipeline  # noqa: E402
from imaginaire.lazy_config import instantiate  # noqa: E402
from imaginaire.utils.config_helper import override  # noqa: E402

PredictionMode = Literal["generated", "oracle"]


@dataclasses.dataclass(frozen=True)
class SO101MimicVideoPolicyConfig:
    experiment_name: str
    video_model_path: pathlib.Path
    action_model_path: pathlib.Path
    data_dir: pathlib.Path | None = None
    stats_path: pathlib.Path | None = None
    prompt_embedding_manifest_path: pathlib.Path | None = None
    video_lora_path: pathlib.Path | None = None
    num_val_episodes: int = 10
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 20
    load_text_encoder: bool = False
    use_cuda_graphs: bool = False
    video_lora_rank: int = 256
    video_lora_alpha: int = 32
    video_lora_target_modules: str = (
        "q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2"
    )


class MimicVideoSO101Policy:
    """Loads mimic-video VAM components and predicts SO-101 6D action chunks."""

    def __init__(self, cfg: SO101MimicVideoPolicyConfig) -> None:
        if cfg.device != "cuda":
            raise ValueError("Current mimic-video pipelines assume CUDA internally; use device='cuda'.")
        if not cfg.video_model_path.exists():
            raise FileNotFoundError(f"Missing video model checkpoint: {cfg.video_model_path}")
        if cfg.video_lora_path is not None and not cfg.video_lora_path.exists():
            raise FileNotFoundError(f"Missing video LoRA checkpoint: {cfg.video_lora_path}")
        if not cfg.action_model_path.exists():
            raise FileNotFoundError(f"Missing action model checkpoint: {cfg.action_model_path}")
        if cfg.stats_path is not None and not cfg.stats_path.exists():
            raise FileNotFoundError(f"Missing SO-101 stats file: {cfg.stats_path}")
        if cfg.prompt_embedding_manifest_path is not None and not cfg.prompt_embedding_manifest_path.exists():
            raise FileNotFoundError(f"Missing SO-101 prompt embedding manifest: {cfg.prompt_embedding_manifest_path}")
        if cfg.stats_path is None and (cfg.data_dir is None or not cfg.data_dir.exists()):
            raise FileNotFoundError(f"Missing SO-101 data directory: {cfg.data_dir}")

        self.cfg = cfg
        self.prompt_embedding_manifest = self._load_prompt_embedding_manifest(cfg.prompt_embedding_manifest_path)
        self.prompt_embedding_cache: dict[str, torch.Tensor] = {}
        self.config, self.data_config = self._load_resolved_config()
        resize_sizes = list(self.data_config.policy_io.img_resize_sizes)
        self.image_height = int(resize_sizes[0])
        self.image_width = int(resize_sizes[1])
        self.obs_image_horizon = int(self.data_config.policy_io.policy_io.obs.workspace_rgb.horizon)
        self.front_image_history: collections.deque[np.ndarray] = collections.deque(maxlen=self.obs_image_horizon)
        self.pipeline = self._load_pipeline()
        self.action_horizon = int(self.data_config.policy_io.policy_io.action.joint_action_lowdim.horizon)
        self.action_dim = int(self.pipeline.world2action_pipeline.dit.out_channels)

    def _load_resolved_config(self):
        data_dir = self.cfg.data_dir or pathlib.Path("/tmp/mimic_video_so101_stats_only")
        os.environ["SO101_DATA_DIR"] = str(data_dir)
        os.environ["SO101_NUM_VAL_EPISODES"] = str(self.cfg.num_val_episodes)

        config = make_config()
        config = override(config, ["--", f"experiment={self.cfg.experiment_name}"])
        config.model.config.video_pipe_config.guardrail_config.enabled = False
        data_config = instantiate(config.data_config)
        return config, data_config

    def _load_pipeline(self) -> Video2World2ActionPipeline:
        video_pipe = Video2WorldPipeline.from_config(
            config=self.config.model.config.video_pipe_config,
            dit_path=str(self.cfg.video_model_path),
            use_text_encoder=self.cfg.load_text_encoder,
            device=self.cfg.device,
            torch_dtype=self.cfg.dtype,
            load_ema_to_reg=False,
        )
        if self.cfg.video_lora_path is not None:
            self._load_video_lora_adapter(video_pipe)
        action_pipe = World2ActionPipeline.from_config(
            config=self.config.model.config.pipe_config,
            dit_path=str(self.cfg.action_model_path),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

        stats = self._load_or_compute_statistics()
        action_pipe.normalizer.build_from_stats(
            stats,
            normalization_types=extract_normalization_types(self.data_config.policy_io.policy_io),
            concat_groups=self.data_config.policy_io.concat_groups,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        action_pipe.normalizer.requires_grad_(False)
        return Video2World2ActionPipeline(video_pipe, action_pipe).cuda().eval()

    def _load_or_compute_statistics(self) -> dict:
        if self.cfg.stats_path is not None:
            return self._load_statistics_file(self.cfg.stats_path)
        dataset = hydra.utils.instantiate(self.data_config.dataset.dataset, train=True, verbose=False)
        stats = dataset.get_statistics()
        return {key: {k: np.asarray(v, dtype=np.float32) for k, v in value.items()} for key, value in stats.items()}

    @staticmethod
    def _load_statistics_file(path: pathlib.Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload.get("stats", payload)
        return {
            key: {stat_key: np.asarray(stat_value, dtype=np.float32) for stat_key, stat_value in value.items()}
            for key, value in stats.items()
        }

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return " ".join(prompt.strip().split())

    @classmethod
    def _load_prompt_embedding_manifest(cls, path: pathlib.Path | None) -> dict[str, pathlib.Path]:
        if path is None:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("prompts", payload)
        if not isinstance(entries, dict):
            raise ValueError(f"Prompt embedding manifest must be a JSON object: {path}")

        manifest_dir = path.parent
        prompt_to_path = {}
        for prompt, embedding_path in entries.items():
            if not isinstance(prompt, str) or not isinstance(embedding_path, str):
                raise ValueError(f"Invalid prompt embedding manifest entry in {path}: {prompt!r} -> {embedding_path!r}")
            resolved = pathlib.Path(embedding_path)
            if not resolved.is_absolute():
                resolved = manifest_dir / resolved
            prompt_to_path[cls._normalize_prompt(prompt)] = resolved
        return prompt_to_path

    def _load_video_lora_adapter(self, video_pipe: Video2WorldPipeline) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        from cosmos_predict2.utils.checkpointer import load_matching_state_dict_tensors

        lora_config = LoraConfig(
            r=self.cfg.video_lora_rank,
            lora_alpha=self.cfg.video_lora_alpha,
            init_lora_weights=False,
            target_modules=self.cfg.video_lora_target_modules.split(","),
        )
        video_pipe.dit = inject_adapter_in_model(lora_config, video_pipe.dit)

        state_dict = torch.load(self.cfg.video_lora_path, map_location="cpu")
        adapter_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("net."):
                key = key[len("net.") :]
            if ".lora_" in key:
                adapter_state_dict[key] = value
        incompatible = load_matching_state_dict_tensors(video_pipe.dit, adapter_state_dict)
        missing_lora = [key for key in incompatible.missing_keys if ".lora_" in key]
        unexpected_lora = [key for key in incompatible.unexpected_keys if ".lora_" in key]
        if missing_lora or unexpected_lora:
            raise RuntimeError(
                "Failed to load video LoRA adapter cleanly: "
                f"missing_lora={missing_lora[:10]}, unexpected_lora={unexpected_lora[:10]}"
            )

    @torch.no_grad()
    def predict_from_dataset_sample(
        self,
        sample: dict[str, np.ndarray],
        *,
        mode: PredictionMode,
        seed: int,
    ) -> dict[str, np.ndarray | float]:
        started_at = time.perf_counter()
        input_vid = self._batch_tensor(sample["obs/workspace_rgb"])
        state = self._batch_tensor(sample["obs/lowdim_concat"])
        prompt_embedding = self._batch_tensor(sample["obs/language_embedding"])

        if mode == "generated":
            actions = self.pipeline(
                input_vid=input_vid,
                state_B_HO_O=state,
                prompt="",
                prompt_embedding=prompt_embedding,
                num_sampling_step=self.cfg.num_sampling_steps,
                stop_after_step=self.cfg.stop_video_denoising_step,
                seed=seed,
                use_cuda_graphs=self.cfg.use_cuda_graphs,
                fps=5.0,
            )
        elif mode == "oracle":
            actions = self._predict_with_oracle_future_video(
                input_vid=input_vid,
                future_vid=self._batch_tensor(sample["action/workspace_rgb"]),
                state=state,
                prompt_embedding=prompt_embedding,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported prediction mode: {mode}")

        actions_np = actions[0].float().cpu().numpy()
        if actions_np.shape != (self.action_horizon, self.action_dim):
            raise RuntimeError(
                f"Expected action shape {(self.action_horizon, self.action_dim)}, got {actions_np.shape}"
            )
        if not np.isfinite(actions_np).all():
            raise RuntimeError("Policy produced non-finite action values.")
        return {"actions": actions_np, "elapsed_ms": (time.perf_counter() - started_at) * 1000.0}

    @torch.no_grad()
    def predict_from_observation(self, observation: dict[str, object], *, seed: int) -> dict[str, np.ndarray | float]:
        """Predict a SO-101 action chunk from an OpenPI-style observation dict.

        Expected keys are ``observation/images/front`` and ``observation/state``.
        A request may either include ``observation/prompt_embedding`` with shape
        ``(512, 1024)`` or use ``prompt`` when the policy was constructed with
        ``load_text_encoder=True``.
        """
        started_at = time.perf_counter()
        input_vid = self._online_front_video_tensor(
            observation["observation/images/front"],
            reset_history=bool(observation.get("reset_history", False)),
        )
        state = self._online_state_tensor(observation["observation/state"])
        prompt_embedding = self._online_prompt_embedding(observation)
        prompt = str(observation.get("prompt", ""))

        if prompt_embedding is None and not self.cfg.load_text_encoder:
            raise ValueError(
                "Request did not include observation/prompt_embedding, the prompt was not found in the "
                "precomputed prompt embedding manifest, and the policy server was started without "
                "--load-text-encoder. Send a precomputed prompt embedding, add the prompt to the manifest, "
                "or start the server with --load-text-encoder."
            )

        actions = self.pipeline(
            input_vid=input_vid,
            state_B_HO_O=state,
            prompt=prompt,
            prompt_embedding=prompt_embedding,
            num_sampling_step=self.cfg.num_sampling_steps,
            stop_after_step=self.cfg.stop_video_denoising_step,
            seed=seed,
            use_cuda_graphs=self.cfg.use_cuda_graphs,
            fps=5.0,
        )
        actions_np = actions[0].float().cpu().numpy()
        if actions_np.shape != (self.action_horizon, self.action_dim):
            raise RuntimeError(
                f"Expected action shape {(self.action_horizon, self.action_dim)}, got {actions_np.shape}"
            )
        if not np.isfinite(actions_np).all():
            raise RuntimeError("Policy produced non-finite action values.")
        return {"actions": actions_np, "elapsed_ms": (time.perf_counter() - started_at) * 1000.0}

    def reset_history(self) -> None:
        self.front_image_history.clear()

    def _online_front_video_tensor(self, image: object, *, reset_history: bool) -> torch.Tensor:
        if reset_history:
            self.reset_history()
        frame = self._preprocess_front_image(image)
        self.front_image_history.append(frame)
        frames = list(self.front_image_history)
        if len(frames) < self.obs_image_horizon:
            frames = [frames[0]] * (self.obs_image_horizon - len(frames)) + frames
        video = np.array(np.stack(frames, axis=1), copy=True)  # C,T,H,W in [-1, 1], matching CosmosProcessImage.
        return torch.as_tensor(video, device=self.cfg.device, dtype=self.cfg.dtype).unsqueeze(0)

    def _preprocess_front_image(self, image: object) -> np.ndarray:
        from PIL import Image

        array = np.array(image, copy=True)
        if array.ndim != 3:
            raise ValueError(f"Expected front image with 3 dimensions, got shape {array.shape}")
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"Expected RGB front image with 3 channels, got shape {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            if float(np.nanmin(array)) < -0.05:
                array = (array + 1.0) / 2.0
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.asarray(array, dtype=np.uint8)
        resized = Image.fromarray(array, "RGB").resize((self.image_width, self.image_height), resample=Image.BILINEAR)
        chw = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
        return 2.0 * (chw / 255.0 - 0.5)

    def _online_state_tensor(self, state: object) -> torch.Tensor:
        state_array = np.array(state, dtype=np.float32, copy=True)
        if state_array.shape == (self.action_dim,):
            state_array = state_array[None, :]
        if state_array.shape != (1, self.action_dim):
            raise ValueError(f"Expected observation/state shape (6,) or (1, 6), got {state_array.shape}")
        return torch.as_tensor(state_array, device=self.cfg.device, dtype=self.cfg.dtype).unsqueeze(0)

    def _online_prompt_embedding(self, observation: dict[str, object]) -> torch.Tensor | None:
        embedding = observation.get("observation/prompt_embedding", observation.get("prompt_embedding"))
        loaded_from_manifest = False
        prompt = ""
        if embedding is None:
            prompt = self._normalize_prompt(str(observation.get("prompt", "")))
            if not prompt:
                return None
            cached = self.prompt_embedding_cache.get(prompt)
            if cached is not None:
                return cached
            embedding_path = self.prompt_embedding_manifest.get(prompt)
            if embedding_path is None:
                return None
            if not embedding_path.exists():
                raise FileNotFoundError(f"Missing cached prompt embedding for {prompt!r}: {embedding_path}")
            embedding = np.load(embedding_path)
            loaded_from_manifest = True
        embedding_array = np.array(embedding, dtype=np.float32, copy=True)
        if embedding_array.shape == (512, 1024):
            embedding_array = embedding_array[None, ...]
        if embedding_array.shape != (1, 512, 1024):
            raise ValueError(f"Expected prompt embedding shape (512, 1024) or (1, 512, 1024), got {embedding_array.shape}")
        tensor = torch.as_tensor(embedding_array, device=self.cfg.device, dtype=self.cfg.dtype)
        if loaded_from_manifest:
            self.prompt_embedding_cache[prompt] = tensor
        return tensor

    @torch.no_grad()
    def generate_video_from_dataset_sample(self, sample: dict[str, np.ndarray], *, seed: int) -> np.ndarray:
        """Generate a decoded video from the current observation for visual inspection.

        This is intentionally separate from action prediction: the online action path
        stops diffusion at ``stop_video_denoising_step`` and consumes hidden states,
        while this method completes the video diffusion process and decodes frames so
        that we can visually inspect whether the video backbone predicts plausible
        SO-101 futures.
        """
        input_vid = self._batch_tensor(sample["obs/workspace_rgb"])
        prompt_embedding = self._batch_tensor(sample["obs/language_embedding"])
        observed_frames = int(input_vid.shape[2])
        if observed_frames not in {1, 5}:
            raise ValueError(f"Expected 1 or 5 observed frames, got {observed_frames}")

        video = self.pipeline.video2world_pipeline.generate_video(
            vid_input=input_vid,
            num_latent_conditional_frames=1 if observed_frames == 1 else 2,
            prompt="",
            prompt_embedding=prompt_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=self.cfg.num_sampling_steps,
            seed=seed,
            use_cuda_graphs=self.cfg.use_cuda_graphs,
            fps=5.0,
        )
        return video[0].float().cpu().numpy()

    def _batch_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(array), device=self.cfg.device)
        if tensor.ndim == 3 and tensor.shape[-2:] == (512, 1024):
            return tensor.to(dtype=self.cfg.dtype)
        return tensor.unsqueeze(0).to(dtype=self.cfg.dtype)

    def _predict_with_oracle_future_video(
        self,
        *,
        input_vid: torch.Tensor,
        future_vid: torch.Tensor,
        state: torch.Tensor,
        prompt_embedding: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        video_pipe = self.pipeline.video2world_pipeline
        action_pipe = self.pipeline.world2action_pipeline
        batch_size, _channels, observed_frames, _height, _width = input_vid.shape
        if observed_frames not in {1, 5}:
            raise ValueError(f"Expected 1 or 5 observed frames, got {observed_frames}")
        expected_future = 61 - observed_frames
        if future_vid.shape[2] != expected_future:
            raise ValueError(f"Expected {expected_future} oracle future frames, got {future_vid.shape[2]}")

        data_batch = {
            "obs/workspace_rgb": input_vid,
            "action/workspace_rgb": future_vid,
            "obs/language_embedding": prompt_embedding,
            "num_conditional_frames": video_pipe.tokenizer.get_latent_num_frames(observed_frames),
            "is_preprocessed": True,
        }
        _, video, condition = video_pipe.get_mimic_data_and_condition(data_batch)
        noise = torch.randn(video.size(), dtype=self.cfg.dtype, device=self.cfg.device)

        torch.manual_seed(seed)
        video_pipe.scheduler.set_timesteps(self.cfg.num_sampling_steps, device=self.cfg.device)
        sigma = video_pipe.scheduler.sigmas[self.cfg.stop_video_denoising_step].repeat(batch_size).unsqueeze(1)
        world_pred = video_pipe.denoise(
            video + noise * rearrange(sigma, "b t -> b 1 t 1 1"),
            sigma,
            condition,
            use_cuda_graphs=False,
            return_only_hidden_states_up_to=action_pipe.config.xattn_layer_idx,
            return_decoded_video=False,
        )
        crossattn = world_pred.hidden_states[action_pipe.config.xattn_layer_idx]
        crossattn = crossattn.reshape(crossattn.shape[0], -1, crossattn.shape[-1])
        return action_pipe(
            state_B_HO_O=state,
            crossattn_emb=crossattn,
            context_timesteps_B_1=sigma,
            seed=seed,
            use_cuda_graphs=self.cfg.use_cuda_graphs,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "policy": "mimic-video-so101",
            "experiment_name": self.cfg.experiment_name,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "views": ["front"],
            "data_dir": None if self.cfg.data_dir is None else str(self.cfg.data_dir),
            "stats_path": None if self.cfg.stats_path is None else str(self.cfg.stats_path),
            "video_model_path": str(self.cfg.video_model_path),
            "video_lora_path": None if self.cfg.video_lora_path is None else str(self.cfg.video_lora_path),
            "action_model_path": str(self.cfg.action_model_path),
            "num_sampling_steps": self.cfg.num_sampling_steps,
            "stop_video_denoising_step": self.cfg.stop_video_denoising_step,
        }

    def write_metadata(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.metadata(), indent=2) + "\n", encoding="utf-8")
