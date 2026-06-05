"""Prediction server for mimic-video SO-101 action chunks.

This server intentionally only predicts action chunks. It never talks to robot
hardware; robot-side dry-run/rollout clients are separate phases.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import socketserver
import sys
import time
import traceback
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mimic_video_so101_policy import MimicVideoSO101Policy, SO101MimicVideoPolicyConfig  # noqa: E402
from websocket_protocol import PAYLOAD_PROTOCOL, TCP_TRANSPORT, packb, recv_framed, send_framed, unpackb  # noqa: E402

_DEFAULT_ROOT = pathlib.Path(os.environ.get("MIMIC_VIDEO_ROOT", f"/cluster/scratch/{os.environ.get('USER', 'unknown')}/mimic_video"))
_DEFAULT_EXPERIMENT = os.environ.get("MIMIC_VIDEO_EXPERIMENT", "so101-bottle-delta30")
_DEFAULT_EXPERIMENT_ROOT = pathlib.Path(
    os.environ.get("MIMIC_VIDEO_EXPERIMENT_ROOT", _DEFAULT_ROOT / "experiments" / _DEFAULT_EXPERIMENT)
)
DEFAULT_EXPERIMENT = os.environ.get(
    "POLICY_EXPERIMENT_NAME",
    "w2a_so101_delta_30hz_partial_bridge_init_"
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
    "lr1.000e-04_layer20_bsz4",
)
DEFAULT_VIDEO_CKPT = pathlib.Path(
    os.environ.get(
        "MIMIC_VIDEO_BACKBONE_PATH",
        _DEFAULT_ROOT / "shared/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt",
    )
)
DEFAULT_ACTION_CKPT = pathlib.Path(
    os.environ.get(
        "MIMIC_VIDEO_ACTION_DECODER_CKPT_PATH",
        _DEFAULT_EXPERIMENT_ROOT / "checkpoints/action_decoder/latest",
    )
)
DEFAULT_DATA_DIR = pathlib.Path(
    os.environ.get(
        "MIMIC_VIDEO_SHARED_ACTION_DATA_DIR",
        _DEFAULT_ROOT / "shared/data/so101_bottle_action_only_full",
    )
)


@dataclasses.dataclass(frozen=True)
class ServerArgs:
    host: str = "127.0.0.1"
    port: int = 8000
    experiment_name: str = DEFAULT_EXPERIMENT
    video_model_path: pathlib.Path = DEFAULT_VIDEO_CKPT
    video_lora_path: pathlib.Path | None = None
    action_model_path: pathlib.Path = DEFAULT_ACTION_CKPT
    data_dir: pathlib.Path = DEFAULT_DATA_DIR
    stats_path: pathlib.Path | None = None
    num_val_episodes: int = 10
    num_sampling_steps: int = 35
    stop_video_denoising_step: int = 10
    seed: int = 0
    load_text_encoder: bool = False
    use_cuda_graphs: bool = False
    transport: str = "tcp"


def parse_args() -> ServerArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=ServerArgs.host)
    parser.add_argument("--port", type=int, default=ServerArgs.port)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--video-model-path", type=pathlib.Path, default=DEFAULT_VIDEO_CKPT)
    parser.add_argument("--video-lora-path", type=pathlib.Path, default=None)
    parser.add_argument("--action-model-path", type=pathlib.Path, default=DEFAULT_ACTION_CKPT)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--stats-path",
        type=pathlib.Path,
        default=None,
        help="Load normalizer statistics from JSON and skip dataset/zarr access for online prediction.",
    )
    parser.add_argument("--num-val-episodes", type=int, default=ServerArgs.num_val_episodes)
    parser.add_argument("--num-sampling-steps", type=int, default=ServerArgs.num_sampling_steps)
    parser.add_argument("--stop-video-denoising-step", type=int, default=ServerArgs.stop_video_denoising_step)
    parser.add_argument("--seed", type=int, default=ServerArgs.seed)
    parser.add_argument(
        "--load-text-encoder",
        action="store_true",
        help="Allow text-prompt requests. Without this, requests must include observation/prompt_embedding.",
    )
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--transport", choices=("tcp", "websocket"), default=ServerArgs.transport)
    return ServerArgs(**vars(parser.parse_args()))


class SO101PolicyServer:
    def __init__(self, args: ServerArgs) -> None:
        self.args = args
        self.policy = MimicVideoSO101Policy(
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
                load_text_encoder=args.load_text_encoder,
                use_cuda_graphs=args.use_cuda_graphs,
            )
        )
        self.request_count = 0

    def metadata(self) -> dict[str, Any]:
        metadata = self.policy.metadata()
        metadata.update(
            {
                "server": "mimic-video-so101-prediction",
                "host": self.args.host,
                "port": self.args.port,
                "payload_protocol": PAYLOAD_PROTOCOL,
                "transport": TCP_TRANSPORT if self.args.transport == "tcp" else "websocket",
                "observation_keys": [
                    "observation/images/front",
                    "observation/state",
                    "prompt or observation/prompt_embedding",
                ],
                "action_horizon": self.policy.action_horizon,
                "action_dim": self.policy.action_dim,
                "load_text_encoder": self.args.load_text_encoder,
            }
        )
        return metadata

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self.request_count += 1
        seed = int(request.get("seed", self.args.seed + self.request_count - 1))
        start = time.perf_counter()
        result = self.policy.predict_from_observation(request, seed=seed)
        actions = np.asarray(result["actions"], dtype=np.float32)
        return {
            "actions": actions,
            "server_timing": {
                "total_ms": (time.perf_counter() - start) * 1000.0,
                "policy_ms": float(result["elapsed_ms"]),
                "request_count": self.request_count,
                "seed": seed,
            },
            "metadata": {
                "action_horizon": self.policy.action_horizon,
                "action_dim": self.policy.action_dim,
                "action_target_frequency": self.policy.action_target_frequency,
                "action_delta_mode": self.policy.action_delta_mode,
            },
        }


def serve(args: ServerArgs) -> None:
    if args.transport == "websocket":
        serve_websocket(args)
    else:
        serve_tcp(args)


def serve_tcp(args: ServerArgs) -> None:
    server_state = SO101PolicyServer(args)
    print(json.dumps(server_state.metadata(), indent=2), flush=True)

    class RequestHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            send_framed(self.request, server_state.metadata())
            while True:
                try:
                    request = recv_framed(self.request)
                except EOFError:
                    break
                except Exception:
                    send_framed(self.request, traceback.format_exc())
                    break
                try:
                    if not isinstance(request, dict):
                        raise TypeError(f"Expected request dict, got {type(request)!r}")
                    send_framed(self.request, server_state.handle_request(request))
                except Exception:
                    send_framed(self.request, traceback.format_exc())

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    print(f"Serving mimic-video SO-101 policy on tcp://{args.host}:{args.port}", flush=True)
    with socketserver.ThreadingTCPServer((args.host, args.port), RequestHandler) as tcp_server:
        tcp_server.serve_forever()


def serve_websocket(args: ServerArgs) -> None:
    try:
        import websockets.sync.server
    except ImportError as exc:
        raise RuntimeError("--transport websocket requires the optional 'websockets' package. Use the default TCP transport instead.") from exc

    server_state = SO101PolicyServer(args)
    print(json.dumps(server_state.metadata(), indent=2), flush=True)

    def handler(websocket) -> None:
        websocket.send(packb(server_state.metadata()))
        while True:
            try:
                message = websocket.recv()
            except Exception:
                break
            try:
                request = unpackb(message)
                if not isinstance(request, dict):
                    raise TypeError(f"Expected request dict, got {type(request)!r}")
                websocket.send(packb(server_state.handle_request(request)))
            except Exception:
                websocket.send(traceback.format_exc())

    print(f"Serving mimic-video SO-101 policy on ws://{args.host}:{args.port}", flush=True)
    with websockets.sync.server.serve(handler, args.host, args.port, compression=None, max_size=None) as ws_server:
        ws_server.serve_forever()


if __name__ == "__main__":
    serve(parse_args())
