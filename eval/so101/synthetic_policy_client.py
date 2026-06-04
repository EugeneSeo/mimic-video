"""Small raw client for smoke-testing the mimic-video SO-101 server."""

from __future__ import annotations

import argparse
import dataclasses
import socket
import time
from typing import Any

import numpy as np

from websocket_protocol import recv_framed, send_framed


@dataclasses.dataclass(frozen=True)
class ClientArgs:
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "pick up the bottle"
    num_requests: int = 1
    image_height: int = 480
    image_width: int = 640
    seed: int = 0
    use_prompt_embedding: bool = False
    transport: str = "tcp"


def parse_args() -> ClientArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=ClientArgs.host)
    parser.add_argument("--port", type=int, default=ClientArgs.port)
    parser.add_argument("--prompt", default=ClientArgs.prompt)
    parser.add_argument("--num-requests", type=int, default=ClientArgs.num_requests)
    parser.add_argument("--image-height", type=int, default=ClientArgs.image_height)
    parser.add_argument("--image-width", type=int, default=ClientArgs.image_width)
    parser.add_argument("--seed", type=int, default=ClientArgs.seed)
    parser.add_argument(
        "--use-prompt-embedding",
        action="store_true",
        help="Send a zero embedding for server smoke tests without loading the text encoder.",
    )
    parser.add_argument("--transport", choices=("tcp", "websocket"), default=ClientArgs.transport)
    return ClientArgs(**vars(parser.parse_args()))


def build_observation(args: ClientArgs, rng: np.random.Generator, request_idx: int) -> dict[str, Any]:
    image = rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8)
    state = np.zeros((6,), dtype=np.float32)
    observation: dict[str, Any] = {
        "observation/images/front": image,
        "observation/state": state,
        "prompt": args.prompt,
        "seed": args.seed + request_idx,
    }
    if args.use_prompt_embedding:
        observation["observation/prompt_embedding"] = np.zeros((512, 1024), dtype=np.float32)
    return observation


def main(args: ClientArgs) -> None:
    if args.transport == "websocket":
        main_websocket(args)
    else:
        main_tcp(args)


def main_tcp(args: ClientArgs) -> None:
    rng = np.random.default_rng(args.seed)
    with socket.create_connection((args.host, args.port)) as sock:
        metadata = recv_framed(sock)
        run_requests(args, rng, metadata, send=lambda obj: send_framed(sock, obj), recv=lambda: recv_framed(sock))


def main_websocket(args: ClientArgs) -> None:
    try:
        import websockets.sync.client
        from websocket_protocol import packb, unpackb
    except ImportError as exc:
        raise RuntimeError("--transport websocket requires the optional 'websockets' package. Use the default TCP transport instead.") from exc

    uri = f"ws://{args.host}:{args.port}"
    rng = np.random.default_rng(args.seed)
    with websockets.sync.client.connect(uri, compression=None, max_size=None) as websocket:
        metadata = unpackb(websocket.recv())
        run_requests(args, rng, metadata, send=lambda obj: websocket.send(packb(obj)), recv=lambda: unpackb(websocket.recv()))


def run_requests(args: ClientArgs, rng: np.random.Generator, metadata: dict[str, Any], *, send, recv) -> None:
    print(f"metadata={metadata}")
    expected_shape = (int(metadata.get("action_horizon", 15)), int(metadata.get("action_dim", 6)))

    for request_idx in range(args.num_requests):
        start = time.perf_counter()
        send(build_observation(args, rng, request_idx))
        response = recv()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(response, str):
            raise RuntimeError(f"Server returned error:\n{response}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != expected_shape:
            raise RuntimeError(f"Expected action shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned non-finite actions")
        print(
            f"request={request_idx} actions.shape={actions.shape} "
            f"client_elapsed_ms={elapsed_ms:.1f} first_action={np.array2string(actions[0], precision=3)}"
        )
        print(f"server_timing={response.get('server_timing', {})}")


if __name__ == "__main__":
    main(parse_args())
