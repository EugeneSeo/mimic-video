"""Export one SO-101 prompt embedding for dataset-free policy serving."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig  # noqa: E402
from imaginaire.constants import T5_MODEL_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = CosmosT5TextEncoder(
        config=CosmosT5TextEncoderConfig(ckpt_path=T5_MODEL_DIR),
        device=args.device,
    )
    embedding = encoder.encode_prompts(args.prompt, max_length=512, return_mask=False).cpu().numpy()
    embedding = embedding[0].astype(np.float16 if args.dtype == "float16" else np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embedding)
    if args.manifest is not None:
        update_manifest(args.manifest, prompt=args.prompt, embedding_path=args.output)
    print(f"Wrote {args.output}")
    print(f"shape={embedding.shape} dtype={embedding.dtype}")


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def update_manifest(manifest_path: pathlib.Path, *, prompt: str, embedding_path: pathlib.Path) -> None:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    prompts = payload.setdefault("prompts", {})
    relative_embedding_path = pathlib.Path(os.path.relpath(embedding_path.resolve(), start=manifest_path.parent.resolve()))
    prompts[normalize_prompt(prompt)] = relative_embedding_path.as_posix()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
