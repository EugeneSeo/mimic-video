#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101_hetero_ee}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi

mvs_load_paths "${experiment}"
mvs_create_layout
mvs_prepare_runtime_env

repo_id="${MVS_PROMPT_EMBEDDING_REPO:-dreamdifferent/mimic-video-so101-multi-object-delta30-action-decoder}"
remote_dir="${MVS_PROMPT_EMBEDDING_REMOTE_DIR:-prompt_embeddings}"

echo "Downloading SO-101 prompt embeddings"
echo "  repo:        ${repo_id}"
echo "  remote dir:  ${remote_dir}"
echo "  local dir:   ${MVS_PROMPT_EMBEDDING_DIR}"
echo "  manifest:    ${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}"

if command -v hf >/dev/null 2>&1; then
  hf download "${repo_id}" \
    --repo-type model \
    --include "${remote_dir}/*" \
    --local-dir "${MVS_EXP_ROOT}" \
    --local-dir-use-symlinks False \
    "$@"
else
  python - <<PY "$repo_id" "$remote_dir" "$MVS_EXP_ROOT"
import sys

repo_id, remote_dir, local_dir = sys.argv[1:]
try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "ERROR: need either the `hf` CLI or Python package `huggingface_hub` "
        "to download prompt embeddings."
    ) from exc

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=[f"{remote_dir}/*"],
)
PY
fi

mvs_require_file "${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}" "prompt embedding manifest"

python - <<PY "$MVS_PROMPT_EMBEDDING_MANIFEST_PATH"
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
prompts = payload.get("prompts", payload)
if not isinstance(prompts, dict) or not prompts:
    raise SystemExit(f"ERROR: invalid or empty prompt embedding manifest: {manifest_path}")

missing = []
for prompt, rel_path in prompts.items():
    if not isinstance(prompt, str) or not isinstance(rel_path, str):
        raise SystemExit(f"ERROR: invalid manifest entry: {prompt!r} -> {rel_path!r}")
    path = pathlib.Path(rel_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        missing.append((prompt, str(path)))

if missing:
    preview = "\\n".join(f"  {prompt!r}: {path}" for prompt, path in missing[:10])
    raise SystemExit(f"ERROR: missing {len(missing)} prompt embedding files:\\n{preview}")

print(f"Validated {len(prompts)} prompt embeddings in {manifest_path}")
print("Example prompts:")
for prompt in sorted(prompts)[:5]:
    print(f"  - {prompt}")
PY

cat <<EOF

Prompt embedding cache is ready.

Start the EE policy server with:

  MVS_SO101_PORT="\${MVS_SO101_PORT:-18000}" \\
  MVS_SERVER_USE_PROMPT_CACHE=1 \\
  bash scripts/so101/run_policy_server.sh ${experiment}

Client --task must exactly match one prompt in:
  ${MVS_PROMPT_EMBEDDING_MANIFEST_PATH}
EOF
