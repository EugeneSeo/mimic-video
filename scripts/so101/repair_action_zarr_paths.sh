#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/paths.sh"

experiment="${MVS_EXPERIMENT:-so101-homogeneous-delta30}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  experiment="$1"
  shift
fi
mvs_load_paths "${experiment}"

mvs_require_dir "${MVS_SHARED_ACTION_DATA_DIR}" "action zarr"

python_bin="python"
if [[ -x "${REPO_ROOT}/model/.venv/bin/python" ]]; then
  python_bin="${REPO_ROOT}/model/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
fi

"${python_bin}" - "${MVS_SHARED_ACTION_DATA_DIR}" <<'PY'
import pathlib
import pickle
import sys

root = pathlib.Path(sys.argv[1]).resolve()
episode_paths = sorted(root.glob("episode_*.zarr"))
if not episode_paths:
    raise SystemExit(f"ERROR: no episode_*.zarr directories found under {root}")

paths_cache = root / "paths.pkl"
with paths_cache.open("wb") as f:
    pickle.dump(episode_paths, f)

print(f"Rewrote {paths_cache}")
print(f"Episodes: {len(episode_paths)}")
print(f"First: {episode_paths[0]}")
print(f"Last: {episode_paths[-1]}")
PY
