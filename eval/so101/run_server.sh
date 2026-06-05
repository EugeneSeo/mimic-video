#!/usr/bin/env bash
set -euo pipefail

python eval/so101/policy_server.py \
  --host "${MIMIC_VIDEO_SO101_HOST:-127.0.0.1}" \
  --port "${MIMIC_VIDEO_SO101_PORT:-8000}" \
  "$@"
