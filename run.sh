#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HANA_DTYPE="${HANA_DTYPE:-fp32}"
export HANA_REMOVE_WEIGHT_NORM="${HANA_REMOVE_WEIGHT_NORM:-1}"
export HANA_TORCH_COMPILE="${HANA_TORCH_COMPILE:-0}"
export HANA_QUEUE_SIZE="${HANA_QUEUE_SIZE:-4}"
export HANA_CACHE_ITEMS="${HANA_CACHE_ITEMS:-32}"
export HANA_CACHE_MAX_MB="${HANA_CACHE_MAX_MB:-64}"
export HANA_TEXT_CACHE_ITEMS="${HANA_TEXT_CACHE_ITEMS:-128}"
exec python -m uvicorn api.server:app --host "${HANA_HOST:-127.0.0.1}" --port "${HANA_PORT:-7860}" --workers 1
