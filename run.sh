#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python -m uvicorn api.server:app --host "${HANA_HOST:-127.0.0.1}" --port "${HANA_PORT:-7860}" --workers 1
