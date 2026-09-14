#!/usr/bin/env bash
# Streamlit front end for the assistant.
#
# Bound to 127.0.0.1 deliberately. Streamlit's default is 0.0.0.0, which on a laptop advertises
# the app on the local network and, behind a permissive firewall, beyond it -- an unauthenticated
# page that can drive a GPU. Pass UI_HOST=0.0.0.0 to override, knowingly.
set -euo pipefail
cd "$(dirname "$0")/.."

UI_PORT="${UI_PORT:-8501}"
UI_HOST="${UI_HOST:-127.0.0.1}"

[ -x .venv-ui/bin/streamlit ] || {
  echo "UI venv missing. Create it with:" >&2
  echo "  uv venv --python 3.12 .venv-ui && uv pip install --python .venv-ui/bin/python streamlit httpx" >&2
  exit 1
}

echo "UI on http://${UI_HOST}:${UI_PORT}  (expects the API on :8000 and vLLM on :8001)"
exec .venv-ui/bin/streamlit run ui/app.py \
  --server.address "$UI_HOST" --server.port "$UI_PORT" \
  --server.headless true --browser.gatherUsageStats false
