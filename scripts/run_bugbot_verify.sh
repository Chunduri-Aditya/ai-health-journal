#!/usr/bin/env bash
# GateGuard: callers operator shell / local verify after Bugbot fixes.
# No prior run_bugbot_verify.sh. Affected API: import smoke + targeted pytest + optional live smoke.
# Data schemas: none. User: launch it on the cursor browser and make a script and run tests
#
# Verifies the Bugbot src/ migration: imports, compile, and core unit tests.
# Optionally smoke-tests a running service when JOURNAL_AGENT_URL is set
# (default http://127.0.0.1:8080 if that port answers).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

export ENV="${ENV:-test}"
export RETRIEVAL_ENABLED="${RETRIEVAL_ENABLED:-false}"
export VECTOR_BACKEND="${VECTOR_BACKEND:-none}"
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export ALLOW_CLOUD_LLM="${ALLOW_CLOUD_LLM:-false}"
export ALLOW_HASH_EMBEDDER="${ALLOW_HASH_EMBEDDER:-true}"

echo "== 1. import smoke =="
"$PY" - <<'PY'
from src.service.main import app as fastapi_app
from src.agent.graph import run_agent_turn
from src.safety import redact
from src.app import app as flask_app

assert fastapi_app is not None
assert callable(run_agent_turn)
assert callable(redact)
assert flask_app.name
print("imports ok:", type(fastapi_app).__name__, flask_app.name)
PY

echo "== 2. compileall src =="
"$PY" -m compileall -q src
echo "compileall ok"

echo "== 3. targeted pytest =="
"$PY" -m pytest -q \
  tests/test_service_health.py \
  tests/test_service_agent.py \
  tests/test_agent_routing.py \
  tests/test_agent_respond.py \
  tests/test_safety_agent.py \
  tests/test_providers.py

SMOKE_URL="${JOURNAL_AGENT_URL:-}"
FLASK_URL="${JOURNAL_FLASK_URL:-}"

if [ -z "$SMOKE_URL" ] && curl -sf "http://127.0.0.1:8080/healthz" >/dev/null 2>&1; then
  SMOKE_URL="http://127.0.0.1:8080"
fi
if [ -z "$FLASK_URL" ] && curl -sf "http://127.0.0.1:5050/ping" >/dev/null 2>&1; then
  FLASK_URL="http://127.0.0.1:5050"
fi

if [ -n "$SMOKE_URL" ]; then
  echo "== 4. live service smoke against ${SMOKE_URL} =="
  JOURNAL_AGENT_URL="$SMOKE_URL" "$PY" scripts/smoke_service.py
else
  echo "== 4. live service smoke skipped (no :8080) =="
fi

if [ -n "$FLASK_URL" ]; then
  echo "== 5. live flask analyze smoke against ${FLASK_URL} =="
  # Use the server's real .env LLM (do not force ollama from this script's ENV=test).
  unset LLM_BACKEND ALLOW_CLOUD_LLM
  JOURNAL_FLASK_URL="$FLASK_URL" "$PY" scripts/smoke_flask.py
else
  echo "== 5. live flask smoke skipped (no :5050) =="
fi

echo "PASS: all verify steps"
