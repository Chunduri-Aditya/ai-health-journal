#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AI Health Journal — one-shot launcher
#
# Usage:
#   ./start.sh                 core install, start Flask on http://127.0.0.1:5000
#   ./start.sh --full          also install optional deps (Chroma, Pinecone, Whisper)
#   ./start.sh --no-install    skip dependency install (just preflight + run)
#   ./start.sh --skip-model    don't prompt to pull a default model
#   ./start.sh --check         run preflight + setup but DO NOT launch Flask
#   ./start.sh --help
#
# Behavior:
#   1. Verifies Python 3.8+, Ollama binary, and the Ollama daemon.
#      If the daemon isn't running, tries to start it in the background
#      and shuts it down again on exit.
#   2. Creates ./venv/ on first run, reinstalls deps only when the
#      requirements files change.
#   3. Writes a template .env on first run (never overwrites).
#   4. Launches app.py in the foreground. Ctrl+C stops cleanly.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
REQS_CORE="requirements-core.txt"
REQS_OPT="requirements-optional.txt"
ENV_FILE=".env"

# ── Flags ────────────────────────────────────────────────────────────────────
INSTALL_FULL=0
SKIP_INSTALL=0
SKIP_MODEL_PROMPT=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --full)        INSTALL_FULL=1 ;;
    --no-install)  SKIP_INSTALL=1 ;;
    --skip-model)  SKIP_MODEL_PROMPT=1 ;;
    --check)       CHECK_ONLY=1; SKIP_MODEL_PROMPT=1 ;;
    -h|--help)
      sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# ── Logging helpers ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
  C_OK="$(printf '\033[0;32m')"
  C_WARN="$(printf '\033[0;33m')"
  C_ERR="$(printf '\033[0;31m')"
  C_DIM="$(printf '\033[2m')"
  C_OFF="$(printf '\033[0m')"
else
  C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi

say()   { printf "%s%s%s\n" "$C_DIM" "$1" "$C_OFF"; }
ok()    { printf "%s✓%s %s\n" "$C_OK" "$C_OFF" "$1"; }
warn()  { printf "%s!%s %s\n" "$C_WARN" "$C_OFF" "$1"; }
die()   { printf "%s✗%s %s\n" "$C_ERR" "$C_OFF" "$1" >&2; exit 1; }
section() { printf "\n%s── %s ──%s\n" "$C_DIM" "$1" "$C_OFF"; }

# ── Cleanup hook ─────────────────────────────────────────────────────────────
OLLAMA_PID=""
cleanup() {
  if [ -n "${OLLAMA_PID}" ] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    say "Stopping Ollama daemon we started (pid ${OLLAMA_PID})…"
    kill "${OLLAMA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Hash helper (macOS + Linux) ──────────────────────────────────────────────
hash_of() {
  local f="$1"
  if [ ! -f "$f" ]; then echo ""; return; fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    stat -f '%m' "$f" 2>/dev/null || stat -c '%Y' "$f" 2>/dev/null
  fi
}

# ── 1. Python ────────────────────────────────────────────────────────────────
section "Preflight"

if ! command -v python3 >/dev/null 2>&1; then
  die "python3 not found. Install Python 3.8+ and retry. https://www.python.org/downloads/"
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
  die "Python 3.8+ required (found ${PY_MAJOR}.${PY_MINOR})."
fi
ok "Python ${PY_MAJOR}.${PY_MINOR}"

# ── 2. Ollama binary ─────────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
  cat >&2 <<EOF
${C_ERR}✗${C_OFF} Ollama is not installed.
    macOS:  brew install ollama      (or download the .dmg from https://ollama.com)
    Linux:  curl -fsSL https://ollama.com/install.sh | sh
    Then re-run: ./start.sh
EOF
  exit 1
fi
ok "Ollama CLI: $(ollama --version 2>/dev/null | head -1 || echo present)"

# ── 3. Ollama daemon ─────────────────────────────────────────────────────────
ollama_up() { curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; }

if ollama_up; then
  ok "Ollama daemon is reachable on :11434"
else
  warn "Ollama daemon is not running — starting it in the background…"
  mkdir -p .runtime
  nohup ollama serve >.runtime/ollama.log 2>&1 &
  OLLAMA_PID=$!
  # Wait up to ~15s for the daemon to accept connections.
  for i in $(seq 1 30); do
    if ollama_up; then break; fi
    sleep 0.5
  done
  if ! ollama_up; then
    die "Could not reach Ollama daemon after starting it. See .runtime/ollama.log"
  fi
  ok "Ollama daemon started (pid ${OLLAMA_PID}, log: .runtime/ollama.log)"
fi

# ── 4. At least one chat model ───────────────────────────────────────────────
INSTALLED_MODELS_JSON="$(curl -fsS http://localhost:11434/api/tags 2>/dev/null || echo '{}')"
MODEL_COUNT=$(python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read() or '{}')
    models = [m for m in data.get('models', []) if 'embed' not in m.get('name','').lower()]
    print(len(models))
except Exception:
    print(0)
" <<<"$INSTALLED_MODELS_JSON")

if [ "$MODEL_COUNT" -eq 0 ]; then
  if [ "$SKIP_MODEL_PROMPT" -eq 1 ]; then
    warn "No Ollama chat models installed. Pull one before using /analyze (e.g. ollama pull gemma3:4b)."
  else
    warn "No Ollama chat models installed yet."
    printf "  Pull a balanced default now? [gemma3:4b, ~3 GB]  [Y/n] "
    read -r ANSWER || ANSWER=""
    case "${ANSWER:-Y}" in
      y|Y|yes|YES|"")
        say "Pulling gemma3:4b (this may take a few minutes)…"
        ollama pull gemma3:4b || die "ollama pull failed. Check network / disk space."
        ok "gemma3:4b ready"
        ;;
      *)
        warn "Skipped. The app will still start, but /analyze will fail until a model is pulled."
        ;;
    esac
  fi
else
  ok "Ollama chat models installed: ${MODEL_COUNT}"
fi

# ── 5. Virtualenv ────────────────────────────────────────────────────────────
section "Python environment"

if [ ! -d "$VENV_DIR" ]; then
  say "Creating virtualenv at ./${VENV_DIR}/"
  python3 -m venv "$VENV_DIR"
  ok "virtualenv created"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -x "$VENV_PY" ]; then
  die "virtualenv looks broken (missing $VENV_PY). Remove ./${VENV_DIR}/ and re-run."
fi

# ── 6. Dependency install (idempotent via content hash) ──────────────────────
MARKER_DIR="$VENV_DIR/.start-sh"
mkdir -p "$MARKER_DIR"

install_if_changed() {
  local req="$1"
  local tag="$2"
  local marker="$MARKER_DIR/${tag}.hash"
  local current expected
  current="$(hash_of "$req")"
  expected="$(cat "$marker" 2>/dev/null || echo "")"
  if [ "$SKIP_INSTALL" -eq 1 ]; then
    say "--no-install: skipping ${req}"
    return 0
  fi
  if [ "$current" = "$expected" ] && [ -n "$current" ]; then
    ok "${tag} dependencies already up to date"
    return 0
  fi
  say "Installing ${tag} dependencies from ${req}…"
  "$VENV_PIP" install --quiet --upgrade pip
  "$VENV_PIP" install --quiet -r "$req"
  echo "$current" > "$marker"
  ok "${tag} dependencies installed"
}

install_if_changed "$REQS_CORE" "core"

if [ "$INSTALL_FULL" -eq 1 ]; then
  if [ -f "$REQS_OPT" ]; then
    install_if_changed "$REQS_OPT" "optional"
  else
    warn "--full requested but ${REQS_OPT} is missing"
  fi
fi

# ── 7. .env bootstrap ────────────────────────────────────────────────────────
section "Configuration"

if [ ! -f "$ENV_FILE" ]; then
  if [ -f ".env.example" ]; then
    cp ".env.example" "$ENV_FILE"
    ok "Wrote ${ENV_FILE} (copied from .env.example)"
  else
    warn ".env.example missing — writing a minimal ${ENV_FILE} you can edit later."
    cat > "$ENV_FILE" <<'ENV_MIN'
MODEL_SELECTION_STRATEGY=balanced
RETRIEVAL_ENABLED=false
VECTOR_BACKEND=none
QUALITY_MODE_DEFAULT=false
ENV_MIN
    ok "Wrote ${ENV_FILE} (minimal fallback)"
  fi
else
  ok "${ENV_FILE} already present (left untouched)"
fi

# ── 8. Launch ────────────────────────────────────────────────────────────────
section "Launch"
HOST_URL="http://127.0.0.1:5000"

if [ "$CHECK_ONLY" -eq 1 ]; then
  ok "Preflight complete. --check requested; not launching Flask."
  exit 0
fi

ok "Starting Flask app — open ${HOST_URL} in your browser"
say "Press Ctrl+C to stop."
echo

# Keep ourselves in the cleanup trap by NOT using exec when we started the
# daemon, so the trap can kill it on Ctrl+C.
if [ -n "$OLLAMA_PID" ]; then
  "$VENV_PY" app.py
else
  # Nothing to clean up — exec for a tidier process tree.
  exec "$VENV_PY" app.py
fi
