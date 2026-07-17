#!/usr/bin/env bash
#
# AI Health Journal launcher.
#
# Usage:
#   ./start.sh                 install deps if needed, then start Flask
#   ./start.sh --no-install    skip dependency installation
#   ./start.sh --skip-model    do not offer to pull missing Ollama models
#   ./start.sh --check         run setup checks without launching Flask
#   ./start.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
REQ_FILE="requirements.txt"
MARKER_DIR="$VENV_DIR/.start-sh"
ENV_FILE=".env"

SKIP_INSTALL=0
SKIP_MODEL_PROMPT=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-install)
      SKIP_INSTALL=1
      ;;
    --skip-model)
      SKIP_MODEL_PROMPT=1
      ;;
    --check)
      CHECK_ONLY=1
      SKIP_MODEL_PROMPT=1
      ;;
    -h|--help)
      sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf "Unknown flag: %s\n" "$1" >&2
      exit 2
      ;;
  esac
  shift
done

if [ -t 1 ]; then
  C_OK="$(printf '\033[0;32m')"
  C_WARN="$(printf '\033[0;33m')"
  C_ERR="$(printf '\033[0;31m')"
  C_DIM="$(printf '\033[2m')"
  C_OFF="$(printf '\033[0m')"
else
  C_OK=""
  C_WARN=""
  C_ERR=""
  C_DIM=""
  C_OFF=""
fi

say() { printf "%s%s%s\n" "$C_DIM" "$1" "$C_OFF"; }
ok() { printf "%sOK%s %s\n" "$C_OK" "$C_OFF" "$1"; }
warn() { printf "%sWARN%s %s\n" "$C_WARN" "$C_OFF" "$1"; }
die() {
  printf "%sERROR%s %s\n" "$C_ERR" "$C_OFF" "$1" >&2
  exit 1
}
section() { printf "\n%s== %s ==%s\n" "$C_DIM" "$1" "$C_OFF"; }

OLLAMA_PID=""
cleanup() {
  if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    say "Stopping Ollama daemon started by this script."
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

hash_of() {
  local file="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  else
    stat -f '%m' "$file" 2>/dev/null || stat -c '%Y' "$file" 2>/dev/null
  fi
}

section "Preflight"

if ! command -v python3 >/dev/null 2>&1; then
  die "python3 was not found. Install Python 3.8+ and retry."
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_OK="$(python3 -c 'import sys; print(int(sys.version_info >= (3, 8)))')"
if [ "$PYTHON_OK" != "1" ]; then
  die "Python 3.8+ is required; found ${PYTHON_VERSION}."
fi
ok "Python ${PYTHON_VERSION}"

if [ ! -f "$REQ_FILE" ]; then
  die "Missing ${REQ_FILE}."
fi

section "Python environment"

if [ ! -d "$VENV_DIR" ]; then
  say "Creating virtualenv at ./${VENV_DIR}/"
  python3 -m venv "$VENV_DIR"
  ok "virtualenv created"
else
  ok "virtualenv present"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -x "$VENV_PY" ] || [ ! -x "$VENV_PIP" ]; then
  die "virtualenv is incomplete. Remove ./${VENV_DIR}/ and run ./start.sh again."
fi

mkdir -p "$MARKER_DIR"
REQ_HASH="$(hash_of "$REQ_FILE")"
HASH_FILE="$MARKER_DIR/requirements.hash"
OLD_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"

if [ "$SKIP_INSTALL" -eq 1 ]; then
  say "--no-install: skipping dependency installation"
elif [ "$REQ_HASH" = "$OLD_HASH" ] && [ -n "$REQ_HASH" ]; then
  ok "dependencies already up to date"
else
  say "Installing dependencies from ${REQ_FILE}"
  "$VENV_PIP" install --quiet -r "$REQ_FILE"
  printf "%s\n" "$REQ_HASH" > "$HASH_FILE"
  ok "dependencies installed"
fi

section "Configuration"

if [ ! -f "$ENV_FILE" ]; then
  printf "FLASK_ENV=development\n" > "$ENV_FILE"
  ok "wrote ${ENV_FILE}"
else
  ok "${ENV_FILE} already present"
fi

section "Ollama"

ollama_up() {
  curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1
}

if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama CLI is not installed. The app will start, but /analyze and /prompt need Ollama."
elif ollama_up; then
  ok "Ollama daemon reachable on localhost:11434"
else
  warn "Ollama daemon is not running; trying to start it."
  mkdir -p .runtime
  nohup ollama serve > .runtime/ollama.log 2>&1 &
  OLLAMA_PID=$!

  for _ in $(seq 1 30); do
    if ollama_up; then
      break
    fi
    sleep 0.5
  done

  if ollama_up; then
    ok "Ollama daemon started"
  else
    warn "Could not reach Ollama after starting it. See .runtime/ollama.log."
  fi
fi

if command -v ollama >/dev/null 2>&1 && ollama_up; then
  for model in "phi3:3.8b" "samantha-mistral:7b"; do
    if ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "$model"; then
      ok "Ollama model present: ${model}"
    elif [ "$SKIP_MODEL_PROMPT" -eq 1 ] || [ ! -t 0 ]; then
      warn "Ollama model missing: ${model}"
    else
      printf "Pull missing Ollama model %s now? [Y/n] " "$model"
      read -r answer || answer=""
      case "${answer:-Y}" in
        y|Y|yes|YES)
          ollama pull "$model"
          ok "Ollama model ready: ${model}"
          ;;
        *)
          warn "Skipped ${model}"
          ;;
      esac
    fi
  done
fi

section "Launch"

if [ "$CHECK_ONLY" -eq 1 ]; then
  ok "setup checks complete"
  exit 0
fi

ok "Starting Flask app at http://127.0.0.1:5000"
say "Press Ctrl+C to stop."
echo

if [ -n "$OLLAMA_PID" ]; then
  "$VENV_PY" app.py
else
  exec "$VENV_PY" app.py
fi
