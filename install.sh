#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
RUN_INIT=true

ensure_binary() {
  local binary="$1"
  if command -v "$binary" >/dev/null 2>&1; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "[install] Missing $binary, installing via Homebrew"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install ffmpeg
    return 0
  fi
  echo "[install] Missing required binary: $binary"
  exit 1
}


while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-init)
      RUN_INIT=false
      shift
      ;;
    *)
      echo "[install] Unknown argument: $1"
      exit 1
      ;;
  esac
done
ensure_binary ffmpeg
ensure_binary ffprobe

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[install] Creating virtual environment"
  python3 -m venv "$VENV_DIR"
fi

echo "[install] Installing Python dependencies"
"$VENV_DIR/bin/pip" install -U pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"
"$VENV_DIR/bin/pip" install -e "$ROOT_DIR"

if [[ "$RUN_INIT" == "true" ]]; then
  echo "[install] Running GPT-SoVITS initialization"
  "$ROOT_DIR/init.sh"
fi

echo "[install] Done"
echo "[install] Next: ./video-translator /path/to/video.mov"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_CONFIG_PATH="$ROOT_DIR/gpt_sovits.local.json"

if [[ ! -f "$SERVICE_CONFIG_PATH" ]]; then
  echo "[start] Missing GPT-SoVITS service config. Run ./init.sh first."
  exit 1
fi

read_config() {
  local key="$1"
  "$ROOT_DIR/.venv/bin/python" - <<PY
import json
from pathlib import Path
config = json.loads(Path("$SERVICE_CONFIG_PATH").read_text(encoding="utf-8"))
print(config["$key"])
PY
}

URL="$(read_config url)"
HOST="$(read_config host)"
PORT="$(read_config port)"
PYTHON_BIN="$(read_config python_bin)"
API_SCRIPT="$(read_config api_script)"
CONFIG_PATH="$(read_config config_path)"
GPT_ROOT="$(read_config gpt_sovits_root)"
LOG_PATH="$(read_config log_path)"
PID_PATH="$(read_config pid_path)"

mkdir -p "$(dirname "$LOG_PATH")"

if curl -sS -m 3 "$URL/" >/dev/null 2>&1; then
  echo "[start] GPT-SoVITS service already reachable at $URL"
  exit 0
fi

echo "[start] Starting GPT-SoVITS service at $URL"
nohup env PYTHONPATH="$GPT_ROOT:$GPT_ROOT/GPT_SoVITS" "$PYTHON_BIN" "$API_SCRIPT" -a "$HOST" -p "$PORT" -c "$CONFIG_PATH" >"$LOG_PATH" 2>&1 &
echo $! > "$PID_PATH"

for _ in {1..60}; do
  if curl -sS -m 3 "$URL/" >/dev/null 2>&1; then
    echo "[start] GPT-SoVITS service is ready"
    exit 0
  fi
  sleep 2
done

echo "[start] GPT-SoVITS service did not become ready. Check $LOG_PATH"
exit 1
*** Delete File: /Users/wencharm/projects/video_translator/video-translator-local