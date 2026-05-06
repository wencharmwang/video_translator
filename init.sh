#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/artifacts"
DEFAULT_REF="$ARTIFACTS_DIR/gpt_sovits_ref_from_translated_video.wav"
FALLBACK_REF="$ARTIFACTS_DIR/gpt_sovits_ref.wav"
SAMPLE_REF_VIDEO="$ROOT_DIR/samples/translated_videos/video_test_5_zh_CN.mov"
CONFIG_PATH="$ROOT_DIR/video_translator.defaults.json"
SERVICE_CONFIG_PATH="$ROOT_DIR/gpt_sovits.local.json"
LOCAL_PYTHON="$ROOT_DIR/.venv/bin/python"
DEFAULT_SERVICE_URL="http://127.0.0.1:9880"
GPT_REPO_CANDIDATES=(
  "${GPT_SOVITS_HOME:-}"
  "$ROOT_DIR/../GPT-SoVITS"
  "$ROOT_DIR/GPT-SoVITS"
)

ensure_binary() {
  local binary="$1"
  if command -v "$binary" >/dev/null 2>&1; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "[init] Missing $binary, installing via Homebrew"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install ffmpeg
    return 0
  fi
  echo "[init] Missing required binary: $binary"
  exit 1
}

ensure_local_python() {
  if [[ -x "$LOCAL_PYTHON" ]]; then
    return 0
  fi
  echo "[init] Missing local application environment. Run ./install.sh first."
  exit 1
}

find_gpt_repo() {
  local candidate
  for candidate in "${GPT_REPO_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -f "$candidate/api_v2.py" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

clone_gpt_repo_if_needed() {
  local target="$ROOT_DIR/../GPT-SoVITS"
  if [[ -f "$target/api_v2.py" ]]; then
    echo "$target"
    return 0
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "[init] git not found and GPT-SoVITS checkout is missing"
    exit 1
  fi
  echo "[init] Cloning GPT-SoVITS into $target"
  git clone https://github.com/RVC-Boss/GPT-SoVITS.git "$target"
  echo "$target"
}

ensure_python311() {
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "[init] Installing python@3.11 via Homebrew"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install python@3.11 >/dev/null
    command -v python3.11
    return 0
  fi
  echo "[init] python3.11 not found"
  exit 1
}

ensure_gpt_env() {
  local gpt_root="$1"
  local python311
  python311="$(ensure_python311)"
  if [[ ! -x "$gpt_root/.venv311/bin/python" ]]; then
    echo "[init] Creating GPT-SoVITS Python environment"
    "$python311" -m venv "$gpt_root/.venv311"
  fi

  echo "[init] Installing GPT-SoVITS dependencies"
  "$gpt_root/.venv311/bin/pip" install -U pip setuptools wheel
  "$gpt_root/.venv311/bin/pip" install torch torchaudio
  "$gpt_root/.venv311/bin/pip" install -r "$gpt_root/extra-req.txt" --no-deps
  "$gpt_root/.venv311/bin/pip" install -r "$gpt_root/requirements.txt"
}

ensure_default_reference() {
  mkdir -p "$ARTIFACTS_DIR"
  if [[ -f "$DEFAULT_REF" ]]; then
    return 0
  fi
  if [[ -f "$SAMPLE_REF_VIDEO" ]]; then
    echo "[init] Extracting default GPT-SoVITS reference clip"
    ffmpeg -y -ss 16.3 -to 26.3 -i "$SAMPLE_REF_VIDEO" -vn -ac 1 -ar 32000 "$DEFAULT_REF" >/dev/null 2>&1
    return 0
  fi
  if [[ -f "$FALLBACK_REF" ]]; then
    echo "[init] Reusing fallback GPT-SoVITS reference clip"
    cp "$FALLBACK_REF" "$DEFAULT_REF"
    return 0
  fi
  echo "[init] No bundled reference source available"
  exit 1
}

write_runtime_defaults() {
  cat > "$CONFIG_PATH" <<EOF
{
  "target_language": "zh-CN",
  "gpt_sovits_url": "$DEFAULT_SERVICE_URL",
  "gpt_sovits_ref_mode": "auto-single-speaker",
  "gpt_sovits_ref_audio": "$DEFAULT_REF",
  "gpt_sovits_fallback_female_audio": "$ROOT_DIR/female.mp3",
  "gpt_sovits_fallback_male_audio": "$ROOT_DIR/male.wav",
  "gpt_sovits_prompt_lang": "zh",
  "gpt_sovits_prompt_text": "您好，很高兴能为您提供配音服务，选择您感兴趣的音色，让我们一起开启声音创作的奇幻之旅吧。",
  "gpt_sovits_text_lang": "zh",
  "gpt_sovits_text_split_method": "cut5",
  "gpt_sovits_speed": 1.0
}
EOF
  echo "[init] Wrote runtime defaults to $CONFIG_PATH"
}

write_service_config() {
  local gpt_root="$1"
  cat > "$SERVICE_CONFIG_PATH" <<EOF
{
  "gpt_sovits_root": "$gpt_root",
  "python_bin": "$gpt_root/.venv311/bin/python",
  "api_script": "$gpt_root/api_v2.py",
  "config_path": "$gpt_root/GPT_SoVITS/configs/tts_infer.yaml",
  "host": "127.0.0.1",
  "port": 9880,
  "url": "$DEFAULT_SERVICE_URL",
  "log_path": "$ROOT_DIR/artifacts/gpt_sovits_service.log",
  "pid_path": "$ROOT_DIR/artifacts/gpt_sovits_service.pid"
}
EOF
  echo "[init] Wrote service config to $SERVICE_CONFIG_PATH"
}

ensure_binary ffmpeg
ensure_binary ffprobe
ensure_local_python

echo "[init] Preparing local translation, ASR, OCR and separation assets"
"$LOCAL_PYTHON" "$ROOT_DIR/prepare_runtime_assets.py"

GPT_ROOT="$(find_gpt_repo || true)"
if [[ -z "$GPT_ROOT" ]]; then
  GPT_ROOT="$(clone_gpt_repo_if_needed)"
fi

ensure_gpt_env "$GPT_ROOT"
ensure_default_reference

if [[ -f "$ROOT_DIR/female.mp3" ]]; then
  echo "[init] Conservative fallback female reference found: $ROOT_DIR/female.mp3"
else
  echo "[init] Conservative fallback female reference missing: $ROOT_DIR/female.mp3"
fi

if [[ -f "$ROOT_DIR/male.wav" ]]; then
  echo "[init] Conservative fallback male reference found: $ROOT_DIR/male.wav"
else
  echo "[init] Conservative fallback male reference missing: $ROOT_DIR/male.wav"
fi

write_runtime_defaults
write_service_config "$GPT_ROOT"

echo "[init] Starting GPT-SoVITS service"
"$ROOT_DIR/start-gpt-sovits.sh"

echo "[init] Done"
echo "[init] Next: ./video-translator /path/to/video.mov"
