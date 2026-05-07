#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import re
import requests
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from ocr_subtitles import SubtitleBlock


__version__ = "0.1.5"
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_TARGET_LANGUAGE = "zh-CN"
DEFAULT_ASR_MODEL = "small"
DEFAULT_CHUNK_SECONDS = 300
DEFAULT_TTS_TIMEOUT = 60
DEFAULT_TTS_RETRIES = 3
DEFAULT_TRANSLATE_RETRIES = 3
DEFAULT_TRANSLATE_BATCH_SIZE = 12
DEFAULT_TRANSLATE_TIMEOUT = 30
DEFAULT_TRANSLATION_PROVIDER = "hy-local"
DEFAULT_GPT_SOVITS_URL = "http://127.0.0.1:9880"
DEFAULT_GPT_SOVITS_TEXT_LANG = "auto"
DEFAULT_GPT_SOVITS_TEXT_SPLIT_METHOD = "cut5"
DEFAULT_GPT_SOVITS_SPEED = 1.0
DEFAULT_ASR_DEVICE = "auto"
DEFAULT_ASR_COMPUTE_TYPE = "default"
DEFAULT_DEMUCS_DEVICE = "auto"
DEFAULT_SUBTITLE_SOURCE = "auto"
DEFAULT_OCR_FPS = 1.0
DEFAULT_OCR_MIN_CHARS = 10
DEFAULT_OCR_SIMILARITY = 0.72
DEFAULT_HY_MODEL = "tencent/HY-MT1.5-1.8B"
DEFAULT_HY_GPTQ_MODEL = "tencent/HY-MT1.5-1.8B-GPTQ-Int4"
DEFAULT_HY_FP8_MODEL = "tencent/HY-MT1.5-1.8B-FP8"
DEFAULT_HY_DEVICE = "auto"
DEFAULT_HY_MAX_NEW_TOKENS = 256
DEFAULT_MDX_MODEL = "UVR_MDXNET_KARA_2.onnx"
DEFAULT_PROFILE_FILENAMES = ("video_translator.defaults.json", ".video_translator.defaults.json")
DEFAULT_PROFILE_DIRNAME = ".video-translator"
DEFAULT_PROFILE_BUNDLED_REF = PROJECT_ROOT / "artifacts" / "gpt_sovits_ref_from_translated_video.wav"
DEFAULT_PROFILE_PROMPT_TEXT = "您好，很高兴能为您提供配音服务，选择您感兴趣的音色，让我们一起开启声音创作的奇幻之旅吧。"
DEFAULT_PROFILE_PROMPT_LANG = "zh"
DEFAULT_GPT_SOVITS_REF_MODE = "auto-single-speaker"
DEFAULT_GPT_SOVITS_FALLBACK_FEMALE_REF = PROJECT_ROOT / "female.mp3"
DEFAULT_GPT_SOVITS_FALLBACK_MALE_REF = PROJECT_ROOT / "male.wav"
DEFAULT_RUNTIME_CACHE_DIR = PROJECT_ROOT / DEFAULT_PROFILE_DIRNAME / "models"
GPT_SOVITS_LOCAL_CONFIG = PROJECT_ROOT / "gpt_sovits.local.json"
GPT_SOVITS_START_SCRIPT = PROJECT_ROOT / "start-gpt-sovits.sh"
DUB_SAMPLE_RATE = 44100
DUB_CHANNELS = 1
DEFAULT_BATCH_OUTPUT_DIRNAME = "translated_videos"
MIN_AUDIO_FIT_RATIO = 0.92
MAX_AUDIO_FIT_RATIO = 1.22
QUIET_TTS_RETRIES = 2
QUIET_TTS_RETRY_ISSUES = {"very-quiet", "mostly-silent"}
SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？", "…")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".wmv",
    ".flv",
    ".webm",
    ".mpg",
    ".mpeg",
}

HY_MT_LANGUAGE_NAMES = {
    "zh": "中文",
    "zh-Hant": "繁体中文",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
}


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    source_text: str
    translated_text: str = ""
    tts_path: str = ""
    adjusted_tts_path: str = ""
    tts_duration: float = 0.0
    speed_factor: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class BatchTranslator(Protocol):
    def translate_batch(self, batch: list[str]) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class TranslationBackend:
    name: str
    translator: BatchTranslator
    uses_network_timeout: bool


@dataclass(frozen=True, slots=True)
class GptSovitsConfig:
    api_url: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text_lang: str
    text_split_method: str
    speed_factor: float


@dataclass(frozen=True, slots=True)
class GptSovitsReferenceSelection:
    config: GptSovitsConfig
    source: str
    detail: str
    detected_pitch_hz: float | None = None


@dataclass(slots=True)
class HyMtRuntime:
    tokenizer: object
    model: object
    device: str
    model_id: str


class SubtitleAcquisition(Protocol):
    source_name: str
    subtitle_language: str
    needs_translation: bool
    raw_source_srt: Path | None
    raw_secondary_srt: Path | None


@dataclass(slots=True)
class SrtBlock:
    start: float
    end: float
    text: str


def runtime_default_config_paths() -> list[Path]:
    config_home = Path.home() / ".config" / DEFAULT_PROFILE_DIRNAME
    return [
        PROJECT_ROOT / name for name in DEFAULT_PROFILE_FILENAMES
    ] + [config_home / "config.json"]


def gpt_sovits_service_config_paths() -> list[Path]:
    config_home = Path.home() / ".config" / DEFAULT_PROFILE_DIRNAME
    return [GPT_SOVITS_LOCAL_CONFIG, config_home / GPT_SOVITS_LOCAL_CONFIG.name]


def bundled_runtime_defaults() -> dict[str, object]:
    defaults: dict[str, object] = {
        "target_language": DEFAULT_TARGET_LANGUAGE,
        "gpt_sovits_url": DEFAULT_GPT_SOVITS_URL,
        "gpt_sovits_ref_mode": DEFAULT_GPT_SOVITS_REF_MODE,
        "gpt_sovits_prompt_lang": DEFAULT_PROFILE_PROMPT_LANG,
        "gpt_sovits_prompt_text": DEFAULT_PROFILE_PROMPT_TEXT,
        "gpt_sovits_fallback_female_audio": DEFAULT_GPT_SOVITS_FALLBACK_FEMALE_REF,
        "gpt_sovits_fallback_male_audio": DEFAULT_GPT_SOVITS_FALLBACK_MALE_REF,
    }
    if DEFAULT_PROFILE_BUNDLED_REF.is_file():
        defaults["gpt_sovits_ref_audio"] = DEFAULT_PROFILE_BUNDLED_REF
    return defaults


def load_runtime_defaults() -> tuple[dict[str, object], Path | None]:
    defaults = bundled_runtime_defaults()
    path_like_keys = {
        "gpt_sovits_ref_audio",
        "gpt_sovits_fallback_female_audio",
        "gpt_sovits_fallback_male_audio",
    }
    for config_path in runtime_default_config_paths():
        if not config_path.is_file():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Runtime defaults file must contain a JSON object: {config_path}")
        for key, value in payload.items():
            if key in path_like_keys and value:
                candidate = Path(str(value)).expanduser()
                if not candidate.is_absolute():
                    candidate = (config_path.parent / candidate).resolve()
                defaults[key] = candidate
            else:
                defaults[key] = value
        return defaults, config_path
    return defaults, None


def apply_runtime_defaults(args: argparse.Namespace, raw_args: list[str]) -> tuple[argparse.Namespace, Path | None]:
    defaults, source_path = load_runtime_defaults()
    provided_flags = {arg for arg in raw_args if arg.startswith("--")}
    args.gpt_sovits_reference_manual = any(
        flag in provided_flags
        for flag in ("--gpt-sovits-ref-audio", "--gpt-sovits-prompt-text", "--gpt-sovits-prompt-lang")
    )
    args.gpt_sovits_ref_mode = (
        "manual"
        if args.gpt_sovits_reference_manual
        else str(defaults.get("gpt_sovits_ref_mode") or DEFAULT_GPT_SOVITS_REF_MODE)
    )

    if "--target-language" not in provided_flags and defaults.get("target_language"):
        args.target_language = str(defaults["target_language"])
    if "--gpt-sovits-url" not in provided_flags and defaults.get("gpt_sovits_url"):
        args.gpt_sovits_url = str(defaults["gpt_sovits_url"])
    if args.gpt_sovits_ref_audio is None and defaults.get("gpt_sovits_ref_audio"):
        args.gpt_sovits_ref_audio = Path(str(defaults["gpt_sovits_ref_audio"]))
    if not args.gpt_sovits_prompt_lang.strip() and defaults.get("gpt_sovits_prompt_lang"):
        args.gpt_sovits_prompt_lang = str(defaults["gpt_sovits_prompt_lang"])
    if not args.gpt_sovits_prompt_text.strip() and defaults.get("gpt_sovits_prompt_text"):
        args.gpt_sovits_prompt_text = str(defaults["gpt_sovits_prompt_text"])
    if "--gpt-sovits-text-lang" not in provided_flags and defaults.get("gpt_sovits_text_lang"):
        args.gpt_sovits_text_lang = str(defaults["gpt_sovits_text_lang"])
    if "--gpt-sovits-text-split-method" not in provided_flags and defaults.get("gpt_sovits_text_split_method"):
        args.gpt_sovits_text_split_method = str(defaults["gpt_sovits_text_split_method"])
    if "--gpt-sovits-speed" not in provided_flags and defaults.get("gpt_sovits_speed") is not None:
        args.gpt_sovits_speed = float(defaults["gpt_sovits_speed"])
    args.gpt_sovits_fallback_female_audio = defaults.get("gpt_sovits_fallback_female_audio")
    args.gpt_sovits_fallback_male_audio = defaults.get("gpt_sovits_fallback_male_audio")
    return args, source_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a video into a target language and rebuild the dubbed video.")
    parser.add_argument("--video", type=Path, default=None, help="Path to the source video file")
    parser.add_argument("--input-dir", type=Path, default=None, help="Process all supported video files in a directory")
    parser.add_argument("--output", type=Path, default=None, help="Path to the translated output video")
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE, help="Target language for translation, e.g. zh-CN")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="faster-whisper model size, e.g. small or medium")
    parser.add_argument(
        "--asr-device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_ASR_DEVICE,
        help="ASR inference device. auto lets faster-whisper choose the best supported backend.",
    )
    parser.add_argument(
        "--asr-compute-type",
        default=DEFAULT_ASR_COMPUTE_TYPE,
        help="ASR compute type passed to faster-whisper, e.g. default, float16, int8.",
    )
    parser.add_argument("--subtitle-source", choices=("auto", "ocr", "asr"), default=DEFAULT_SUBTITLE_SOURCE, help="Subtitle acquisition source: OCR first, ASR only, or auto fallback")
    parser.add_argument("--ocr-fps", type=float, default=DEFAULT_OCR_FPS, help="Frame sampling rate for OCR subtitle extraction")
    parser.add_argument("--ocr-min-chars", type=int, default=DEFAULT_OCR_MIN_CHARS, help="Minimum characters required for an OCR subtitle sample")
    parser.add_argument("--ocr-similarity", type=float, default=DEFAULT_OCR_SIMILARITY, help="Similarity threshold for merging OCR subtitle samples")
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim this many seconds from the start of each input video before processing")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Trim this many seconds from the end of each input video before processing")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS, help="Chunk duration for ASR processing")
    parser.add_argument("--workdir", type=Path, default=None, help="Directory for intermediate files")
    parser.add_argument(
        "--separator",
        choices=("auto", "mdx", "demucs", "ffmpeg", "none"),
        default="auto",
        help="Voice/background separation strategy",
    )
    parser.add_argument("--mdx-model", default=DEFAULT_MDX_MODEL, help="MDX-Net/UVR model filename for offline vocal separation")
    parser.add_argument(
        "--demucs-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=DEFAULT_DEMUCS_DEVICE,
        help="Demucs inference device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--keep-workdir", action="store_true", help="Keep intermediate artifacts")
    parser.add_argument("--dub-volume", type=float, default=0.78, help="Mix weight for synthesized speech")
    parser.add_argument("--background-volume", type=float, default=1.18, help="Mix weight for background audio")
    parser.add_argument("--tts-timeout", type=int, default=DEFAULT_TTS_TIMEOUT, help="Per-request timeout in seconds for GPT-SoVITS")
    parser.add_argument("--tts-retries", type=int, default=DEFAULT_TTS_RETRIES, help="Retry count for GPT-SoVITS request failures")
    parser.add_argument("--translate-retries", type=int, default=DEFAULT_TRANSLATE_RETRIES, help="Retry count for the selected translation backend")
    parser.add_argument("--translate-batch-size", type=int, default=DEFAULT_TRANSLATE_BATCH_SIZE, help="Number of segments to send in each batch translation request")
    parser.add_argument("--translate-timeout", type=int, default=DEFAULT_TRANSLATE_TIMEOUT, help="Reserved for translation request timeout compatibility; local HY-MT ignores it")
    parser.add_argument("--hy-model", default=DEFAULT_HY_MODEL, help="Base HY-MT Hugging Face model id")
    parser.add_argument(
        "--hy-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=DEFAULT_HY_DEVICE,
        help="Runtime device for the local HY-MT translation model.",
    )
    parser.add_argument("--hy-max-new-tokens", type=int, default=DEFAULT_HY_MAX_NEW_TOKENS, help="Maximum number of tokens to generate per HY-MT translation segment")
    parser.add_argument("--gpt-sovits-url", default=DEFAULT_GPT_SOVITS_URL, help="Base URL of the local GPT-SoVITS API server")
    parser.add_argument("--gpt-sovits-ref-audio", type=Path, default=None, help="Reference audio for GPT-SoVITS zero-shot synthesis")
    parser.add_argument("--gpt-sovits-prompt-text", default="", help="Transcript matching the GPT-SoVITS reference audio")
    parser.add_argument("--gpt-sovits-prompt-lang", default="", help="Language of the GPT-SoVITS reference audio, e.g. zh or en")
    parser.add_argument(
        "--gpt-sovits-text-lang",
        default=DEFAULT_GPT_SOVITS_TEXT_LANG,
        help="Target text language for GPT-SoVITS, e.g. auto, zh, en, ja, ko, yue",
    )
    parser.add_argument(
        "--gpt-sovits-text-split-method",
        default=DEFAULT_GPT_SOVITS_TEXT_SPLIT_METHOD,
        help="Text split strategy passed to GPT-SoVITS, e.g. cut5",
    )
    parser.add_argument(
        "--gpt-sovits-speed",
        type=float,
        default=DEFAULT_GPT_SOVITS_SPEED,
        help="Base speaking speed factor passed to GPT-SoVITS",
    )
    raw_args = sys.argv[1:]
    if raw_args and not raw_args[0].startswith("-") and "--video" not in raw_args and "--input-dir" not in raw_args:
        candidate = Path(raw_args[0]).expanduser()
        inferred_flag = "--input-dir" if candidate.is_dir() else "--video"
        raw_args = [inferred_flag, raw_args[0], *raw_args[1:]]

    args = parser.parse_args(raw_args)
    args, runtime_defaults_path = apply_runtime_defaults(args, raw_args)
    args.input_video = args.video
    args.runtime_defaults_path = runtime_defaults_path
    if args.video and args.input_dir:
        parser.error("--video and --input-dir are mutually exclusive")
    if args.input_video is None and args.input_dir is None:
        parser.error("one of the following arguments is required: --video, --input-dir")
    if args.trim_start < 0:
        parser.error("--trim-start must be >= 0")
    if args.trim_end < 0:
        parser.error("--trim-end must be >= 0")
    args.translation_provider = DEFAULT_TRANSLATION_PROVIDER
    args.tts_provider = "gpt-sovits"
    if args.gpt_sovits_ref_mode == "manual" and args.gpt_sovits_ref_audio is None:
        parser.error("missing GPT-SoVITS defaults: run ./install.sh and ./init.sh once, or pass --gpt-sovits-ref-audio explicitly")
    if args.gpt_sovits_ref_audio is not None:
        args.gpt_sovits_ref_audio = args.gpt_sovits_ref_audio.expanduser()
    if args.gpt_sovits_ref_mode == "manual" and not args.gpt_sovits_ref_audio.is_file():
        parser.error(f"GPT-SoVITS reference audio not found: {args.gpt_sovits_ref_audio}")
    if args.gpt_sovits_ref_mode == "manual" and not args.gpt_sovits_prompt_lang.strip():
        parser.error("missing GPT-SoVITS prompt language: run ./init.sh once, or pass --gpt-sovits-prompt-lang explicitly")
    try:
        if args.gpt_sovits_ref_mode == "manual":
            normalize_gpt_sovits_language(args.gpt_sovits_prompt_lang)
        if args.gpt_sovits_text_lang != "auto":
            normalize_gpt_sovits_language(args.gpt_sovits_text_lang)
        else:
            normalize_gpt_sovits_language(args.target_language)
    except ValueError as exc:
        parser.error(str(exc))
    if args.input_dir is not None and args.output is not None and args.output.suffix:
        parser.error("--output must be a directory path when used with --input-dir")
    return args


def run_command(
    command: list[str],
    *,
    check: bool = True,
    heartbeat_message: str | None = None,
    heartbeat_interval: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    if heartbeat_message is None:
        result = subprocess.run(command, text=True, capture_output=True)
    else:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            try:
                stdout, stderr = process.communicate(timeout=heartbeat_interval)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                break
            except subprocess.TimeoutExpired:
                print(f"[info] {heartbeat_message}")
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(command)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def gpt_sovits_service_available(api_url: str, timeout_seconds: float = 3.0) -> bool:
    with contextlib.suppress(requests.RequestException):
        requests.get(api_url.rstrip("/") + "/", timeout=timeout_seconds)
        return True
    return False


def load_gpt_sovits_service_config() -> tuple[dict[str, object] | None, Path | None]:
    for config_path in gpt_sovits_service_config_paths():
        if not config_path.is_file():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"GPT-SoVITS service config must contain a JSON object: {config_path}")
        return payload, config_path
    return None, None


def start_gpt_sovits_service_from_config() -> bool:
    payload, config_path = load_gpt_sovits_service_config()
    if payload is None or config_path is None:
        return False

    required_keys = ("python_bin", "api_script", "config_path", "gpt_sovits_root", "host", "port", "log_path", "pid_path")
    missing = [key for key in required_keys if not payload.get(key)]
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(f"GPT-SoVITS service config is missing required keys ({joined}): {config_path}")

    python_bin = Path(str(payload["python_bin"])).expanduser()
    api_script = Path(str(payload["api_script"])).expanduser()
    runtime_config = Path(str(payload["config_path"])).expanduser()
    gpt_root = Path(str(payload["gpt_sovits_root"])).expanduser()
    log_path = Path(str(payload["log_path"])).expanduser()
    pid_path = Path(str(payload["pid_path"])).expanduser()
    host = str(payload["host"])
    port = str(payload["port"])

    for required_path in (python_bin, api_script, runtime_config):
        if not required_path.is_file():
            raise RuntimeError(f"GPT-SoVITS startup dependency is missing: {required_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath_entries = [str(gpt_root), str(gpt_root / "GPT_SoVITS")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [str(python_bin), str(api_script), "-a", host, "-p", port, "-c", str(runtime_config)],
            cwd=str(gpt_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid), encoding="utf-8")
    return True


def ensure_gpt_sovits_service(api_url: str) -> None:
    if gpt_sovits_service_available(api_url):
        return

    started = start_gpt_sovits_service_from_config()
    if not started:
        if not GPT_SOVITS_START_SCRIPT.is_file():
            raise RuntimeError(
                "GPT-SoVITS service is not reachable and no local service config was found. Run ./init.sh first."
            )

        result = run_command([str(GPT_SOVITS_START_SCRIPT)], check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to start GPT-SoVITS service automatically.\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

    for _ in range(30):
        if gpt_sovits_service_available(api_url):
            return
        time.sleep(2)

    raise RuntimeError("GPT-SoVITS service did not become ready in time. Run ./init.sh or inspect start-gpt-sovits.sh logs.")


def require_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FileNotFoundError(f"Required binary not found in PATH: {binary}")


def ffprobe_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def trim_video(input_video: Path, output_video: Path, trim_start: float, trim_end: float) -> float:
    source_duration = ffprobe_duration(input_video)
    safe_trim_start = max(trim_start, 0.0)
    safe_trim_end = max(trim_end, 0.0)
    trimmed_duration = source_duration - safe_trim_start - safe_trim_end
    if trimmed_duration <= 0.0:
        raise ValueError(
            f"Invalid trim window for {input_video.name}: start={safe_trim_start:.3f}s end={safe_trim_end:.3f}s exceeds duration={source_duration:.3f}s"
        )

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-ss",
            f"{safe_trim_start:.3f}",
            "-t",
            f"{trimmed_duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    return trimmed_duration


def prepare_input_video(input_video: Path, workdir: Path, trim_start: float, trim_end: float) -> tuple[Path, float, float]:
    original_duration = ffprobe_duration(input_video)
    if trim_start <= 0.0 and trim_end <= 0.0:
        return input_video, original_duration, original_duration

    prepared_video = workdir / "trimmed_input.mp4"
    trimmed_duration = trim_video(input_video, prepared_video, trim_start, trim_end)
    return prepared_video, original_duration, trimmed_duration


def probe_duration_or_none(path: Path) -> float | None:
    if not path.exists():
        return None
    with contextlib.suppress(Exception):
        return ffprobe_duration(path)
    return None


@dataclass(frozen=True, slots=True)
class AudioLevelStats:
    mean_volume_db: float | None
    max_volume_db: float | None


def probe_audio_levels(path: Path) -> AudioLevelStats:
    if not path.exists():
        return AudioLevelStats(mean_volume_db=None, max_volume_db=None)
    result = run_command(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"

    def extract_volume(name: str) -> float | None:
        match = re.search(rf"{name}:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB", output)
        if not match:
            return None
        value = match.group(1)
        if value == "inf":
            return float("inf")
        return float(value)

    return AudioLevelStats(
        mean_volume_db=extract_volume("mean_volume"),
        max_volume_db=extract_volume("max_volume"),
    )


def format_db_for_log(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf dB"
    return f"{value:.1f} dB"


def segment_audio_issue(levels: AudioLevelStats) -> str | None:
    if levels.max_volume_db is None:
        return "missing-levels"
    if levels.max_volume_db <= -30.0:
        return "very-quiet"
    if levels.mean_volume_db is not None and levels.mean_volume_db <= -48.0:
        return "mostly-silent"
    return None


def choose_better_audio_attempt(
    current: tuple[Path, AudioLevelStats, float],
    candidate: tuple[Path, AudioLevelStats, float],
) -> tuple[Path, AudioLevelStats, float]:
    current_issue = segment_audio_issue(current[1])
    candidate_issue = segment_audio_issue(candidate[1])
    if current_issue and not candidate_issue:
        return candidate
    if candidate_issue and not current_issue:
        return current
    current_max = current[1].max_volume_db
    candidate_max = candidate[1].max_volume_db
    if current_max is None:
        return candidate
    if candidate_max is None:
        return current
    if candidate_max > current_max:
        return candidate
    return current


def sanitize_log_text(value: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def format_duration_for_log(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def write_audio_audit_header(
    log_path: Path,
    *,
    tts_provider: str,
    total_duration: float,
    gpt_sovits_config: GptSovitsConfig | None,
    reference_source: str | None = None,
    reference_detail: str | None = None,
) -> None:
    lines = [
        "# Audio Audit Log",
        f"tts_provider: {tts_provider}",
        f"target_total_duration: {total_duration:.3f}s",
    ]
    if gpt_sovits_config is not None:
        lines.extend(
            [
                f"gpt_sovits_api_url: {gpt_sovits_config.api_url}",
                f"gpt_sovits_ref_audio: {gpt_sovits_config.ref_audio_path}",
                f"gpt_sovits_ref_source: {reference_source or 'n/a'}",
                f"gpt_sovits_ref_detail: {reference_detail or 'n/a'}",
                f"gpt_sovits_prompt_lang: {gpt_sovits_config.prompt_lang}",
                f"gpt_sovits_text_lang: {gpt_sovits_config.text_lang}",
                f"gpt_sovits_speed_factor: {gpt_sovits_config.speed_factor}",
            ]
        )
    lines.extend(["", "## Segment Audit"])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_audio_audit_line(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_segment_audit_line(
    log_path: Path,
    *,
    index: int,
    backend: str,
    original_start: float,
    original_end: float,
    scheduled_start: float,
    scheduled_end: float,
    translated_text: str,
    raw_tts: Path,
    adjusted_tts: Path | None,
    raw_duration: float | None,
    adjusted_duration: float | None,
    speed_factor: float | None,
    raw_levels: AudioLevelStats | None,
    adjusted_levels: AudioLevelStats | None,
    status: str,
    quiet_retry_attempts: int = 0,
    error: str | None = None,
) -> None:
    line = (
        f"segment={index:04d} status={status} backend={backend} "
        f"slot={original_start:.3f}-{original_end:.3f}s scheduled={scheduled_start:.3f}-{scheduled_end:.3f}s "
        f"raw_exists={raw_tts.exists()} raw_duration={format_duration_for_log(raw_duration)} raw_path={raw_tts}"
    )
    if adjusted_tts is not None:
        line += (
            f" adjusted_exists={adjusted_tts.exists()} adjusted_duration={format_duration_for_log(adjusted_duration)}"
            f" adjusted_path={adjusted_tts}"
        )
    if raw_levels is not None:
        line += (
            f" raw_mean_volume={format_db_for_log(raw_levels.mean_volume_db)}"
            f" raw_max_volume={format_db_for_log(raw_levels.max_volume_db)}"
        )
    if adjusted_levels is not None:
        line += (
            f" adjusted_mean_volume={format_db_for_log(adjusted_levels.mean_volume_db)}"
            f" adjusted_max_volume={format_db_for_log(adjusted_levels.max_volume_db)}"
        )
    if speed_factor is not None:
        line += f" speed_factor={speed_factor:.3f}"
    issue = segment_audio_issue(raw_levels) if raw_levels is not None else None
    if issue:
        line += f" issue={issue}"
    line += f" quiet_retry_attempts={quiet_retry_attempts}"
    line += f" text=\"{sanitize_log_text(translated_text)}\""
    if error:
        line += f" error=\"{sanitize_log_text(error, limit=240)}\""
    append_audio_audit_line(log_path, line)


def synthesize_segment_with_quiet_retry(
    *,
    text: str,
    raw_tts: Path,
    timeout_seconds: int,
    retries: int,
    gpt_sovits_config: GptSovitsConfig | None,
    audit_log: Path,
    segment_index: int,
) -> tuple[str, float, AudioLevelStats, int]:
    best_attempt: tuple[Path, AudioLevelStats, float] | None = None
    backend = "gpt-sovits"
    quiet_retry_attempts = 0

    for attempt_index in range(QUIET_TTS_RETRIES + 1):
        attempt_path = raw_tts.with_name(f"{raw_tts.stem}_attempt{attempt_index + 1}{raw_tts.suffix}")
        backend = synthesize_segment_audio(
            text,
            attempt_path,
            timeout_seconds,
            retries,
            gpt_sovits_config,
        )
        raw_duration = ffprobe_duration(attempt_path)
        raw_levels = probe_audio_levels(attempt_path)
        issue = segment_audio_issue(raw_levels)
        attempt_record = (attempt_path, raw_levels, raw_duration)
        best_attempt = attempt_record if best_attempt is None else choose_better_audio_attempt(best_attempt, attempt_record)
        append_audio_audit_line(
            audit_log,
            (
                f"segment={segment_index:04d} retry_attempt={attempt_index + 1} "
                f"status={'accepted' if issue is None else 'suspicious'} raw_duration={format_duration_for_log(raw_duration)} "
                f"raw_mean_volume={format_db_for_log(raw_levels.mean_volume_db)} "
                f"raw_max_volume={format_db_for_log(raw_levels.max_volume_db)}"
                + (f" issue={issue}" if issue else "")
            ),
        )
        if issue not in QUIET_TTS_RETRY_ISSUES:
            quiet_retry_attempts = attempt_index
            shutil.copyfile(attempt_path, raw_tts)
            return backend, raw_duration, raw_levels, quiet_retry_attempts
        if attempt_index < QUIET_TTS_RETRIES:
            quiet_retry_attempts = attempt_index + 1

    if best_attempt is None:
        raise RuntimeError("TTS attempt selection failed without producing audio")

    shutil.copyfile(best_attempt[0], raw_tts)
    return backend, best_attempt[2], best_attempt[1], quiet_retry_attempts


def append_mix_audit_section(
    log_path: Path,
    *,
    background_track: Path,
    dub_track: Path,
    output_video: Path,
    background_volume: float,
    dub_volume: float,
) -> None:
    append_audio_audit_line(log_path, "")
    append_audio_audit_line(log_path, "## Mix Audit")
    append_audio_audit_line(
        log_path,
        (
            f"background_track_exists={background_track.exists()} "
            f"background_track_duration={format_duration_for_log(probe_duration_or_none(background_track))} "
            f"background_track_path={background_track} background_volume={background_volume}"
        ),
    )
    append_audio_audit_line(
        log_path,
        (
            f"dub_track_exists={dub_track.exists()} "
            f"dub_track_duration={format_duration_for_log(probe_duration_or_none(dub_track))} "
            f"dub_track_path={dub_track} dub_volume={dub_volume}"
        ),
    )
    append_audio_audit_line(
        log_path,
        (
            f"output_video_exists={output_video.exists()} "
            f"output_video_duration={format_duration_for_log(probe_duration_or_none(output_video))} "
            f"output_video_path={output_video}"
        ),
    )


def ffprobe_streams(path: Path, stream_selector: str) -> list[dict[str, object]]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            stream_selector,
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams")
    return streams if isinstance(streams, list) else []


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_cache_root() -> Path:
    return ensure_dir(DEFAULT_RUNTIME_CACHE_DIR)


def runtime_hf_cache_dir() -> Path:
    return ensure_dir(runtime_cache_root() / "huggingface")


def runtime_whisper_cache_dir() -> Path:
    return ensure_dir(runtime_cache_root() / "faster-whisper")


def runtime_mdx_model_dir() -> Path:
    return ensure_dir(runtime_cache_root() / "mdx")


def runtime_demucs_model_dir() -> Path:
    return ensure_dir(runtime_cache_root() / "demucs")


def log_progress(stage: str, current: int, total: int, *, detail: str = "") -> None:
    if total <= 0:
        return
    if total <= 20 or current == 1 or current == total or current % 10 == 0:
        suffix = f" - {detail}" if detail else ""
        print(f"[info] {stage}: {current}/{total}{suffix}")


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, seconds, milliseconds = re.split(r":|,", value.strip())
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def parse_srt_blocks(srt_path: Path) -> list[SrtBlock]:
    raw = srt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    blocks: list[SrtBlock] = []
    for chunk in re.split(r"\n\s*\n", raw):
        lines = [line.strip("\ufeff") for line in chunk.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        timestamp_line = lines[1]
        if "-->" not in timestamp_line:
            continue
        start_raw, end_raw = [part.strip() for part in timestamp_line.split("-->", maxsplit=1)]
        text = normalize_segment_text(" ".join(lines[2:]))
        if not text:
            continue
        blocks.append(SrtBlock(start=parse_srt_timestamp(start_raw), end=parse_srt_timestamp(end_raw), text=text))
    return blocks


def subtitle_stream_language(stream: dict[str, object]) -> str:
    tags = stream.get("tags")
    if isinstance(tags, dict):
        language = tags.get("language") or tags.get("LANGUAGE")
        if isinstance(language, str) and language.strip():
            return language.strip().lower()
    return "unknown"


def detect_subtitle_language(texts: list[str], fallback_language: str) -> str:
    joined = " ".join(texts)
    if contains_cjk(joined):
        return "zh"
    if fallback_language in {"zh", "chi", "zho", "zh-cn", "zh-tw"}:
        return "zh"
    if fallback_language in {"en", "eng"}:
        return "en"
    return fallback_language or "unknown"


def acquire_subtitle_track_segments(input_video: Path, workdir: Path) -> tuple[list[Segment], dict[str, object]]:
    subtitle_streams = ffprobe_streams(input_video, "s")
    if not subtitle_streams:
        return [], {
            "source_name": "subtitle-track",
            "subtitle_language": "none",
            "needs_translation": True,
            "raw_source_srt": None,
            "raw_secondary_srt": None,
        }

    supported_codecs = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}
    chosen_stream: dict[str, object] | None = None
    for stream in subtitle_streams:
        codec_name = str(stream.get("codec_name") or "").lower()
        if codec_name in supported_codecs:
            chosen_stream = stream
            break
    if chosen_stream is None:
        return [], {
            "source_name": "subtitle-track",
            "subtitle_language": "none",
            "needs_translation": True,
            "raw_source_srt": None,
            "raw_secondary_srt": None,
        }

    stream_index = chosen_stream.get("index")
    if not isinstance(stream_index, int):
        return [], {
            "source_name": "subtitle-track",
            "subtitle_language": "none",
            "needs_translation": True,
            "raw_source_srt": None,
            "raw_secondary_srt": None,
        }

    subtitle_dir = ensure_dir(workdir / "subtitle_track")
    extracted_srt = subtitle_dir / "embedded_subtitles.srt"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-map",
            f"0:{stream_index}",
            str(extracted_srt),
        ]
    )

    blocks = parse_srt_blocks(extracted_srt)
    if not blocks:
        return [], {
            "source_name": "subtitle-track",
            "subtitle_language": "none",
            "needs_translation": True,
            "raw_source_srt": extracted_srt,
            "raw_secondary_srt": None,
        }

    texts = [block.text for block in blocks]
    subtitle_language = detect_subtitle_language(texts, subtitle_stream_language(chosen_stream))
    segments = [
        Segment(
            start=block.start,
            end=block.end,
            source_text=block.text,
            translated_text=block.text if contains_cjk(block.text) else "",
        )
        for block in blocks
    ]
    return segments, {
        "source_name": "subtitle-track",
        "subtitle_language": subtitle_language,
        "needs_translation": any(not segment.translated_text for segment in segments),
        "raw_source_srt": extracted_srt,
        "raw_secondary_srt": None,
    }


def load_whisper_model_class():
    from faster_whisper import WhisperModel

    return WhisperModel


def load_ocr_helpers():
    from ocr_subtitles import extract_subtitle_blocks, write_debug_json, write_srt

    return extract_subtitle_blocks, write_debug_json, write_srt


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_demucs_device(requested_device: str) -> str:
    if requested_device == "cuda":
        if not torch_cuda_available():
            raise RuntimeError("Demucs device 'cuda' was requested, but CUDA is not available in the current PyTorch runtime")
        return requested_device
    if requested_device == "mps":
        if not torch_mps_available():
            raise RuntimeError("Demucs device 'mps' was requested, but Apple MPS is not available in the current PyTorch runtime")
        return requested_device
    if requested_device != "auto":
        return requested_device
    if torch_cuda_available():
        return "cuda"
    if torch_mps_available():
        return "mps"
    return "cpu"


def resolve_asr_runtime(requested_device: str, requested_compute_type: str) -> tuple[str, str]:
    import ctranslate2

    if requested_device == "auto":
        try:
            supported_compute_types = ctranslate2.get_supported_compute_types("cuda")
            device = "cuda"
        except ValueError:
            supported_compute_types = ctranslate2.get_supported_compute_types("cpu")
            device = "cpu"
    elif requested_device == "cuda":
        device = "cuda"
        try:
            supported_compute_types = ctranslate2.get_supported_compute_types("cuda")
        except ValueError as exc:
            raise RuntimeError("ASR device 'cuda' was requested, but faster-whisper/CTranslate2 was installed without CUDA support") from exc
    else:
        device = requested_device
        supported_compute_types = ctranslate2.get_supported_compute_types(device)

    compute_type = requested_compute_type
    if compute_type == "default":
        if device == "cuda":
            if "float16" in supported_compute_types:
                compute_type = "float16"
            elif "int8_float16" in supported_compute_types:
                compute_type = "int8_float16"
        elif device == "cpu" and "int8" in supported_compute_types:
            compute_type = "int8"
    elif compute_type not in supported_compute_types:
        supported_list = ", ".join(sorted(supported_compute_types))
        raise RuntimeError(
            f"ASR compute type '{compute_type}' is not supported for device '{device}'. Supported values: {supported_list}"
        )
    return device, compute_type


def extract_audio(input_video: Path, workdir: Path) -> tuple[Path, Path]:
    full_mix = workdir / "full_mix.wav"
    speech_mix = workdir / "speech_input.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(full_mix),
        ]
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(speech_mix),
        ]
    )
    return full_mix, speech_mix


def separate_with_demucs(speech_mix: Path, full_mix: Path, workdir: Path, demucs_device: str) -> tuple[Path, Path]:
    demucs_output = workdir / "demucs"
    demucs_repo = runtime_demucs_model_dir()
    print(
        "[info] Demucs separation started "
        f"(device={demucs_device}). This step can take several minutes on long videos or on the first run."
    )
    run_command(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs_ft",
            "--repo",
            str(demucs_repo),
            "--two-stems=vocals",
            "-d",
            demucs_device,
            "-o",
            str(demucs_output),
            str(full_mix),
        ],
        heartbeat_message="Demucs is still separating vocals and background...",
    )
    separated_dir = demucs_output / "htdemucs_ft" / full_mix.stem
    vocals = separated_dir / "vocals.wav"
    background = separated_dir / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        raise FileNotFoundError("Demucs did not produce expected vocals/background stems")

    speech_vocals = workdir / "vocals_16k.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vocals),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(speech_vocals),
        ]
    )
    return speech_vocals, background


def separate_with_mdx(speech_mix: Path, full_mix: Path, workdir: Path, model_filename: str) -> tuple[Path, Path]:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "MDX-Net separation requires audio-separator. Install it with: pip install 'audio-separator[cpu]'"
        ) from exc

    mdx_output = ensure_dir(workdir / "mdx")
    print(f"[info] MDX-Net separation started (model={model_filename}). This step may download the model on first run.")
    separator = Separator(
        output_dir=str(mdx_output),
        output_format="WAV",
        model_file_dir=str(runtime_mdx_model_dir()),
        sample_rate=DUB_SAMPLE_RATE,
        use_soundfile=True,
    )
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(
        str(full_mix),
        {
            "Vocals": "vocals",
            "Instrumental": "instrumental",
        },
    )

    vocals = mdx_output / "vocals.wav"
    background = mdx_output / "instrumental.wav"
    if not vocals.exists() or not background.exists():
        produced = ", ".join(str(Path(path).name) for path in output_files)
        raise FileNotFoundError(f"MDX-Net did not produce expected stems. Output files: {produced}")

    speech_vocals = workdir / "vocals_16k.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vocals),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(speech_vocals),
        ]
    )
    return speech_vocals, background


def separate_with_ffmpeg(speech_mix: Path, full_mix: Path, workdir: Path) -> tuple[Path, Path]:
    vocals = workdir / "vocals_16k.wav"
    background = workdir / "background.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(speech_mix),
            "-af",
            "highpass=f=120,lowpass=f=4800,afftdn=nf=-25,dynaudnorm",
            str(vocals),
        ]
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(full_mix),
            "-af",
            "volume=0.45",
            str(background),
        ]
    )
    return vocals, background


def separate_audio(
    strategy: str,
    speech_mix: Path,
    full_mix: Path,
    workdir: Path,
    demucs_device: str,
    mdx_model: str,
) -> tuple[Path, Path, str]:
    if strategy == "none":
        return speech_mix, full_mix, "none"
    if strategy == "ffmpeg":
        vocals, background = separate_with_ffmpeg(speech_mix, full_mix, workdir)
        return vocals, background, "ffmpeg"
    if strategy == "mdx":
        vocals, background = separate_with_mdx(speech_mix, full_mix, workdir, mdx_model)
        return vocals, background, "mdx"
    if strategy == "demucs":
        vocals, background = separate_with_demucs(speech_mix, full_mix, workdir, demucs_device)
        return vocals, background, "demucs"

    try:
        vocals, background = separate_with_mdx(speech_mix, full_mix, workdir, mdx_model)
        return vocals, background, "mdx"
    except Exception as exc:
        print(f"[warn] MDX-Net unavailable, falling back to Demucs separation: {exc}")

    try:
        vocals, background = separate_with_demucs(speech_mix, full_mix, workdir, demucs_device)
        return vocals, background, "demucs"
    except Exception as exc:
        print(f"[warn] Demucs unavailable, falling back to ffmpeg separation: {exc}")
        vocals, background = separate_with_ffmpeg(speech_mix, full_mix, workdir)
        return vocals, background, "ffmpeg"


def iter_chunk_ranges(total_duration: float, chunk_seconds: int) -> Iterable[tuple[float, float]]:
    start = 0.0
    while start < total_duration:
        end = min(total_duration, start + chunk_seconds)
        yield start, end
        start = end


def transcribe_audio(
    vocals_path: Path,
    model_name: str,
    chunk_seconds: int,
    asr_device: str,
    asr_compute_type: str,
) -> list[Segment]:
    print(f"[info] Loading ASR model '{model_name}' (device={asr_device}, compute_type={asr_compute_type})")
    model = load_whisper_model_class()(
        model_name,
        device=asr_device,
        compute_type=asr_compute_type,
        download_root=str(runtime_whisper_cache_dir()),
    )
    duration = ffprobe_duration(vocals_path)
    segments: list[Segment] = []
    chunk_dir = ensure_dir(vocals_path.parent / "chunks")
    chunk_ranges = list(iter_chunk_ranges(duration, chunk_seconds))

    for index, (start, end) in enumerate(chunk_ranges, start=1):
        log_progress("ASR chunk", index, len(chunk_ranges), detail=f"{start:.0f}s-{end:.0f}s")
        chunk_path = chunk_dir / f"chunk_{index:04d}.wav"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{end - start:.3f}",
                "-i",
                str(vocals_path),
                str(chunk_path),
            ]
        )
        detected_segments, _info = model.transcribe(
            str(chunk_path),
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        for piece in detected_segments:
            text = piece.text.strip()
            if not text:
                continue
            absolute_start = start + max(0.0, piece.start)
            absolute_end = start + max(piece.end, piece.start + 0.2)
            segments.append(Segment(start=absolute_start, end=absolute_end, source_text=text))

    if not segments:
        raise RuntimeError("No speech segments were detected in the source audio")
    return merge_close_segments(segments)


def ends_sentence(text: str) -> bool:
    normalized = text.rstrip()
    return normalized.endswith(SENTENCE_ENDINGS)


def merge_close_segments(
    segments: list[Segment],
    max_gap: float = 0.45,
    max_chars: int = 120,
    max_duration: float = 6.5,
    min_chars: int = 18,
    min_duration: float = 1.5,
) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap = max(0.0, segment.start - previous.end)
        combined_text = f"{previous.source_text} {segment.source_text}".strip()
        combined_duration = max(segment.end, previous.end) - previous.start
        previous_is_short = len(previous.source_text) < min_chars or previous.duration < min_duration
        should_merge = (
            gap <= max_gap
            and len(combined_text) <= max_chars
            and combined_duration <= max_duration
            and (not ends_sentence(previous.source_text) or previous_is_short)
        )
        if should_merge:
            previous.end = max(previous.end, segment.end)
            previous.source_text = combined_text
            continue
        merged.append(segment)
    return merged


def normalize_hy_language_code(target_language: str) -> str:
    normalized = normalize_translate_code(target_language)
    mapping = {
        "zh-CN": "zh",
        "zh-cn": "zh",
        "zh-TW": "zh-Hant",
        "zh-tw": "zh-Hant",
    }
    return mapping.get(normalized, normalized)


def hy_target_language_name(target_language: str) -> str:
    normalized = normalize_hy_language_code(target_language)
    return HY_MT_LANGUAGE_NAMES.get(normalized, normalized)


def resolve_hy_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_hy_prompt(source_text: str, target_language: str) -> str:
    target_name = hy_target_language_name(target_language)
    normalized_target = normalize_hy_language_code(target_language)
    if normalized_target.startswith("zh"):
        return f"将以下文本翻译为{target_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
    return f"Translate the following segment into {target_name}, without additional explanation.\n\n{source_text}"


def clean_hy_translation(text: str) -> str:
    normalized = text.strip()
    target_match = re.search(r"<target>(.*?)</target>", normalized, flags=re.DOTALL)
    if target_match:
        normalized = target_match.group(1).strip()
    normalized = re.sub(r"^(assistant|译文)\s*[:：]\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def resolve_hy_model_candidates(base_model: str, resolved_device: str) -> list[str]:
    if resolved_device == "cuda":
        return [DEFAULT_HY_GPTQ_MODEL, DEFAULT_HY_FP8_MODEL, base_model]
    return [base_model]


@functools.lru_cache(maxsize=8)
def load_hy_runtime(base_model: str, requested_device: str) -> HyMtRuntime:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "HY-MT local translation requires transformers. Install it with: pip install transformers==4.56.0"
        ) from exc

    resolved_device = resolve_hy_device(requested_device)
    candidates = resolve_hy_model_candidates(base_model, resolved_device)
    errors: list[str] = []

    for model_id in candidates:
        try:
            if model_id == DEFAULT_HY_GPTQ_MODEL and resolved_device != "cuda":
                raise RuntimeError("GPTQ Int4 HY-MT is only attempted on CUDA runtimes")
            if model_id == DEFAULT_HY_FP8_MODEL:
                raise RuntimeError("FP8 HY-MT requires extra compressed-tensors setup and is skipped by default")

            print(f"[info] Loading translation model: {model_id} (device={resolved_device})")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(runtime_hf_cache_dir()))
            dtype = torch.float32 if resolved_device == "cpu" else torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, cache_dir=str(runtime_hf_cache_dir()))
            model.eval()
            if resolved_device != "cpu":
                model.to(resolved_device)
            return HyMtRuntime(tokenizer=tokenizer, model=model, device=resolved_device, model_id=model_id)
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    joined_errors = " | ".join(errors)
    raise RuntimeError(f"Failed to load HY-MT translation model. {joined_errors}")


class HyLocalTranslator:
    def __init__(self, target_language: str, model_id: str, requested_device: str, max_new_tokens: int):
        self.target_language = target_language
        self.model_id = model_id
        self.requested_device = requested_device
        self.max_new_tokens = max(32, max_new_tokens)

    def translate_batch(self, batch: list[str]) -> list[str]:
        if not batch:
            return []

        import torch

        runtime = load_hy_runtime(self.model_id, self.requested_device)
        results: list[str] = []
        for text in batch:
            prompt = build_hy_prompt(text, self.target_language)
            messages = [{"role": "user", "content": prompt}]
            if hasattr(runtime.tokenizer, "apply_chat_template"):
                input_ids = runtime.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                input_ids = runtime.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(runtime.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = runtime.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.6,
                    repetition_penalty=1.05,
                    pad_token_id=getattr(runtime.tokenizer, "pad_token_id", None) or getattr(runtime.tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(runtime.tokenizer, "eos_token_id", None),
                )
            generated_ids = outputs[:, input_ids.shape[-1] :]
            translated = runtime.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            cleaned = clean_hy_translation(translated)
            if not cleaned:
                raise RuntimeError(f"HY-MT returned empty text for source segment: {text!r}")
            results.append(cleaned)
        return results

    def resolved_model_id(self) -> str | None:
        try:
            return load_hy_runtime(self.model_id, self.requested_device).model_id
        except Exception:
            return None

    def resolved_device(self) -> str | None:
        try:
            return load_hy_runtime(self.model_id, self.requested_device).device
        except Exception:
            return None


def translation_backend_metadata(backend: TranslationBackend, *, resolve_runtime: bool = False) -> dict[str, str | None]:
    metadata = {
        "resolved_translation_provider": backend.name,
        "translation_model": None,
        "resolved_translation_device": None,
    }
    if isinstance(backend.translator, HyLocalTranslator):
        metadata["translation_model"] = backend.translator.model_id
        metadata["resolved_translation_device"] = resolve_hy_device(backend.translator.requested_device)
        if resolve_runtime:
            metadata["translation_model"] = backend.translator.resolved_model_id() or metadata["translation_model"]
            metadata["resolved_translation_device"] = backend.translator.resolved_device() or metadata["resolved_translation_device"]
    return metadata


def build_translation_backend(
    target_language: str,
    provider: str,
    hy_model: str,
    hy_device: str,
    hy_max_new_tokens: int,
) -> TranslationBackend:
    _ = provider
    _ = normalize_translate_code(target_language)
    return TranslationBackend(
        name="hy-local",
        translator=HyLocalTranslator(target_language, hy_model, hy_device, hy_max_new_tokens),
        uses_network_timeout=False,
    )


@contextlib.contextmanager
def requests_timeout(timeout_seconds: int):
    original_request = requests.sessions.Session.request
    original_socket_timeout = socket.getdefaulttimeout()

    def request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original_request(self, method, url, **kwargs)

    socket.setdefaulttimeout(timeout_seconds)
    setattr(requests.sessions.Session, "request", request_with_timeout)
    try:
        yield
    finally:
        socket.setdefaulttimeout(original_socket_timeout)
        setattr(requests.sessions.Session, "request", original_request)


def run_with_timeout(timeout_seconds: int, operation):
    if timeout_seconds <= 0:
        return operation()

    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def translate_batch_with_backend(texts: list[str], backend: TranslationBackend, retries: int, timeout_seconds: int) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if backend.uses_network_timeout:
                with requests_timeout(timeout_seconds):
                    translated = run_with_timeout(timeout_seconds, lambda: backend.translator.translate_batch(texts))
            else:
                translated = backend.translator.translate_batch(texts)
            if len(translated) != len(texts):
                raise RuntimeError(f"{backend.name} returned {len(translated)} items for {len(texts)} inputs")
            normalized = [item.strip() if item else "" for item in translated]
            if any(not item for item in normalized):
                raise RuntimeError(f"{backend.name} returned empty text inside a batch")
            return normalized
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(attempt, 3))
    raise RuntimeError(f"Failed to translate batch with provider={backend.name}. Last error: {last_error}")


def translate_segments(segments: list[Segment], backend: TranslationBackend, retries: int, batch_size: int, timeout_seconds: int) -> None:
    safe_batch_size = max(1, batch_size)
    total_batches = (len(segments) + safe_batch_size - 1) // safe_batch_size
    for batch_index, batch_start in enumerate(range(0, len(segments), safe_batch_size), start=1):
        log_progress("Translation batch", batch_index, total_batches)
        batch = segments[batch_start : batch_start + safe_batch_size]
        translated_texts = translate_batch_with_backend([segment.source_text for segment in batch], backend, retries, timeout_seconds)
        for segment, translated_text in zip(batch, translated_texts, strict=True):
            segment.translated_text = translated_text


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[Segment], output_path: Path, field_name: str) -> Path:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = getattr(segment, field_name).strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def normalize_segment_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def build_segments_from_subtitle_blocks(blocks: list[SubtitleBlock], prefer_chinese: bool) -> list[Segment]:
    segments: list[Segment] = []
    for block in blocks:
        if prefer_chinese and block.chinese_text:
            chosen_text = block.chinese_text
        else:
            chosen_text = block.english_text or block.chinese_text
        chosen_text = normalize_segment_text(chosen_text)
        if not chosen_text:
            continue
        translated_text = chosen_text if contains_cjk(chosen_text) else ""
        segments.append(Segment(start=block.start, end=block.end, source_text=chosen_text, translated_text=translated_text))
    return segments


def acquire_ocr_segments(input_video: Path, workdir: Path, fps: float, min_chars: int, similarity: float) -> tuple[list[Segment], dict[str, object]]:
    extract_subtitle_blocks, write_ocr_debug_json, write_ocr_srt = load_ocr_helpers()
    ocr_workdir = ensure_dir(workdir / "ocr")
    samples, blocks = extract_subtitle_blocks(
        input_video=input_video,
        workdir=ocr_workdir,
        fps=fps,
        min_chars=min_chars,
        similarity=similarity,
        engine_name="rapidocr",
    )
    if not blocks:
        return [], {
            "source_name": "ocr",
            "subtitle_language": "none",
            "needs_translation": True,
            "raw_source_srt": None,
            "raw_secondary_srt": None,
        }

    has_chinese = any(block.chinese_text for block in blocks)
    subtitle_language = "zh" if has_chinese else "en"
    prefer_chinese = has_chinese
    segments = build_segments_from_subtitle_blocks(blocks, prefer_chinese=prefer_chinese)
    source_language = "chinese" if prefer_chinese else "english"
    secondary_language = "english" if prefer_chinese else "chinese"
    source_srt = write_ocr_srt(blocks, ocr_workdir / f"ocr_{source_language}.srt", source_language)
    secondary_srt = write_ocr_srt(blocks, ocr_workdir / f"ocr_{secondary_language}.srt", secondary_language) if any((block.english_text if prefer_chinese else block.chinese_text) for block in blocks) else None
    debug_json = write_ocr_debug_json(samples, blocks, ocr_workdir / "ocr_debug.json")
    return segments, {
        "source_name": "ocr",
        "subtitle_language": subtitle_language,
        "needs_translation": any(not segment.translated_text for segment in segments),
        "raw_source_srt": source_srt,
        "raw_secondary_srt": secondary_srt,
        "ocr_debug_json": debug_json,
        "ocr_block_count": len(blocks),
    }


def acquire_segments(
    input_video: Path,
    vocals_path: Path,
    workdir: Path,
    subtitle_source: str,
    asr_model: str,
    asr_device: str,
    asr_compute_type: str,
    chunk_seconds: int,
    ocr_fps: float,
    ocr_min_chars: int,
    ocr_similarity: float,
) -> tuple[list[Segment], dict[str, object]]:
    if subtitle_source == "auto":
        segments, metadata = acquire_subtitle_track_segments(input_video, workdir)
        if segments:
            return segments, metadata

    if subtitle_source in {"auto", "ocr"}:
        segments, metadata = acquire_ocr_segments(input_video, workdir, ocr_fps, ocr_min_chars, ocr_similarity)
        if segments:
            return segments, metadata
        if subtitle_source == "ocr":
            raise RuntimeError("OCR subtitle extraction did not yield usable dialogue subtitles")

    segments = transcribe_audio(vocals_path, asr_model, chunk_seconds, asr_device, asr_compute_type)
    return segments, {
        "source_name": "asr",
        "subtitle_language": "unknown",
        "needs_translation": True,
        "raw_source_srt": None,
        "raw_secondary_srt": None,
    }


def translate_missing_segments(
    segments: list[Segment],
    target_language: str,
    provider: str,
    retries: int,
    batch_size: int,
    timeout_seconds: int,
    hy_model: str,
    hy_device: str,
    hy_max_new_tokens: int,
) -> TranslationBackend:
    pending = [segment for segment in segments if not segment.translated_text.strip()]
    backend = build_translation_backend(target_language, provider, hy_model, hy_device, hy_max_new_tokens)
    if not pending:
        return backend
    print(f"[info] Translation provider: {backend.name}")
    safe_batch_size = max(1, batch_size)
    total_batches = (len(pending) + safe_batch_size - 1) // safe_batch_size
    for batch_index, batch_start in enumerate(range(0, len(pending), safe_batch_size), start=1):
        log_progress("Translation batch", batch_index, total_batches)
        batch = pending[batch_start : batch_start + safe_batch_size]
        translated_texts = translate_batch_with_backend([segment.source_text for segment in batch], backend, retries, timeout_seconds)
        for segment, translated_text in zip(batch, translated_texts, strict=True):
            segment.translated_text = translated_text
    return backend


def normalize_translate_code(target_language: str) -> str:
    mapping = {
        "zh-CN": "zh-CN",
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
    }
    return mapping.get(target_language, target_language)


def normalize_gpt_sovits_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    mapping = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-tw": "zh",
        "zh-hant": "zh",
        "zho": "zh",
        "chi": "zh",
        "en": "en",
        "english": "en",
        "ja": "ja",
        "jp": "ja",
        "japanese": "ja",
        "ko": "ko",
        "korean": "ko",
        "yue": "yue",
        "cantonese": "yue",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unsupported GPT-SoVITS language: {value}. Expected one of zh, en, ja, ko, yue, or auto for target text.")


def build_gpt_sovits_config(args: argparse.Namespace) -> GptSovitsConfig | None:
    text_lang_value = args.gpt_sovits_text_lang
    if text_lang_value == "auto":
        text_lang_value = args.target_language

    return GptSovitsConfig(
        api_url=args.gpt_sovits_url.rstrip("/"),
        ref_audio_path=args.gpt_sovits_ref_audio.resolve(),
        prompt_text=args.gpt_sovits_prompt_text.strip(),
        prompt_lang=normalize_gpt_sovits_language(args.gpt_sovits_prompt_lang),
        text_lang=normalize_gpt_sovits_language(text_lang_value),
        text_split_method=args.gpt_sovits_text_split_method,
        speed_factor=max(args.gpt_sovits_speed, 0.05),
    )


def detect_reference_prompt_language(text: str) -> str:
    normalized = normalize_segment_text(text)
    if not normalized:
        return DEFAULT_PROFILE_PROMPT_LANG
    if contains_cjk(normalized):
        return "zh"
    return "en"


def trim_reference_prompt_text(text: str, limit: int = 120) -> str:
    normalized = normalize_segment_text(text)
    if len(normalized) <= limit:
        return normalized
    trimmed = normalized[:limit].rsplit(" ", 1)[0].strip()
    return trimmed or normalized[:limit].strip()


def build_reference_config_from_values(
    *,
    args: argparse.Namespace,
    ref_audio_path: Path,
    prompt_text: str,
    prompt_lang: str,
) -> GptSovitsConfig:
    text_lang_value = args.gpt_sovits_text_lang
    if text_lang_value == "auto":
        text_lang_value = args.target_language
    return GptSovitsConfig(
        api_url=args.gpt_sovits_url.rstrip("/"),
        ref_audio_path=ref_audio_path.resolve(),
        prompt_text=prompt_text.strip(),
        prompt_lang=normalize_gpt_sovits_language(prompt_lang),
        text_lang=normalize_gpt_sovits_language(text_lang_value),
        text_split_method=args.gpt_sovits_text_split_method,
        speed_factor=max(args.gpt_sovits_speed, 0.05),
    )


def reference_segment_priority(segment: Segment) -> tuple[float, float, int]:
    duration_score = -abs(min(segment.duration, 9.5) - 6.5)
    punctuation_bonus = 1.0 if ends_sentence(segment.source_text) else 0.0
    return (
        duration_score + punctuation_bonus,
        segment.duration,
        min(len(normalize_segment_text(segment.source_text)), 180),
    )


def export_reference_clip(source_audio: Path, start: float, end: float, output_path: Path) -> None:
    clip_duration = max(end - start, 0.35)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{clip_duration:.3f}",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "32000",
            str(output_path),
        ]
    )


def detect_single_speaker_reference(
    args: argparse.Namespace,
    vocals_path: Path,
    segments: list[Segment],
    workdir: Path,
) -> GptSovitsReferenceSelection | None:
    reference_dir = ensure_dir(workdir / "reference")
    candidates = [
        segment
        for segment in segments
        if 3.0 <= segment.duration <= 12.0 and len(normalize_segment_text(segment.source_text)) >= 12
    ]
    if not candidates:
        return None

    for candidate_index, segment in enumerate(sorted(candidates, key=reference_segment_priority, reverse=True)[:8], start=1):
        prompt_text = trim_reference_prompt_text(segment.source_text)
        if len(prompt_text) < 8:
            continue
        clip_path = reference_dir / f"auto_ref_{candidate_index:02d}.wav"
        export_reference_clip(vocals_path, segment.start, segment.end, clip_path)
        levels = probe_audio_levels(clip_path)
        if segment_audio_issue(levels) in {"very-quiet", "mostly-silent"}:
            continue
        detected_pitch_hz = estimate_pitch_hz(clip_path)
        return GptSovitsReferenceSelection(
            config=build_reference_config_from_values(
                args=args,
                ref_audio_path=clip_path,
                prompt_text=prompt_text,
                prompt_lang=detect_reference_prompt_language(prompt_text),
            ),
            source="auto-source-segment",
            detail=f"segment={segment.start:.3f}-{segment.end:.3f}s",
            detected_pitch_hz=detected_pitch_hz,
        )
    return None


def fallback_reference_search_paths(filename: str) -> list[Path]:
    candidates: list[Path] = []
    with contextlib.suppress(Exception):
        from importlib import resources

        bundled_resource = resources.files("video_translator_assets").joinpath(filename)
        if bundled_resource.is_file():
            candidates.append(Path(bundled_resource))
    candidates.extend([
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "artifacts" / filename,
        PROJECT_ROOT / "samples" / filename,
        Path.home() / ".config" / DEFAULT_PROFILE_DIRNAME / filename,
    ])
    return candidates


def resolve_existing_reference_path(value: object, default_filename: str) -> Path | None:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(str(value)).expanduser())
    candidates.extend(fallback_reference_search_paths(default_filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None

def estimate_pitch_hz(path: Path) -> float | None:
    try:
        import numpy as np
    except Exception:
        return None

    with tempfile.TemporaryDirectory(prefix="vt-pitch-") as temp_dir:
        wav_path = Path(temp_dir) / "pitch_probe.wav"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ]
        )
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    if samples.size < sample_rate:
        return None

    samples = samples - float(np.mean(samples))
    window_size = min(sample_rate * 2, samples.size)
    step = max(window_size // 4, sample_rate // 5)
    best_window = None
    best_energy = 0.0
    max_start = max(samples.size - window_size, 1)
    for start in range(0, max_start, step):
        window = samples[start : start + window_size]
        energy = float(np.sqrt(np.mean(window * window)))
        if energy > best_energy:
            best_energy = energy
            best_window = window
    if best_window is None or best_energy < 0.01:
        return None

    min_lag = max(1, int(sample_rate / 280.0))
    max_lag = min(len(best_window) - 1, int(sample_rate / 85.0))
    if max_lag <= min_lag:
        return None
    autocorr = np.correlate(best_window, best_window, mode="full")[len(best_window) - 1 :]
    if autocorr.size <= max_lag or autocorr[0] <= 0:
        return None
    lag_slice = autocorr[min_lag : max_lag + 1]
    peak_offset = int(np.argmax(lag_slice))
    peak_lag = min_lag + peak_offset
    peak_value = float(autocorr[peak_lag])
    if peak_value <= autocorr[0] * 0.18:
        return None
    pitch_hz = sample_rate / float(peak_lag)
    if not math.isfinite(pitch_hz):
        return None
    return pitch_hz


def preferred_fallback_gender(vocals_path: Path, workdir: Path, segments: list[Segment]) -> str:
    pitch_estimates: list[float] = []
    reference_dir = ensure_dir(workdir / "reference")
    candidates = [segment for segment in segments if 2.0 <= segment.duration <= 10.0 and len(normalize_segment_text(segment.source_text)) >= 8]
    for index, segment in enumerate(sorted(candidates, key=reference_segment_priority, reverse=True)[:5], start=1):
        probe_clip = reference_dir / f"pitch_probe_{index:02d}.wav"
        export_reference_clip(vocals_path, segment.start, segment.end, probe_clip)
        pitch_hz = estimate_pitch_hz(probe_clip)
        if pitch_hz is not None:
            pitch_estimates.append(pitch_hz)
    if not pitch_estimates:
        return "female"
    median_pitch = statistics.median(pitch_estimates)
    return "female" if median_pitch >= 165.0 else "male"


def detect_fallback_reference(
    args: argparse.Namespace,
    vocals_path: Path,
    workdir: Path,
    segments: list[Segment],
) -> GptSovitsReferenceSelection | None:
    preferred_gender = preferred_fallback_gender(vocals_path, workdir, segments)
    ordered = [preferred_gender, "male" if preferred_gender == "female" else "female"]
    path_map = {
        "female": resolve_existing_reference_path(args.gpt_sovits_fallback_female_audio, DEFAULT_GPT_SOVITS_FALLBACK_FEMALE_REF.name),
        "male": resolve_existing_reference_path(args.gpt_sovits_fallback_male_audio, DEFAULT_GPT_SOVITS_FALLBACK_MALE_REF.name),
    }
    for gender in ordered:
        audio_path = path_map[gender]
        if audio_path is None:
            continue
        return GptSovitsReferenceSelection(
            config=build_reference_config_from_values(
                args=args,
                ref_audio_path=audio_path,
                prompt_text=DEFAULT_PROFILE_PROMPT_TEXT,
                prompt_lang=DEFAULT_PROFILE_PROMPT_LANG,
            ),
            source=f"fallback-{gender}",
            detail=audio_path.name,
            detected_pitch_hz=None,
        )
    return None


def resolve_gpt_sovits_reference(
    args: argparse.Namespace,
    vocals_path: Path,
    segments: list[Segment],
    workdir: Path,
) -> GptSovitsReferenceSelection:
    manual_config = build_gpt_sovits_config(args) if args.gpt_sovits_ref_audio else None
    if args.gpt_sovits_ref_mode == "manual":
        if manual_config is None:
            raise RuntimeError("Manual GPT-SoVITS mode requires an explicit reference audio")
        return GptSovitsReferenceSelection(config=manual_config, source="manual", detail="explicit-or-default")

    auto_reference = detect_single_speaker_reference(args, vocals_path, segments, workdir)
    if auto_reference is not None:
        return auto_reference

    fallback_reference = detect_fallback_reference(args, vocals_path, workdir, segments)
    if fallback_reference is not None:
        return fallback_reference

    if manual_config is not None:
        return GptSovitsReferenceSelection(config=manual_config, source="bundled-fallback", detail=manual_config.ref_audio_path.name)
    raise RuntimeError("Unable to resolve any GPT-SoVITS reference audio")


def synthesize_segment_audio(
    text: str,
    output_path: Path,
    timeout_seconds: int,
    retries: int,
    gpt_sovits_config: GptSovitsConfig | None,
) -> str:
    if gpt_sovits_config is None:
        raise RuntimeError("GPT-SoVITS configuration is missing")
    try:
        synthesize_gpt_sovits_with_retry(text, output_path, timeout_seconds, retries, gpt_sovits_config)
        return "gpt-sovits"
    except Exception as exc:
        raise RuntimeError(f"GPT-SoVITS failed: {exc}") from exc


def synthesize_gpt_sovits(text: str, output_path: Path, timeout_seconds: int, config: GptSovitsConfig) -> None:
    endpoint = f"{config.api_url}/tts"
    payload = {
        "text": text,
        "text_lang": config.text_lang,
        "ref_audio_path": str(config.ref_audio_path),
        "prompt_text": config.prompt_text,
        "prompt_lang": config.prompt_lang,
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": config.speed_factor,
        "text_split_method": config.text_split_method,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
    content_type = response.headers.get("content-type", "")
    if not response.ok:
        detail = response.text.strip()
        with contextlib.suppress(Exception):
            detail = response.json().get("message") or detail
        raise RuntimeError(f"{response.status_code} {detail}".strip())
    if "application/json" in content_type.lower():
        detail = response.text.strip()
        with contextlib.suppress(Exception):
            detail = response.json().get("message") or detail
        raise RuntimeError(detail or "GPT-SoVITS returned JSON instead of audio")
    if not response.content:
        raise RuntimeError("GPT-SoVITS returned an empty audio payload")
    output_path.write_bytes(response.content)


def synthesize_gpt_sovits_with_retry(
    text: str,
    output_path: Path,
    timeout_seconds: int,
    retries: int,
    config: GptSovitsConfig,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            synthesize_gpt_sovits(text, output_path, timeout_seconds, config)
            return
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt * 2, 6))
    if last_error is None:
        raise RuntimeError("GPT-SoVITS failed without raising an explicit exception")
    raise last_error


def build_atempo_chain(playback_ratio: float) -> str:
    playback_ratio = min(max(playback_ratio, 0.5), 4.0)
    factors: list[float] = []
    remaining = playback_ratio
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.5f}" for factor in factors)


def fit_audio_to_slot(input_path: Path, output_path: Path, target_duration: float) -> tuple[float, float]:
    safe_target = max(target_duration, 0.25)
    input_duration = ffprobe_duration(input_path)
    desired_ratio = input_duration / safe_target
    applied_ratio = min(max(desired_ratio, MIN_AUDIO_FIT_RATIO), MAX_AUDIO_FIT_RATIO)
    filter_chain = f"{build_atempo_chain(applied_ratio)},apad,atrim=0:{safe_target:.3f}"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            str(DUB_CHANNELS),
            "-ar",
            str(DUB_SAMPLE_RATE),
            "-af",
            filter_chain,
            str(output_path),
        ]
    )
    return ffprobe_duration(output_path), applied_ratio


def planned_segment_duration(input_duration: float, base_duration: float) -> float:
    safe_base = max(base_duration, 0.35)
    min_required_duration = input_duration / MAX_AUDIO_FIT_RATIO
    return max(safe_base, min_required_duration, 0.35)


def synthesize_segments(
    segments: list[Segment],
    total_duration: float,
    workdir: Path,
    timeout_seconds: int,
    retries: int,
    gpt_sovits_config: GptSovitsConfig | None,
    reference_source: str | None = None,
    reference_detail: str | None = None,
) -> tuple[Path, list[str]]:
    tts_dir = ensure_dir(workdir / "tts")
    timeline_dir = ensure_dir(workdir / "timeline")
    audit_log = workdir / "audio_audit.log"
    write_audio_audit_header(
        audit_log,
        tts_provider="gpt-sovits",
        total_duration=total_duration,
        gpt_sovits_config=gpt_sovits_config,
        reference_source=reference_source,
        reference_detail=reference_detail,
    )
    entries: list[Path] = []
    tts_backends: list[str] = []
    cursor = 0.0
    total_segments = len(segments)

    for index, segment in enumerate(segments, start=1):
        log_progress("TTS segment", index, total_segments, detail=f"{segment.start:.1f}s-{segment.end:.1f}s")
        original_start = segment.start
        original_end = segment.end
        scheduled_start = max(original_start, cursor)
        if scheduled_start > cursor:
            silence = timeline_dir / f"silence_{index:04d}.wav"
            render_silence(scheduled_start - cursor, silence)
            entries.append(silence)
            cursor = scheduled_start

        raw_extension = ".wav"
        raw_tts = tts_dir / f"segment_{index:04d}{raw_extension}"
        adjusted_tts = tts_dir / f"segment_{index:04d}_fit.wav"
        try:
            backend, raw_duration, raw_levels, quiet_retry_attempts = synthesize_segment_with_quiet_retry(
                text=segment.translated_text,
                raw_tts=raw_tts,
                timeout_seconds=timeout_seconds,
                retries=retries,
                gpt_sovits_config=gpt_sovits_config,
                audit_log=audit_log,
                segment_index=index,
            )
            tts_backends.append(backend)
            base_duration = max(original_end - original_start, 0.35)
            target_duration = planned_segment_duration(raw_duration, base_duration)
            remaining_duration = max(total_duration - scheduled_start, 0.35)
            target_duration = min(target_duration, remaining_duration)
            adjusted_duration, speed_factor = fit_audio_to_slot(raw_tts, adjusted_tts, target_duration)
            adjusted_levels = probe_audio_levels(adjusted_tts)
        except Exception as exc:
            append_segment_audit_line(
                audit_log,
                index=index,
                backend="gpt-sovits",
                original_start=original_start,
                original_end=original_end,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_start,
                translated_text=segment.translated_text,
                raw_tts=raw_tts,
                adjusted_tts=adjusted_tts if adjusted_tts.exists() else None,
                raw_duration=probe_duration_or_none(raw_tts),
                adjusted_duration=probe_duration_or_none(adjusted_tts),
                speed_factor=None,
                raw_levels=probe_audio_levels(raw_tts),
                adjusted_levels=probe_audio_levels(adjusted_tts) if adjusted_tts.exists() else None,
                status="failed",
                quiet_retry_attempts=0,
                error=str(exc),
            )
            raise

        append_segment_audit_line(
            audit_log,
            index=index,
            backend=backend,
            original_start=original_start,
            original_end=original_end,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_start + adjusted_duration,
            translated_text=segment.translated_text,
            raw_tts=raw_tts,
            adjusted_tts=adjusted_tts,
            raw_duration=raw_duration,
            adjusted_duration=adjusted_duration,
            speed_factor=speed_factor,
            raw_levels=raw_levels,
            adjusted_levels=adjusted_levels,
            status="ok",
            quiet_retry_attempts=quiet_retry_attempts,
        )

        segment.start = scheduled_start
        segment.end = scheduled_start + adjusted_duration
        segment.tts_path = str(raw_tts)
        segment.adjusted_tts_path = str(adjusted_tts)
        segment.tts_duration = adjusted_duration
        segment.speed_factor = speed_factor

        entries.append(adjusted_tts)
        cursor = segment.end

    if total_duration > cursor:
        silence = timeline_dir / "silence_final.wav"
        render_silence(total_duration - cursor, silence)
        entries.append(silence)

    concat_list = workdir / "dub_track.ffconcat"
    concat_lines = ["ffconcat version 1.0"]
    for entry in entries:
        concat_lines.append(f"file '{entry.as_posix()}'")
    concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    dub_track = workdir / "dub_track.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(concat_list),
            str(dub_track),
        ]
    )
    append_audio_audit_line(audit_log, "")
    append_audio_audit_line(audit_log, "## Dub Track Summary")
    append_audio_audit_line(
        audit_log,
        f"dub_track_exists={dub_track.exists()} dub_track_duration={format_duration_for_log(probe_duration_or_none(dub_track))} dub_track_path={dub_track}",
    )
    return dub_track, tts_backends


def render_silence(duration: float, output_path: Path) -> None:
    safe_duration = max(duration, 0.01)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={DUB_SAMPLE_RATE}:cl=mono",
            "-t",
            f"{safe_duration:.3f}",
            str(output_path),
        ]
    )


def mix_audio(input_video: Path, background_track: Path, dub_track: Path, output_video: Path, background_volume: float, dub_volume: float) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(background_track),
            "-i",
            str(dub_track),
            "-filter_complex",
            (
                f"[1:a]volume={background_volume}[bg];"
                f"[2:a]loudnorm=I=-16:TP=-1.5,volume={dub_volume}[dub];"
                "[bg][dub]amix=inputs=2:normalize=0[aout]"
            ),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ]
    )


def save_manifest(segments: list[Segment], workdir: Path, metadata: dict[str, object]) -> Path:
    manifest_path = workdir / "manifest.json"
    payload = {
        "metadata": metadata,
        "segments": [asdict(segment) for segment in segments],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def default_output_path(input_video: Path, target_language: str) -> Path:
    suffix = target_language.replace("-", "_")
    return input_video.with_name(f"{input_video.stem}_{suffix}{input_video.suffix}")


def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def collect_input_videos(input_dir: Path) -> list[Path]:
    videos = [path for path in sorted(input_dir.iterdir()) if is_supported_video_file(path)]
    if not videos:
        raise FileNotFoundError(f"No supported video files found in: {input_dir}")
    return videos


def resolve_batch_output_dir(input_dir: Path, output_arg: Path | None) -> Path:
    if output_arg is None:
        return ensure_dir(input_dir / DEFAULT_BATCH_OUTPUT_DIRNAME)
    return ensure_dir(output_arg.resolve())


def run_translation(args: argparse.Namespace, input_video: Path, output_video: Path, workdir: Path) -> None:
    ensure_gpt_sovits_service(args.gpt_sovits_url)
    asr_device, asr_compute_type = resolve_asr_runtime(args.asr_device, args.asr_compute_type)
    demucs_device = resolve_demucs_device(args.demucs_device)
    prepared_input_video, original_duration, total_duration = prepare_input_video(
        input_video,
        workdir,
        args.trim_start,
        args.trim_end,
    )

    print(f"[info] Working directory: {workdir}")
    print(f"[info] Video duration: {total_duration:.1f}s")
    if prepared_input_video != input_video:
        print(
            f"[info] Trimmed input video: start={args.trim_start:.1f}s end={args.trim_end:.1f}s "
            f"original={original_duration:.1f}s trimmed={total_duration:.1f}s"
        )
    print(f"[info] ASR runtime: device={asr_device}, compute_type={asr_compute_type}")
    print(f"[info] Demucs runtime: device={demucs_device}")

    print("[info] Extracting source audio")
    full_mix, speech_mix = extract_audio(prepared_input_video, workdir)
    print("[info] Separating vocals and background")
    vocals_path, background_track, separation_mode = separate_audio(
        args.separator,
        speech_mix,
        full_mix,
        workdir,
        demucs_device,
        args.mdx_model,
    )
    print(f"[info] Separation mode: {separation_mode}")

    print("[info] Acquiring subtitle segments")
    segments, subtitle_metadata = acquire_segments(
        prepared_input_video,
        vocals_path,
        workdir,
        args.subtitle_source,
        args.asr_model,
        asr_device,
        asr_compute_type,
        args.chunk_seconds,
        args.ocr_fps,
        args.ocr_min_chars,
        args.ocr_similarity,
    )
    print(f"[info] Subtitle source: {subtitle_metadata['source_name']}")
    print(f"[info] Prepared {len(segments)} segments")

    translation_backend = build_translation_backend(
        args.target_language,
        args.translation_provider,
        args.hy_model,
        args.hy_device,
        args.hy_max_new_tokens,
    )
    translation_metadata = translation_backend_metadata(translation_backend)

    if subtitle_metadata["needs_translation"]:
        print("[info] Translating subtitle text")
        translation_backend = translate_missing_segments(
            segments,
            args.target_language,
            args.translation_provider,
            args.translate_retries,
            args.translate_batch_size,
            args.translate_timeout,
            args.hy_model,
            args.hy_device,
            args.hy_max_new_tokens,
        )
        translation_metadata = translation_backend_metadata(translation_backend, resolve_runtime=True)
    else:
        for segment in segments:
            segment.translated_text = segment.source_text
    source_srt = write_srt(segments, workdir / "source.srt", "source_text")
    translated_srt = write_srt(segments, workdir / "translated.srt", "translated_text")
    print("[info] Resolving GPT-SoVITS reference voice")
    gpt_sovits_reference = resolve_gpt_sovits_reference(
        args,
        vocals_path,
        segments,
        workdir,
    )
    print(
        f"[info] GPT-SoVITS reference source: {gpt_sovits_reference.source}"
        + (f" ({gpt_sovits_reference.detail})" if gpt_sovits_reference.detail else "")
    )
    print("[info] Synthesizing dubbed speech")
    dub_track, tts_backends = synthesize_segments(
        segments,
        total_duration,
        workdir,
        args.tts_timeout,
        args.tts_retries,
        gpt_sovits_reference.config,
        reference_source=gpt_sovits_reference.source,
        reference_detail=gpt_sovits_reference.detail,
    )
    print("[info] Mixing dubbed speech back into the video")
    mix_audio(
        prepared_input_video,
        background_track,
        dub_track,
        output_video,
        background_volume=args.background_volume,
        dub_volume=args.dub_volume,
    )
    append_mix_audit_section(
        workdir / "audio_audit.log",
        background_track=background_track,
        dub_track=dub_track,
        output_video=output_video,
        background_volume=args.background_volume,
        dub_volume=args.dub_volume,
    )

    manifest = save_manifest(
        segments,
        workdir,
        {
            "input_video": str(input_video),
            "processed_input_video": str(prepared_input_video),
            "output_video": str(output_video),
            "original_input_duration": original_duration,
            "processed_input_duration": total_duration,
            "target_language": args.target_language,
            "tts_provider": "gpt-sovits",
            "translation_provider": args.translation_provider,
            "resolved_translation_provider": translation_metadata["resolved_translation_provider"],
            "translation_model": translation_metadata["translation_model"],
            "resolved_translation_device": translation_metadata["resolved_translation_device"],
            "subtitle_source": args.subtitle_source,
            "resolved_subtitle_source": subtitle_metadata["source_name"],
            "subtitle_language": subtitle_metadata["subtitle_language"],
            "translation_skipped": not subtitle_metadata["needs_translation"],
            "asr_model": args.asr_model,
            "asr_device": args.asr_device,
            "resolved_asr_device": asr_device,
            "asr_compute_type": asr_compute_type,
            "ocr_fps": args.ocr_fps,
            "ocr_min_chars": args.ocr_min_chars,
            "ocr_similarity": args.ocr_similarity,
            "trim_start": args.trim_start,
            "trim_end": args.trim_end,
            "chunk_seconds": args.chunk_seconds,
            "separation_mode": separation_mode,
            "mdx_model": args.mdx_model,
            "demucs_device": args.demucs_device,
            "resolved_demucs_device": demucs_device,
            "tts_backends": sorted(set(tts_backends)),
            "tts_timeout": args.tts_timeout,
            "tts_retries": args.tts_retries,
            "gpt_sovits_url": args.gpt_sovits_url,
            "gpt_sovits_ref_mode": args.gpt_sovits_ref_mode,
            "gpt_sovits_ref_source": gpt_sovits_reference.source,
            "gpt_sovits_ref_detail": gpt_sovits_reference.detail,
            "gpt_sovits_ref_audio": str(gpt_sovits_reference.config.ref_audio_path),
            "gpt_sovits_prompt_text": gpt_sovits_reference.config.prompt_text,
            "gpt_sovits_prompt_lang": gpt_sovits_reference.config.prompt_lang,
            "gpt_sovits_text_lang": args.gpt_sovits_text_lang,
            "gpt_sovits_text_split_method": args.gpt_sovits_text_split_method,
            "gpt_sovits_speed": args.gpt_sovits_speed,
            "translate_retries": args.translate_retries,
            "translate_batch_size": args.translate_batch_size,
            "translate_timeout": args.translate_timeout,
            "hy_model": args.hy_model,
            "hy_device": args.hy_device,
            "hy_max_new_tokens": args.hy_max_new_tokens,
            "background_volume": args.background_volume,
            "dub_volume": args.dub_volume,
            "audio_audit_log": str(workdir / "audio_audit.log"),
            "source_srt": str(source_srt),
            "translated_srt": str(translated_srt),
            "ocr_source_srt": str(subtitle_metadata["raw_source_srt"]) if subtitle_metadata["raw_source_srt"] else None,
            "ocr_secondary_srt": str(subtitle_metadata["raw_secondary_srt"]) if subtitle_metadata["raw_secondary_srt"] else None,
            "ocr_debug_json": str(subtitle_metadata["ocr_debug_json"]) if subtitle_metadata.get("ocr_debug_json") else None,
        },
    )
    print(f"[info] Manifest written to {manifest}")
    print(f"[info] Output video written to {output_video}")


def process_single_video(args: argparse.Namespace, input_video: Path, output_video: Path, workdir_root: Path | None = None) -> None:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if workdir_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="video_translator_")
        workdir = Path(temp_dir.name)
    else:
        workdir = ensure_dir(workdir_root)

    try:
        run_translation(args, input_video, output_video, workdir)
    except Exception as exc:
        failure_log = write_failure_log(workdir, exc)
        print(f"[error] Failed processing {input_video}: {exc}")
        print(f"[error] Full traceback written to {failure_log}")
        raise
    finally:
        if temp_dir is not None and not args.keep_workdir:
            temp_dir.cleanup()


def process_batch(args: argparse.Namespace, input_dir: Path) -> None:
    videos = collect_input_videos(input_dir)
    output_dir = resolve_batch_output_dir(input_dir, args.output)
    print(f"[info] Found {len(videos)} video(s) in {input_dir}")
    print(f"[info] Batch output directory: {output_dir}")

    failures: list[tuple[Path, Exception]] = []
    for index, video_path in enumerate(videos, start=1):
        output_video = output_dir / default_output_path(video_path, args.target_language).name
        workdir_root = None
        if args.workdir is not None:
            workdir_root = args.workdir.resolve() / video_path.stem

        print(f"[info] [{index}/{len(videos)}] Processing {video_path.name}")
        try:
            process_single_video(args, video_path.resolve(), output_video.resolve(), workdir_root)
        except Exception as exc:
            failures.append((video_path, exc))

    if failures:
        failed_names = ", ".join(path.name for path, _ in failures)
        raise RuntimeError(f"Batch completed with {len(failures)} failure(s): {failed_names}")


def write_failure_log(workdir: Path, exc: Exception) -> Path:
    log_path = workdir / "error.log"
    log_path.write_text(traceback.format_exc(), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")
    if args.input_dir is not None:
        input_dir = args.input_dir.resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
        process_batch(args, input_dir)
        return

    input_video = args.input_video.resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video = (args.output or default_output_path(input_video, args.target_language)).resolve()
    workdir_root = args.workdir.resolve() if args.workdir is not None else None
    process_single_video(args, input_video, output_video, workdir_root)


if __name__ == "__main__":
    main()