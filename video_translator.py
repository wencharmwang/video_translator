#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import requests
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

import edge_tts
from deep_translator import GoogleTranslator, MyMemoryTranslator

if TYPE_CHECKING:
    from ocr_subtitles import SubtitleBlock


__version__ = "0.1.4"

DEFAULT_TARGET_LANGUAGE = "zh-CN"
DEFAULT_TTS_VOICE = "zh-CN-XiaoyiNeural"
DEFAULT_ASR_MODEL = "small"
DEFAULT_CHUNK_SECONDS = 300
DEFAULT_TTS_TIMEOUT = 60
DEFAULT_TTS_RETRIES = 3
DEFAULT_TRANSLATE_RETRIES = 3
DEFAULT_TRANSLATE_BATCH_SIZE = 12
DEFAULT_TRANSLATE_TIMEOUT = 30
DEFAULT_EDGE_RATE = "+0%"
DEFAULT_EDGE_PITCH = "+0Hz"
DEFAULT_EDGE_VOLUME = "+0%"
DEFAULT_ASR_DEVICE = "auto"
DEFAULT_ASR_COMPUTE_TYPE = "default"
DEFAULT_DEMUCS_DEVICE = "auto"
DEFAULT_SUBTITLE_SOURCE = "auto"
DEFAULT_OCR_FPS = 1.0
DEFAULT_OCR_MIN_CHARS = 10
DEFAULT_OCR_SIMILARITY = 0.72
DUB_SAMPLE_RATE = 44100
DUB_CHANNELS = 1
MIN_AUDIO_FIT_RATIO = 0.92
MAX_AUDIO_FIT_RATIO = 1.22
SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？", "…")


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


class SubtitleAcquisition(Protocol):
    source_name: str
    subtitle_language: str
    needs_translation: bool
    raw_source_srt: Path | None
    raw_secondary_srt: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a video into a target language and rebuild the dubbed video.")
    parser.add_argument("--video", type=Path, default=None, help="Path to the source video file")
    parser.add_argument("--output", type=Path, default=None, help="Path to the translated output video")
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE, help="Target language for translation, e.g. zh-CN")
    parser.add_argument("--voice", default=DEFAULT_TTS_VOICE, help="Edge TTS voice name")
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
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS, help="Chunk duration for ASR processing")
    parser.add_argument("--workdir", type=Path, default=None, help="Directory for intermediate files")
    parser.add_argument(
        "--separator",
        choices=("auto", "demucs", "ffmpeg", "none"),
        default="auto",
        help="Voice/background separation strategy",
    )
    parser.add_argument(
        "--demucs-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=DEFAULT_DEMUCS_DEVICE,
        help="Demucs inference device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--keep-workdir", action="store_true", help="Keep intermediate artifacts")
    parser.add_argument("--dub-volume", type=float, default=0.78, help="Mix weight for synthesized speech")
    parser.add_argument("--background-volume", type=float, default=1.18, help="Mix weight for background audio")
    parser.add_argument("--tts-timeout", type=int, default=DEFAULT_TTS_TIMEOUT, help="Per-request timeout in seconds for Edge TTS")
    parser.add_argument("--tts-retries", type=int, default=DEFAULT_TTS_RETRIES, help="Retry count before falling back to local TTS")
    parser.add_argument("--translate-retries", type=int, default=DEFAULT_TRANSLATE_RETRIES, help="Retry count per translator before switching to the next translation backend")
    parser.add_argument("--translate-batch-size", type=int, default=DEFAULT_TRANSLATE_BATCH_SIZE, help="Number of segments to send in each batch translation request")
    parser.add_argument("--translate-timeout", type=int, default=DEFAULT_TRANSLATE_TIMEOUT, help="Timeout in seconds for each translation HTTP request")
    parser.add_argument("--edge-rate", default=DEFAULT_EDGE_RATE, help="Edge TTS base speaking rate, e.g. -8%% or +5%%")
    parser.add_argument("--edge-pitch", default=DEFAULT_EDGE_PITCH, help="Edge TTS pitch, e.g. -12Hz")
    parser.add_argument("--edge-volume", default=DEFAULT_EDGE_VOLUME, help="Edge TTS volume, e.g. +0%%")
    raw_args = sys.argv[1:]
    if raw_args and not raw_args[0].startswith("-") and "--video" not in raw_args:
        raw_args = ["--video", raw_args[0], *raw_args[1:]]

    args = parser.parse_args(raw_args)
    args.input_video = args.video
    if args.input_video is None:
        parser.error("one of the following arguments is required: --video")
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_progress(stage: str, current: int, total: int, *, detail: str = "") -> None:
    if total <= 0:
        return
    if total <= 20 or current == 1 or current == total or current % 10 == 0:
        suffix = f" - {detail}" if detail else ""
        print(f"[info] {stage}: {current}/{total}{suffix}")


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
) -> tuple[Path, Path, str]:
    if strategy == "none":
        return speech_mix, full_mix, "none"
    if strategy == "ffmpeg":
        vocals, background = separate_with_ffmpeg(speech_mix, full_mix, workdir)
        return vocals, background, "ffmpeg"
    if strategy == "demucs":
        vocals, background = separate_with_demucs(speech_mix, full_mix, workdir, demucs_device)
        return vocals, background, "demucs"

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
    model = load_whisper_model_class()(model_name, device=asr_device, compute_type=asr_compute_type)
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


def build_translators(target_language: str) -> list[tuple[str, BatchTranslator]]:
    normalized_target = normalize_translate_code(target_language)
    return [
        ("google", GoogleTranslator(source="auto", target=normalized_target)),
        ("mymemory", MyMemoryTranslator(source="auto", target=normalized_target)),
    ]


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


def translate_batch_with_fallback(texts: list[str], target_language: str, retries: int, timeout_seconds: int) -> list[str]:
    translators = build_translators(target_language)
    last_error: Exception | None = None
    for provider_name, translator in translators:
        for attempt in range(1, retries + 1):
            try:
                with requests_timeout(timeout_seconds):
                    translated = run_with_timeout(timeout_seconds, lambda: translator.translate_batch(texts))
                if len(translated) != len(texts):
                    raise RuntimeError(f"{provider_name} returned {len(translated)} items for {len(texts)} inputs")
                normalized = [item.strip() if item else "" for item in translated]
                if any(not item for item in normalized):
                    raise RuntimeError(f"{provider_name} returned empty text inside a batch")
                return normalized
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(attempt, 3))
    raise RuntimeError(f"Failed to translate batch within timeout={timeout_seconds}s. Last error: {last_error}")


def translate_segments(segments: list[Segment], target_language: str, retries: int, batch_size: int, timeout_seconds: int) -> None:
    safe_batch_size = max(1, batch_size)
    total_batches = (len(segments) + safe_batch_size - 1) // safe_batch_size
    for batch_index, batch_start in enumerate(range(0, len(segments), safe_batch_size), start=1):
        log_progress("Translation batch", batch_index, total_batches)
        batch = segments[batch_start : batch_start + safe_batch_size]
        translated_texts = translate_batch_with_fallback([segment.source_text for segment in batch], target_language, retries, timeout_seconds)
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


def translate_missing_segments(segments: list[Segment], target_language: str, retries: int, batch_size: int, timeout_seconds: int) -> None:
    pending = [segment for segment in segments if not segment.translated_text.strip()]
    if not pending:
        return
    safe_batch_size = max(1, batch_size)
    total_batches = (len(pending) + safe_batch_size - 1) // safe_batch_size
    for batch_index, batch_start in enumerate(range(0, len(pending), safe_batch_size), start=1):
        log_progress("Translation batch", batch_index, total_batches)
        batch = pending[batch_start : batch_start + safe_batch_size]
        translated_texts = translate_batch_with_fallback([segment.source_text for segment in batch], target_language, retries, timeout_seconds)
        for segment, translated_text in zip(batch, translated_texts, strict=True):
            segment.translated_text = translated_text


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


async def synthesize_tts(text: str, voice: str, output_path: Path, rate: str, pitch: str, volume: str) -> None:
    communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    await communicator.save(str(output_path))


async def synthesize_tts_with_retry(
    text: str,
    voice: str,
    output_path: Path,
    timeout_seconds: int,
    retries: int,
    rate: str,
    pitch: str,
    volume: str,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            await asyncio.wait_for(
                synthesize_tts(text, voice, output_path, rate, pitch, volume),
                timeout=timeout_seconds,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            await asyncio.sleep(min(attempt * 2, 6))
    if last_error is None:
        raise RuntimeError("Edge TTS failed without raising an explicit exception")
    raise last_error


def synthesize_segment_audio(
    text: str,
    edge_voice: str,
    output_path: Path,
    timeout_seconds: int,
    retries: int,
    edge_rate: str,
    edge_pitch: str,
    edge_volume: str,
) -> str:
    try:
        asyncio.run(
            synthesize_tts_with_retry(
                text,
                edge_voice,
                output_path,
                timeout_seconds,
                retries,
                edge_rate,
                edge_pitch,
                edge_volume,
            )
        )
        return "edge-tts"
    except Exception as exc:
        raise RuntimeError(f"Edge TTS failed: {exc}") from exc


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
    voice: str,
    total_duration: float,
    workdir: Path,
    timeout_seconds: int,
    retries: int,
    edge_rate: str,
    edge_pitch: str,
    edge_volume: str,
) -> tuple[Path, list[str]]:
    tts_dir = ensure_dir(workdir / "tts")
    timeline_dir = ensure_dir(workdir / "timeline")
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

        raw_tts = tts_dir / f"segment_{index:04d}.mp3"
        backend = synthesize_segment_audio(
            segment.translated_text,
            voice,
            raw_tts,
            timeout_seconds,
            retries,
            edge_rate,
            edge_pitch,
            edge_volume,
        )
        tts_backends.append(backend)
        raw_duration = ffprobe_duration(raw_tts)
        base_duration = max(original_end - original_start, 0.35)
        target_duration = planned_segment_duration(raw_duration, base_duration)
        remaining_duration = max(total_duration - scheduled_start, 0.35)
        target_duration = min(target_duration, remaining_duration)

        adjusted_tts = tts_dir / f"segment_{index:04d}_fit.wav"
        adjusted_duration, speed_factor = fit_audio_to_slot(raw_tts, adjusted_tts, target_duration)

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


def write_failure_log(workdir: Path, exc: Exception) -> Path:
    log_path = workdir / "error.log"
    log_path.write_text(traceback.format_exc(), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")
    asr_device, asr_compute_type = resolve_asr_runtime(args.asr_device, args.asr_compute_type)
    demucs_device = resolve_demucs_device(args.demucs_device)

    input_video = args.input_video.resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video = (args.output or default_output_path(input_video, args.target_language)).resolve()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="video_translator_")
        workdir = Path(temp_dir.name)
    else:
        workdir = ensure_dir(args.workdir.resolve())

    print(f"[info] Working directory: {workdir}")
    total_duration = ffprobe_duration(input_video)
    print(f"[info] Video duration: {total_duration:.1f}s")
    print(f"[info] ASR runtime: device={asr_device}, compute_type={asr_compute_type}")
    print(f"[info] Demucs runtime: device={demucs_device}")

    try:
        print("[info] Extracting source audio")
        full_mix, speech_mix = extract_audio(input_video, workdir)
        print("[info] Separating vocals and background")
        vocals_path, background_track, separation_mode = separate_audio(
            args.separator,
            speech_mix,
            full_mix,
            workdir,
            demucs_device,
        )
        print(f"[info] Separation mode: {separation_mode}")

        print("[info] Acquiring subtitle segments")
        segments, subtitle_metadata = acquire_segments(
            input_video,
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

        if subtitle_metadata["needs_translation"]:
            print("[info] Translating subtitle text")
            translate_missing_segments(segments, args.target_language, args.translate_retries, args.translate_batch_size, args.translate_timeout)
        else:
            for segment in segments:
                segment.translated_text = segment.source_text
        source_srt = write_srt(segments, workdir / "source.srt", "source_text")
        translated_srt = write_srt(segments, workdir / "translated.srt", "translated_text")
        print("[info] Synthesizing dubbed speech")
        dub_track, tts_backends = synthesize_segments(
            segments,
            args.voice,
            total_duration,
            workdir,
            args.tts_timeout,
            args.tts_retries,
            args.edge_rate,
            args.edge_pitch,
            args.edge_volume,
        )
        print("[info] Mixing dubbed speech back into the video")
        mix_audio(
            input_video,
            background_track,
            dub_track,
            output_video,
            background_volume=args.background_volume,
            dub_volume=args.dub_volume,
        )

        manifest = save_manifest(
            segments,
            workdir,
            {
                "input_video": str(input_video),
                "output_video": str(output_video),
                "target_language": args.target_language,
                "voice": args.voice,
                "tts_provider": "edge",
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
                "chunk_seconds": args.chunk_seconds,
                "separation_mode": separation_mode,
                "demucs_device": args.demucs_device,
                "resolved_demucs_device": demucs_device,
                "tts_backends": sorted(set(tts_backends)),
                "tts_timeout": args.tts_timeout,
                "tts_retries": args.tts_retries,
                "translate_retries": args.translate_retries,
                "translate_batch_size": args.translate_batch_size,
                "translate_timeout": args.translate_timeout,
                "background_volume": args.background_volume,
                "dub_volume": args.dub_volume,
                "edge_rate": args.edge_rate,
                "edge_pitch": args.edge_pitch,
                "edge_volume": args.edge_volume,
                "source_srt": str(source_srt),
                "translated_srt": str(translated_srt),
                "ocr_source_srt": str(subtitle_metadata["raw_source_srt"]) if subtitle_metadata["raw_source_srt"] else None,
                "ocr_secondary_srt": str(subtitle_metadata["raw_secondary_srt"]) if subtitle_metadata["raw_secondary_srt"] else None,
                "ocr_debug_json": str(subtitle_metadata["ocr_debug_json"]) if subtitle_metadata.get("ocr_debug_json") else None,
            },
        )
        print(f"[info] Manifest written to {manifest}")
        print(f"[info] Source subtitles written to {source_srt}")
        print(f"[info] Translated subtitles written to {translated_srt}")
        print(f"[info] Output video written to {output_video}")
    except Exception as exc:
        log_path = write_failure_log(workdir, exc)
        print(f"[error] Pipeline failed. Details written to {log_path}", file=sys.stderr)
        raise
    finally:
        if temp_dir is not None and not args.keep_workdir:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()