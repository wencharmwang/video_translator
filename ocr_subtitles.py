#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR


DEFAULT_FPS = 2.0
DEFAULT_MIN_CHARS = 10
DEFAULT_MIN_CHINESE_CHARS = 3
DEFAULT_SIMILARITY = 0.72
DEFAULT_CROP_X = 0.0
DEFAULT_CROP_Y = 0.0
DEFAULT_CROP_W = 1.0
DEFAULT_CROP_H = 1.0
TEXT_RE = re.compile(r"[^A-Za-z0-9 ,.'?!:;\-()\u4e00-\u9fff，。！？：；、“”‘’…]+")
TITLE_KEYWORDS = (
    "apple original",
    "studios production",
    "presented by",
    "visionary",
    "content transfer",
    "production",
    "original",
)


@dataclass(slots=True)
class OcrSample:
    time: float
    english_text: str = ""
    chinese_text: str = ""


@dataclass(slots=True)
class SubtitleBlock:
    start: float
    end: float
    english_text: str = ""
    chinese_text: str = ""

    @property
    def text(self) -> str:
        return self.chinese_text or self.english_text

    @property
    def has_chinese(self) -> bool:
        return bool(self.chinese_text)

    @property
    def has_english(self) -> bool:
        return bool(self.english_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract burned-in subtitles from a video with OCR.")
    parser.add_argument("input_video", type=Path, help="Path to the source video")
    parser.add_argument("--output", type=Path, default=None, help="Output SRT path")
    parser.add_argument("--workdir", type=Path, default=None, help="Directory for intermediate OCR frames and logs")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame sampling rate for OCR")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Discard OCR lines shorter than this")
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY, help="Similarity threshold for merging consecutive OCR lines")
    parser.add_argument("--crop-x", type=float, default=DEFAULT_CROP_X, help="Left crop origin as ratio of video width")
    parser.add_argument("--crop-y", type=float, default=DEFAULT_CROP_Y, help="Top crop origin as ratio of video height")
    parser.add_argument("--crop-w", type=float, default=DEFAULT_CROP_W, help="Crop width as ratio of video width")
    parser.add_argument("--crop-h", type=float, default=DEFAULT_CROP_H, help="Crop height as ratio of video height")
    parser.add_argument("--engine", choices=("rapidocr", "tesseract"), default="rapidocr", help="OCR engine to use")
    parser.add_argument("--language", choices=("auto", "english", "chinese", "bilingual"), default="english", help="Which OCR subtitle line to export")
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_path(input_video: Path) -> Path:
    return input_video.with_name(f"{input_video.stem}.ocr.srt")


def build_filter(args: argparse.Namespace) -> str:
    filters = [f"fps={args.fps:.4f}"]
    if (args.crop_x, args.crop_y, args.crop_w, args.crop_h) != (0.0, 0.0, 1.0, 1.0):
        filters.append(f"crop=iw*{args.crop_w:.4f}:ih*{args.crop_h:.4f}:iw*{args.crop_x:.4f}:ih*{args.crop_y:.4f}")
    return ",".join(filters)


def extract_frames(input_video: Path, frames_dir: Path, args: argparse.Namespace) -> list[Path]:
    for existing_frame in frames_dir.glob("frame_*.png"):
        existing_frame.unlink()
    frame_pattern = frames_dir / "frame_%06d.png"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            build_filter(args),
            str(frame_pattern),
        ]
    )
    return sorted(frames_dir.glob("frame_*.png"))


def normalize_ocr_text(text: str) -> str:
    normalized = text.replace("\n", " ").replace("|", "I")
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    normalized = TEXT_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    return normalized.strip()


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def chinese_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return sum("\u4e00" <= character <= "\u9fff" for character in visible) / len(visible)


def run_tesseract_ocr(frame_path: Path) -> str:
    result = run_command(
        [
            "tesseract",
            str(frame_path),
            "stdout",
            "--psm",
            "7",
            "-l",
            "eng",
        ]
    )
    return normalize_ocr_text(result.stdout)


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(character.isascii() and not character.isspace() for character in text)
    visible_chars = sum(not character.isspace() for character in text)
    if visible_chars == 0:
        return 0.0
    return ascii_chars / visible_chars


def normalize_english_text(text: str) -> str:
    cleaned = normalize_ocr_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_chinese_text(text: str) -> str:
    cleaned = normalize_ocr_text(text)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip("，。！？：；、“”‘’… ")


def combine_subtitle_candidates(
    candidates: list[tuple[float, float, float, str]],
    piece_joiner: str,
    line_joiner: str,
    y_tolerance: float,
) -> str:
    if not candidates:
        return ""

    grouped_lines: list[dict[str, object]] = []
    for average_y, average_x, score, text in sorted(candidates, key=lambda item: (item[0], item[1], -item[2], len(item[3]))):
        if not grouped_lines or abs(average_y - float(grouped_lines[-1]["average_y"])) > y_tolerance:
            grouped_lines.append({"average_y": average_y, "parts": [(average_x, score, text)]})
            continue
        grouped_lines[-1]["parts"].append((average_x, score, text))

    combined_lines: list[str] = []
    for line in grouped_lines:
        parts = sorted(line["parts"], key=lambda item: (item[0], -item[1], len(item[2])))
        merged_parts: list[str] = []
        for _average_x, _score, text in parts:
            if not text:
                continue
            if any(existing == text or text in existing for existing in merged_parts):
                continue
            merged_parts = [existing for existing in merged_parts if existing not in text]
            merged_parts.append(text)
        line_text = piece_joiner.join(merged_parts).strip()
        if line_text and (not combined_lines or combined_lines[-1] != line_text):
            combined_lines.append(line_text)
    return line_joiner.join(combined_lines)


def choose_subtitle_lines(result) -> tuple[str, str]:
    if not result:
        return "", ""
    frame_height = max(point[1] for box, _text, _score in result for point in box)
    bottom_threshold = frame_height * 0.82
    y_tolerance = max(frame_height * 0.035, 18.0)
    english_candidates: list[tuple[float, float, float, str]] = []
    chinese_candidates: list[tuple[float, float, float, str]] = []
    for box, raw_text, score in result:
        average_y = sum(point[1] for point in box) / len(box)
        average_x = sum(point[0] for point in box) / len(box)
        if average_y < bottom_threshold:
            continue
        text = str(raw_text).strip()
        if not text:
            continue
        if contains_cjk(text) and chinese_ratio(text) >= 0.35:
            chinese_text = normalize_chinese_text(text)
            if len(chinese_text) >= DEFAULT_MIN_CHINESE_CHARS:
                chinese_candidates.append((average_y, average_x, float(score), chinese_text))
            continue
        english_text = normalize_english_text(text)
        if ascii_ratio(english_text) >= 0.65 and len(english_text) >= 6:
            english_candidates.append((average_y, average_x, float(score), english_text))
    english_text = combine_subtitle_candidates(english_candidates, piece_joiner=" ", line_joiner=" ", y_tolerance=y_tolerance)
    chinese_text = combine_subtitle_candidates(chinese_candidates, piece_joiner="", line_joiner="", y_tolerance=y_tolerance)
    return english_text, chinese_text


def run_rapidocr(frame_path: Path, engine: RapidOCR) -> str:
    result, _ = engine(str(frame_path))
    if not result:
        return ""
    candidates: list[tuple[float, float, str]] = []
    for box, raw_text, score in result:
        text = normalize_ocr_text(str(raw_text))
        if not text or ascii_ratio(text) < 0.7:
            continue
        average_y = sum(point[1] for point in box) / len(box)
        candidates.append((average_y, float(score), text))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])))
    return candidates[-1][2]


def run_rapidocr_bilingual(frame_path: Path, engine: RapidOCR) -> tuple[str, str]:
    result, _ = engine(str(frame_path))
    return choose_subtitle_lines(result)


def collect_samples(frame_paths: list[Path], fps: float, min_chars: int, engine_name: str) -> list[OcrSample]:
    samples: list[OcrSample] = []
    frame_step = 1.0 / fps
    min_chinese_chars = min(max(1, min_chars), DEFAULT_MIN_CHINESE_CHARS)
    rapidocr_engine = RapidOCR() if engine_name == "rapidocr" else None
    for index, frame_path in enumerate(frame_paths):
        if rapidocr_engine is not None:
            english_text, chinese_text = run_rapidocr_bilingual(frame_path, rapidocr_engine)
        else:
            english_text = run_tesseract_ocr(frame_path)
            chinese_text = ""
        if len(english_text) < min_chars and len(chinese_text) < min_chinese_chars:
            continue
        samples.append(OcrSample(time=index * frame_step, english_text=english_text, chinese_text=chinese_text))
    return samples


def text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def merge_samples(samples: list[OcrSample], fps: float, similarity_threshold: float) -> list[SubtitleBlock]:
    if not samples:
        return []
    frame_step = 1.0 / fps
    blocks: list[SubtitleBlock] = [
        SubtitleBlock(
            start=samples[0].time,
            end=samples[0].time + frame_step,
            english_text=samples[0].english_text,
            chinese_text=samples[0].chinese_text,
        )
    ]
    for sample in samples[1:]:
        current = blocks[-1]
        english_similarity = text_similarity(current.english_text, sample.english_text) if current.english_text and sample.english_text else 0.0
        chinese_similarity = text_similarity(current.chinese_text, sample.chinese_text) if current.chinese_text and sample.chinese_text else 0.0
        if max(english_similarity, chinese_similarity) >= similarity_threshold:
            if len(sample.english_text) > len(current.english_text):
                current.english_text = sample.english_text
            if len(sample.chinese_text) > len(current.chinese_text):
                current.chinese_text = sample.chinese_text
            current.end = sample.time + frame_step
            continue
        blocks.append(
            SubtitleBlock(
                start=sample.time,
                end=sample.time + frame_step,
                english_text=sample.english_text,
                chinese_text=sample.chinese_text,
            )
        )
    return blocks


def looks_like_title_card(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    word_count = len(normalized.split())
    has_sentence_punctuation = any(marker in normalized for marker in ".,?!，。！？")
    if any(keyword in lowered for keyword in TITLE_KEYWORDS):
        return True
    if word_count <= 6 and not has_sentence_punctuation:
        title_case_words = [word for word in normalized.split() if word[:1].isupper()]
        if len(title_case_words) >= max(1, word_count - 1):
            return True
    return False


def is_non_dialogue_block(block: SubtitleBlock, total_duration: float) -> bool:
    text = block.text
    if not text:
        return True
    near_edge = block.start < 45 or block.end > total_duration - 45
    if near_edge and looks_like_title_card(text):
        return True
    lowered = text.lower()
    if "subtitle" in lowered or "caption" in lowered:
        return True
    return False


def filter_blocks(blocks: list[SubtitleBlock], total_duration: float) -> list[SubtitleBlock]:
    return [block for block in blocks if not is_non_dialogue_block(block, total_duration)]


def extract_subtitle_blocks(
    input_video: Path,
    workdir: Path,
    fps: float,
    min_chars: int,
    similarity: float,
    engine_name: str,
    crop_x: float = DEFAULT_CROP_X,
    crop_y: float = DEFAULT_CROP_Y,
    crop_w: float = DEFAULT_CROP_W,
    crop_h: float = DEFAULT_CROP_H,
) -> tuple[list[OcrSample], list[SubtitleBlock]]:
    args = argparse.Namespace(
        fps=fps,
        min_chars=min_chars,
        similarity=similarity,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        engine=engine_name,
    )
    frames_dir = ensure_dir(workdir / "frames")
    frame_paths = extract_frames(input_video, frames_dir, args)
    samples = collect_samples(frame_paths, fps, min_chars, engine_name)
    blocks = merge_samples(samples, fps, similarity)
    total_duration = float(run_command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_video)]).stdout.strip())
    return samples, filter_blocks(blocks, total_duration)


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(blocks: list[SubtitleBlock], output_path: Path, language: str = "english") -> Path:
    lines: list[str] = []
    for index, block in enumerate(blocks, start=1):
        if language == "chinese":
            text = block.chinese_text
        elif language == "bilingual":
            pieces = [piece for piece in [block.chinese_text, block.english_text] if piece]
            text = "\n".join(pieces)
        elif language == "auto":
            text = block.text
        else:
            text = block.english_text or block.text
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{srt_timestamp(block.start)} --> {srt_timestamp(block.end)}",
                text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_debug_json(samples: list[OcrSample], blocks: list[SubtitleBlock], output_path: Path) -> Path:
    payload = {
        "samples": [asdict(sample) for sample in samples],
        "blocks": [asdict(block) for block in blocks],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    require_binary("ffmpeg")
    if args.engine == "tesseract":
        require_binary("tesseract")

    input_video = args.input_video.resolve()
    output_srt = (args.output or default_output_path(input_video)).resolve()

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="ocr_subtitles_")
        workdir = Path(temp_dir.name)
    else:
        workdir = ensure_dir(args.workdir.resolve())

    frames_dir = ensure_dir(workdir / "frames")
    try:
        frame_paths = extract_frames(input_video, frames_dir, args)
        samples = collect_samples(frame_paths, args.fps, args.min_chars, args.engine)
        blocks = filter_blocks(merge_samples(samples, args.fps, args.similarity), float(run_command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_video)]).stdout.strip()))
        debug_json = write_debug_json(samples, blocks, workdir / "ocr_debug.json")
        write_srt(blocks, output_srt, args.language)
        print(output_srt)
        print(debug_json)
        print(len(blocks))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()