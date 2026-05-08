#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_ASR_MODEL = "small"
DEFAULT_MDX_MODEL = "UVR_MDXNET_KARA_2.onnx"
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
DEFAULT_HY_MODEL = "tencent/HY-MT1.5-1.8B"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / ".video-translator" / "models"


def resolve_hy_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available() and mps_backend.is_built():
        return "mps"
    return "cpu"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_cache_dir() -> Path:
    return ensure_dir(DEFAULT_CACHE_ROOT / "huggingface")


def whisper_cache_dir() -> Path:
    return ensure_dir(DEFAULT_CACHE_ROOT / "faster-whisper")


def mdx_cache_dir() -> Path:
    return ensure_dir(DEFAULT_CACHE_ROOT / "mdx")


def demucs_cache_dir() -> Path:
    return ensure_dir(DEFAULT_CACHE_ROOT / "demucs")


def prepare_hy_model(model_id: str, requested_device: str) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = hf_cache_dir()
    resolved_device = resolve_hy_device(requested_device)
    print(f"[init] Preparing HY-MT model cache: {model_id} (device={resolved_device})")
    snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
    AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    dtype = torch.float32 if resolved_device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    )
    del model


def prepare_asr_model(model_name: str) -> None:
    from faster_whisper.utils import download_model

    print(f"[init] Preparing faster-whisper model cache: {model_name}")
    download_model(model_name, output_dir=str(whisper_cache_dir()), cache_dir=str(whisper_cache_dir()))


def prepare_mdx_model(model_filename: str) -> None:
    from audio_separator.separator import Separator

    print(f"[init] Preparing MDX model cache: {model_filename}")
    separator = Separator(
        output_dir=str(ensure_dir(Path(__file__).resolve().parent / "artifacts" / "model_prewarm" / "mdx")),
        output_format="WAV",
        model_file_dir=str(mdx_cache_dir()),
        sample_rate=44100,
        use_soundfile=True,
    )
    separator.load_model(model_filename=model_filename)


def prepare_demucs_model(model_name: str) -> None:
    from demucs.pretrained import get_model

    print(f"[init] Preparing Demucs model cache: {model_name}")
    previous_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(demucs_cache_dir())
    try:
        get_model(model_name)
    finally:
        if previous_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous_torch_home


def prepare_rapidocr() -> None:
    from rapidocr_onnxruntime import RapidOCR

    print("[init] Verifying RapidOCR packaged models")
    RapidOCR()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download runtime models needed by video_translator.")
    parser.add_argument("--hy-model", default=DEFAULT_HY_MODEL)
    parser.add_argument("--hy-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--mdx-model", default=DEFAULT_MDX_MODEL)
    parser.add_argument("--demucs-model", default=DEFAULT_DEMUCS_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(DEFAULT_CACHE_ROOT)
    prepare_hy_model(args.hy_model, args.hy_device)
    prepare_asr_model(args.asr_model)
    prepare_mdx_model(args.mdx_model)
    prepare_demucs_model(args.demucs_model)
    prepare_rapidocr()
    print(f"[init] Runtime assets prepared under {DEFAULT_CACHE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())