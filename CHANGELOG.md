# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning.

## [0.1.4] - 2026-04-28

### Changed

- Added configurable `--asr-device`, `--asr-compute-type`, and `--demucs-device` flags so runtime hardware selection is no longer hardcoded.
- Added automatic hardware resolution that prefers CUDA when available, uses Apple MPS for Demucs on supported Macs, and falls back cleanly to CPU when GPU backends are unavailable.

## [0.1.3] - 2026-04-28

### Changed

- Added stage progress output for long-running ASR, translation, TTS, and Demucs separation steps so the CLI no longer appears stalled during long videos.
- Delayed importing OCR and faster-whisper dependencies until those stages are needed, which avoids the startup-time macOS AVFoundation duplicate class warnings on help and other lightweight CLI paths.

## [0.1.2] - 2026-04-28

### Changed

- Renamed the published PyPI distribution from `video-translator` to `video-translator-cli` to avoid a project name ownership conflict on PyPI.
- Kept the installed CLI commands as `video-translator` and `ocr-subtitles`, so existing command-line usage does not change.

## [0.1.1] - 2026-04-28

### Added

- Added an MIT `LICENSE` file for open source distribution.
- Added package license metadata so PyPI and built distributions include the project license.

### Changed

- Refreshed `README.md` with PyPI installation, release, and tagging instructions.
- Prepared the project for the first tagged release workflow based on the packaged CLI entry points.

## [0.1.0] - 2026-04-28

### Added

- Initial public CLI packaging with `pip install .`, editable installs, and console entry points.
- OCR-first subtitle acquisition with RapidOCR.
- Chinese-direct dubbing flow for videos that already contain Chinese subtitles.
- English subtitle translation flow with ASR fallback when OCR subtitles are unavailable.
- Edge TTS dubbing pipeline, subtitle export, workdir manifest export, and failure logging.

### Changed

- Standardized the public project name to `video_translator` / `video-translator`.
- Standardized the CLI input flag to `--video` while retaining backward-compatible positional invocation.
- Tuned default dub/background mix levels and long-segment timing behavior for more natural output.
