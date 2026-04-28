# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning.

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
