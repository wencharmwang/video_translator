# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning.

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
