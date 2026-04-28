# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

System dependencies:

- `ffmpeg`
- `ffprobe`

## Local Checks

```bash
video-translator --help
ocr-subtitles --help
python -m build
```

## Versioning

- The canonical project version lives in `video_translator.__version__`.
- Update `CHANGELOG.md` together with version changes.
- Use Semantic Versioning for releases.

## Pull Requests

- Keep changes focused and avoid unrelated refactors.
- Update README and changelog when public CLI behavior changes.
- Include a minimal validation note for packaging or media pipeline changes.
