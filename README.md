# Video Translator

`video_translator` 是一个面向本地运行的命令行视频翻译工具，重点解决“已有字幕的视频快速转中文配音”这类场景。

它采用 OCR-first 流程：优先提取烧录字幕；如果视频里已经有中文字幕，直接保留中文进入 TTS；如果只有英文字幕，则先翻译后配音；如果没有可用字幕，再自动回退到 ASR。

## Highlights

- OCR-first 字幕获取，适合纪录片、访谈、解说类视频
- 中文字幕直通配音，避免重复翻译带来的误差
- 英文字幕自动翻译，OCR 失败时自动回退到 faster-whisper
- Edge TTS 配音，支持语速、音高、音量调节
- 输出 `source.srt`、`translated.srt`、`manifest.json` 便于排查和复跑
- 保留背景音并重新混音，尽量维持原视频氛围

## 功能

- 从视频中抽取音频
- 优先提取烧录字幕，并做简单过滤，剔除片头片尾 title card 和明显非对白字幕
- 优先使用 Demucs 做人声分离，失败时自动降级到 ffmpeg 方案
- 使用 faster-whisper 做本地语音识别
- 使用 Google Translate 免费通道翻译文本
- 使用 Edge TTS 合成中文配音
- 将新配音与背景声重新混合并封装回视频
- 按固定时长分块识别，降低长视频处理的内存压力
- 导出 `source.srt` 和 `translated.srt`，便于人工检查字幕边界与时间轴

## 环境准备

- 系统安装 ffmpeg 和 ffprobe
- 建议使用 Python 3.10+

## 安装

从当前仓库安装：

```bash
pip install .
```

本地开发安装：

```bash
pip install -e .
```

直接从 GitHub 安装：

```bash
pip install git+ssh://git@github.com/wencharmwang/video_translator.git
```

发布到 PyPI 之后：

```bash
pip install video-translator
```

如果只想按旧方式装运行依赖，也可以继续使用：

```bash
pip install -r requirements.txt
```

## 用法

```bash
video-translator --video ./video_test.mov --target-language zh-CN --voice zh-CN-XiaoyiNeural --keep-workdir
```

OCR 提取工具也会一起安装：

```bash
ocr-subtitles ./video_test.mov --fps 1.0 --output ./video_test.ocr.srt
```

常用参数：

- `--separator auto|demucs|ffmpeg|none`：选择分离策略
- `--video path/to/video.mov`：指定输入视频路径，推荐作为标准调用方式
- `--chunk-seconds 300`：按 300 秒分块转录
- `--subtitle-source auto|ocr|asr`：默认优先 OCR，失败时回退 ASR
- `--ocr-fps 1.0`：OCR 每秒抽帧数，越高越细，越慢
- `--asr-model small`：切换 faster-whisper 模型大小
- `--output output.mov`：指定输出文件
- `--keep-workdir`：保留中间文件，方便人工检查和复跑
- `--edge-rate -5%`：调整 Edge TTS 语速
- `--edge-pitch -12Hz`：调整 Edge TTS 音高，缓解偏尖的感觉
- `--edge-volume +0%`：调整 Edge TTS 音量
- `--translate-timeout 30`：限制单次翻译 HTTP 请求超时，避免长时间挂住

推荐中文 Edge 音色：

- `zh-CN-XiaoyiNeural`：默认，整体比 `Xiaoxiao` 更柔和一些
- `zh-CN-YunyangNeural`：偏男声，纪录片类旁白通常更稳
- `zh-CN-YunjianNeural`：男声，存在感更强

如果觉得声音偏尖、偏快，建议先试：

```bash
video-translator --video ./video_test.mov --target-language zh-CN --voice zh-CN-XiaoyiNeural --edge-pitch -12Hz --edge-rate -5% --separator ffmpeg --keep-workdir
```

## 开发与版本管理

- 版本号单一来源：`video_translator.__version__`
- 变更记录：`CHANGELOG.md`
- 发布建议遵循 Semantic Versioning

构建发布产物：

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine upload dist/*
```

发布前建议至少做下面三项检查：

- `video-translator --help`
- `ocr-subtitles --help`
- `python -m build`

## 仓库内容

- `video_translator.py`：主 CLI 入口和视频翻译主流程
- `ocr_subtitles.py`：OCR 字幕提取 CLI 和复用逻辑
- `pyproject.toml`：Python 包元数据与 console scripts
- `CHANGELOG.md`：版本变更记录
- `CONTRIBUTING.md`：开发与版本管理约定

## 示例媒体

仓库默认不提交大体积测试视频。

- 本地开发可以自行准备 `video_test.mov` 或其他样例素材
- 如果需要对外提供演示素材，建议放到 GitHub Releases、对象存储或 Git LFS，而不是直接放进 Git 仓库历史

## 输出

- 翻译后的视频文件，例如 `video_test_zh_CN.mov`
- 中间目录中的 `manifest.json`，记录识别、翻译和 TTS 结果
- 中间目录中的 `source.srt` 和 `translated.srt`，分别保存原文字幕块和译文字幕块
- 如果走 OCR，还会在工作目录的 `ocr/` 子目录下保留 OCR 字幕和调试信息
- 任务失败时会在工作目录写入 `error.log`

## 说明

- Demucs 首次运行会下载模型，耗时较长。
- `ffmpeg` 降级方案无法像 Demucs 那样干净地保留背景声，只作为兜底。
- Edge TTS 和 Google Translate 依赖网络访问。
- OCR 主流程适合已有烧录字幕的视频；如果 OCR 没抓到可用字幕，会自动回退到 ASR。
- 仓库暂未附带开源许可证文件；如果要公开分发，建议在发布前补充最终许可证。