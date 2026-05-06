# Video Translator

`video_translator` 是一个纯离线的视频翻译与中文配音命令行工具，面向“本地完成字幕提取、翻译、配音和混音”的工作流。

默认面向两类用户：

- 普通用户：安装一次后，只传一个视频路径
- 进阶用户：按需覆盖字幕、分离、翻译、配音和混音参数

当前产品只保留两条核心能力：

- 本地 `HY-MT` 翻译
- 本地 `GPT-SoVITS` 配音

产品目标是不依赖 Google Translate、Edge TTS 或其他在线 SaaS 能力；运行时不主动访问公网翻译或公网 TTS 服务。

## 特性

- OCR-first 字幕获取：优先内嵌字幕，其次 OCR，最后回退 ASR
- 全链路本地运行：翻译使用本地 `HY-MT`，配音使用本地 `GPT-SoVITS` HTTP API
- 支持 `MDX-Net`、`Demucs`、`ffmpeg` 多种分离策略
- 自动保留背景声并重新混音
- 输出 `source.srt`、`translated.srt`、`manifest.json`、`audio_audit.log`
- 对异常偏静的 TTS 片段自动重试 2 次

## 当前架构

- 字幕获取：内嵌字幕提取 -> OCR -> ASR 回退
- 翻译：本地 `tencent/HY-MT1.5-1.8B`
- 配音：本地 `GPT-SoVITS` API
- 混音：`ffmpeg` 将背景轨与配音轨重新封装回原视频

更完整的架构说明见 [DESIGN.MD](./DESIGN.MD)。

## 快速开始

对普通用户，推荐流程只有两步。

### 第一步：执行一键安装

```bash
./install.sh
```

安装脚本会处理以下事情：

- 创建本地虚拟环境 `.venv`
- 安装 Python 依赖
- 安装当前项目
- 检查 `ffmpeg` 和 `ffprobe`
- 自动执行 `./init.sh`

初始化脚本会继续处理：

- 自动检测已有 `GPT-SoVITS` 服务仓库
- 若本机没有仓库，默认克隆到同级目录 `../GPT-SoVITS`
- 准备或复用 `GPT-SoVITS` 的 Python 环境
- 预下载本地翻译、ASR、MDX、Demucs 所需模型
- 把这些运行时模型统一缓存到 `.video-translator/models/`
- 通过和运行时一致的 `transformers` 加载路径预热 `HY-MT` 完整权重，避免首次翻译时再下载大体积 checkpoint
- 准备自动音色克隆所需的默认参考配置
- 生成 `video_translator.defaults.json` 运行默认配置
- 生成 `gpt_sovits.local.json` 本地服务配置
- 启动本地 `GPT-SoVITS` 服务

如果你只想先装主程序、不初始化服务，也可以：

```bash
./install.sh --skip-init
./init.sh
```

### 第二步：只传入视频路径

```bash
./video-translator /path/to/video.mov
```

如果你更喜欢直接调用虚拟环境里的 CLI，也可以：

```bash
./.venv/bin/video-translator /path/to/video.mov
```

安装脚本生成的默认配置会自动提供：

- 目标语言：中文
- GPT-SoVITS 本地地址
- 自动单旁白音色克隆模式
- 保守回退男声/女声参考音频路径

默认简单模式下，程序会优先从当前输入视频里自动挑选一段较干净的单旁白原声音频，并直接用该片段的原文做 GPT-SoVITS 参考文本。

如果自动抽取失败，程序会进入保守模式：

- 优先按检测到的说话人音高，在 `male.wav` 和 `female.mp3` 之间选择更接近的一条中文参考音频
- 若这两个文件缺失或转写失败，再回退到仓库内置的固定参考音频

## 环境准备

- Python `3.10+`
- 首次安装阶段需要可访问对应模型源，用于下载本地翻译、ASR 与分离模型

`install.sh` 和 `init.sh` 会尽可能处理本地环境；在 macOS 且已安装 Homebrew 的情况下，它们会自动补齐 `ffmpeg`，并在缺失时安装 `python@3.11` 供 `GPT-SoVITS` 使用。

## GPT-SoVITS 前置条件

本仓库不把 `GPT-SoVITS` 推理代码直接 vendor 进当前仓库，但现在会在 `init.sh` 中自动检测或克隆官方仓库，并把服务启动准备好。默认 HTTP 地址仍然是：

```text
http://127.0.0.1:9880
```

对于普通用户，这三项默认由 `init.sh` 生成并写入 `video_translator.defaults.json`；只有进阶用户才需要手动覆盖：

- 一段参考音频 `--gpt-sovits-ref-audio`
- 该参考音频对应的文本 `--gpt-sovits-prompt-text`
- 参考音频语言 `--gpt-sovits-prompt-lang`

如果你要启用保守模式里的男女回退参考，请把你准备好的两个文件放在项目根目录：

- `./female.mp3`
- `./male.wav`

## 用法

### 小白模式

安装完成后，最简单的单文件用法是：

```bash
./video-translator ./video_test.mov
```

也支持直接把视频路径作为唯一参数传给 CLI：

```bash
./.venv/bin/video-translator ./video_test.mov
```

在这个模式下，不需要手动指定参考音频。程序会优先尝试“原视频单旁白音色 -> 中文配音”，只有在自动抽取失败时才回退到 `female.mp3` / `male.wav` 或仓库内置参考音色。

### 进阶模式

当你想替换默认参考音频或其他行为时，再显式传参数。

最小单文件显式参数示例：

```bash
video-translator \
	--video ./video_test.mov \
	--gpt-sovits-ref-audio ./ref_voice.wav \
	--gpt-sovits-prompt-lang zh \
	--gpt-sovits-prompt-text "这是参考音频对应的文本" \
	--keep-workdir
```

批量处理目录：

```bash
video-translator \
	--input-dir ./videos \
	--gpt-sovits-ref-audio ./ref_voice.wav \
	--gpt-sovits-prompt-lang zh \
	--gpt-sovits-prompt-text "这是参考音频对应的文本" \
	--keep-workdir
```

批量模式默认输出到输入目录下的 `translated_videos/`。

## 常用参数

普通用户通常只需要：

- `video-translator /path/to/video.mov`

只有在需要自定义时，再使用下面这些参数：

- `--video path/to/video.mov`：处理单个视频
- `--input-dir path/to/videos`：批量处理目录
- `--output output.mov`：指定单文件输出路径
- `--output output_dir/`：批量模式指定输出目录
- `--workdir path/to/workdir`：指定中间产物目录
- `--keep-workdir`：保留中间文件，方便复查
- `--subtitle-source auto|ocr|asr`：字幕来源策略
- `--separator auto|mdx|demucs|ffmpeg|none`：分离策略
- `--mdx-model UVR_MDXNET_KARA_2.onnx`：指定 MDX 模型
- `--asr-model tiny|small|medium|large-v3`：ASR 模型大小
- `--chunk-seconds 300`：长视频 ASR 分块时长
- `--ocr-fps 1.0`：OCR 抽帧频率
- `--hy-device auto|cpu|mps|cuda`：HY-MT 运行设备
- `--hy-model tencent/HY-MT1.5-1.8B`：HY-MT 模型 ID
- `--gpt-sovits-url http://127.0.0.1:9880`：本地 GPT-SoVITS API
- `--gpt-sovits-ref-audio ./ref.wav`：参考音频
- `--gpt-sovits-prompt-lang zh`：参考音频语言
- `--gpt-sovits-prompt-text "..."`：参考音频对应文本
- `--gpt-sovits-text-lang zh`：目标文本语言
- `--gpt-sovits-text-split-method cut5`：GPT-SoVITS 切句策略
- `--gpt-sovits-speed 1.0`：GPT-SoVITS 基础语速
- `--dub-volume 0.78`：配音混音权重
- `--background-volume 1.18`：背景混音权重

## 示例

### 最简单示例

```bash
./install.sh
./video-translator ./samples/video_test_5.mov
```

### 显式指定参考音频

使用中文参考音频生成中文配音：

```bash
video-translator \
	--video ./samples/video_test_5.mov \
	--output ./samples/video_test_5_gpt_sovits.mov \
	--workdir ./artifacts/gpt_sovits_video_test_5 \
	--keep-workdir \
	--separator mdx \
	--gpt-sovits-url http://127.0.0.1:9880 \
	--gpt-sovits-ref-audio ./artifacts/gpt_sovits_ref.wav \
	--gpt-sovits-prompt-lang zh \
	--gpt-sovits-prompt-text "这是参考音频对应的文本" \
	--gpt-sovits-text-lang zh
```

使用已有中文成片中的旁白抽取参考音频后再克隆：

```bash
ffmpeg -y -ss 16.3 -to 26.3 -i ./samples/translated_videos/video_test_5_zh_CN.mov -vn -ac 1 -ar 32000 ./artifacts/ref_from_translated_video.wav

video-translator \
	--video ./samples/video_test_5.mov \
	--gpt-sovits-ref-audio ./artifacts/ref_from_translated_video.wav \
	--gpt-sovits-prompt-lang zh \
	--gpt-sovits-prompt-text "毫无疑问，有史以来最杰出的动物之一，也是最为著名的生物之一，就是恐龙。" \
	--gpt-sovits-text-lang zh \
	--keep-workdir
```

## 输出内容

每次运行会生成：

- 最终配音视频
- `source.srt`
- `translated.srt`
- `manifest.json`
- `audio_audit.log`
- `tts/`、`timeline/`、`ocr/` 等中间目录

其中 `audio_audit.log` 用于人工审核：

- 每段 TTS 是否生成
- 原始音频和拟合后音频路径
- 每段时长与音量指标
- 是否命中 `very-quiet` 自动重试
- 最终 `dub_track`、背景轨和输出视频时长

## 说明

- `video_translator.defaults.json` 是本地运行默认配置，主要用于“只传视频路径”的简化模式。
- `gpt_sovits.local.json` 是本地 GPT-SoVITS 服务启动配置，由 `init.sh` 生成。
- `.video-translator/models/` 是本地运行模型缓存目录，`init.sh` 会提前准备 `HY-MT`、`faster-whisper`、`MDX-Net`、`Demucs` 所需资产。
- `GPT-SoVITS` 虽通过本地 HTTP API 调用，但服务应部署在本机或内网环境中。
- `video-translator` 在检测到本地服务未启动时，会优先尝试自动拉起 `GPT-SoVITS`。
- `ffmpeg` 降级方案的人声分离效果不如 `MDX-Net` 或 `Demucs`。

## 仓库内容

- `video_translator.py`：主 CLI 和视频翻译主流程
- `ocr_subtitles.py`：OCR 字幕提取
- `pyproject.toml`：包元数据
- `requirements.txt`：运行依赖
- `DESIGN.MD`：架构设计说明
- `README.md`：用户使用说明

## License

本项目使用 MIT License，详见 `LICENSE`。