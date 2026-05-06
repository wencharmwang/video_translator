# Design

## 目标

`video_translator` 当前的产品目标是提供一个纯离线的视频翻译与中文配音流水线.
系统输出需要兼顾两类用户：

- 直接得到可审核的视频成片
- 能够在工作目录中复查每一个中间环节

同时要兼顾两种使用模式：

- 简单模式：用户只传一个视频路径
- 高级模式：用户按需覆盖默认配置和流水线参数

## 非目标

- 不把第三方在线翻译或在线 TTS 作为标准能力
- 不把 GPT-SoVITS 推理代码直接 vendor 到本仓库
- 不负责模型分发平台或模型管理 UI

## 架构总览

端到端处理链路如下：

1. 输入视频
2. 音频提取
3. 人声/背景分离
4. 字幕获取
5. 文本翻译
6. GPT-SoVITS 逐段配音
7. 时长拟合与时间线拼接
8. 与背景声混音
9. 输出成片与审计日志

## 模块划分

### 1. 输入与调度

主入口在 `video_translator.py`。

职责：

- 解析单文件或批量目录输入
- 解析本地翻译与 GPT-SoVITS 参数
- 加载运行默认配置
- 管理工作目录与输出目录

当前输入调度包含一层“运行默认配置”机制：

- 优先读取安装脚本生成的 `video_translator.defaults.json`
- 若未显式传高级参数，则自动回填 GPT-SoVITS 默认参考音频与提示词
- 让普通用户可以只输入一个视频路径

### 2. 音频提取

从原视频抽取：

- `full_mix.wav`
- `speech_input.wav`

其中：

- `full_mix.wav` 用于后续背景分离
- `speech_input.wav` 用于人声相关处理

### 3. 人声分离

支持以下分离策略：

- `mdx`
- `demucs`
- `ffmpeg`
- `none`

默认策略 `auto` 会优先尝试更高质量方案。

输出：

- 前景人声轨
- 背景轨

### 4. 字幕获取

字幕获取遵循以下优先级：

1. 内嵌字幕轨
2. OCR
3. ASR

目的：

- 优先复用已有字幕，降低识别误差
- 当 OCR 无结果时自动回退 ASR

### 5. 翻译

当前只保留一个翻译后端：本地 `HY-MT`。

职责：

- 将非中文字幕翻译为目标语言
- 保持批量翻译与本地模型运行能力

设计取舍：

- 去掉 Google Translate，保证产品默认不依赖公网翻译
- 保留 `translate_timeout` 等兼容参数，但核心路径不依赖外网请求

### 6. GPT-SoVITS 配音

当前只保留一个配音后端：本地 `GPT-SoVITS` HTTP API。

输入参数：

- `gpt_sovits_url`
- `gpt_sovits_ref_audio`
- `gpt_sovits_prompt_text`
- `gpt_sovits_prompt_lang`
- `gpt_sovits_text_lang`
- `gpt_sovits_text_split_method`
- `gpt_sovits_speed`

设计取舍：

- 本仓库只负责调用本地 API，不持有 GPT-SoVITS 服务实现
- 音色由参考音频决定，而不是预设 voice 名称
- 默认参考音频和提示词由安装脚本生成，避免普通用户首次运行时被复杂参数阻塞

### 6.1 安装、初始化与服务拉起

当前用户侧流程拆成三层：`install.sh -> init.sh -> video-translator`。

`install.sh` 职责：

- 创建 `.venv`
- 安装依赖与本项目
- 检查系统二进制依赖
- 默认直接调用 `init.sh`

`init.sh` 职责：

- 检测已有 `GPT-SoVITS` 仓库
- 若缺失则自动克隆官方仓库到同级目录
- 复用或创建 `GPT-SoVITS` 独立 Python 环境
- 安装 `GPT-SoVITS` 依赖
- 调用 `prepare_runtime_assets.py` 预下载运行期模型
- 将 `HY-MT`、`faster-whisper`、`MDX-Net`、`Demucs` 统一缓存到 `.video-translator/models/`
- 对 `HY-MT` 额外走一次与运行时相同的 `transformers.from_pretrained(...)` 预热路径，确保完整权重在初始化阶段落盘，而不是首个翻译任务时再触发下载
- 写入自动单旁白参考模式与男女回退参考路径
- 写入 `video_translator.defaults.json`
- 写入 `gpt_sovits.local.json`
- 启动本地 `GPT-SoVITS` 服务

`video-translator` 职责：

- 读取默认配置
- 若检测到本地 `GPT-SoVITS` 服务未启动，则优先尝试自动拉起
- 运行时优先复用本地统一模型缓存，而不是临时落到用户家目录默认缓存
- 在简单模式下，优先从当前输入视频的单旁白片段中自动提取原声音色参考
- 若自动提取失败，则按音高估计在 `male.wav` / `female.mp3` 之间选择保守回退音色
- 之后再进入翻译、配音与混音主流程

这样可以把“环境准备”“服务准备”“业务处理”拆开，但对普通用户仍然保持一次安装后只传视频路径的体验。

### 7. 时长拟合与时间线

逐段 TTS 结果会经过两步处理：

1. 计算目标槽位长度
2. 使用 `ffmpeg atempo + apad + atrim` 拟合到目标时长

随后生成：

- 每段 `segment_xxxx.wav`
- 每段 `segment_xxxx_fit.wav`
- `dub_track.ffconcat`
- `dub_track.wav`

### 8. 静音异常重试

这是当前架构中新增的稳定性设计。

问题背景：

- GPT-SoVITS 有时会返回“文件存在但几乎静音”的伪成功结果
- 如果只按“文件是否存在”判断，会把问题带入最终成片

当前机制：

- 对每段原始 TTS 做 `volumedetect`
- 若命中 `very-quiet` 或 `mostly-silent`
- 自动重试 2 次
- 选择更好的那次结果落为正式片段

这样可以在不重跑整条链路的情况下，提高异常片段修复率。

### 9. 混音与封装

最终混音由 `ffmpeg` 完成：

- 背景轨单独调节音量
- 配音轨先做 `loudnorm` 再调节音量
- 两轨 `amix`
- 保持原视频流，重新封装输出

### 10. 审计与可复查性

每次运行都会产出：

- `manifest.json`
- `audio_audit.log`

其中 `audio_audit.log` 记录：

- 每段 TTS 的生成状态
- 原始与拟合音频路径
- 每段音量统计
- 是否命中静音重试
- 混音输入输出轨时长

这份日志的目标是把“问题发生在 TTS 生成、时长拟合，还是最终混音”区分开。

## 关键设计原则

### 离线优先

产品默认能力不依赖公网翻译或公网 TTS。

### 可复跑

保留中间文件和 manifest，允许用户只重做某个阶段或某些异常片段。

### 可审核

通过 `source.srt`、`translated.srt`、`audio_audit.log` 让人工审核能快速定位问题。

### 松耦合

将本仓库与 GPT-SoVITS 服务进程解耦，降低模型和推理栈变更对主项目的影响。

## 当前主要文件

- `video_translator.py`：主流水线实现
- `ocr_subtitles.py`：OCR 字幕提取和过滤
- `README.md`：面向用户的使用文档
- `DESIGN.MD`：面向维护者的架构说明

## 后续可演进方向

- 只重跑异常静音片段的正式 CLI 子命令
- 参考音频片段管理和别名化
- 审计日志导出为结构化 JSON
- 混音策略分档，例如“旁白更贴背景”与“旁白更突出”