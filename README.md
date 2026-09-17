# Zoey Tool

[**English**](#english) | [**中文**](#chinese)

---

<a name="english"></a>

## English

Multi-functional ComfyUI custom nodes plugin for image/video processing, translation, prompt generation, and more.

### Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liangzoey/comfyui-ZoeyTool.git
cd comfyui-ZoeyTool
pip install -r requirements.txt
```

### Nodes

#### 🖼️ Image Processing

| Node | Description |
|------|-------------|
| **True Size Image Loader** | Load images from folder at original resolution, sequential batch output, natural sorting, alpha channel as mask output |
| **Batch Image Saver** | Save images + text files in batch with custom index, digit count, overwrite mode |
| **Mask Bounding Box Drawer** | Draw bounding boxes from mask with HTML5 color picker, opacity/fill/width controls, preset colors, light type + prompt output, behind-subject compositing, optional rembg background removal |
| **Batch Image Cropper** | Batch crop images (left/right/top/bottom), overwrite mode, filename preservation |
| **Light Handle Control** | Interactive lighting direction with draggable handle, circular gradient mask, lighting prompt output (16 light types), behind-subject compositing, multiple handle shapes |
| **Text Overlay** | Overlay text on images with custom or system fonts, adjustable size/color/position |
| **Multi-function Image Editor** | Flip, rotate, split, edge detect, blur, sharpen, threshold, color invert, grayscale, perspective warp, blend, stylize |
| **Image Edit Prompt Generator** | Auto-generate edit prompts for text/object/style/background editing, virtual try-on, object add/remove |
| **Frame Crop/Outpaint** | Interactive frame-based cropping and outpainting with fill color, real-time canvas preview |
| **Multi-layer Canvas** | 5-layer image compositing with drag reposition, scroll scale, rotation handle, and horizontal/vertical flip |

#### 🎥 Video Processing

| Node | Description |
|------|-------------|
| **Batch Video Loader** | Load video file list from directory, multi-format (mp4/avi/mov), natural sort, limit control |
| **Video Batch Processor** | Batch process videos (format, frame rate), auto-install dependencies, skip existing |
| **Smart Video Saver** | Save processed videos with date folders and metadata |

#### 📝 Text Tools

| Node | Description |
|------|-------------|
| **Pure Translator** | Multi-engine translation: Helsinki-NLP models / Baidu API / Google API. Supports ZH/EN/JA/KO |
| **Hunyuan Translator (HY-MT1.5)** | Tencent Hunyuan MT local deployment, 20+ languages, auto-download model cache, terminology & context support |
| **Wan2.2 Prompt Generator** | Cinematic video prompt generator with 16 control dimensions: subject, scene, action, lighting, composition, lens, style |

#### 🔧 Other

| Node | Description |
|------|-------------|
| **Multi File Batch Renamer** | Batch rename files with regex find/replace, natural sort, custom start index |
| **ZOEYTextEncodeQwenImageEditPlus** | Qwen image edit encoder with multi-reference images for Qwen2-VL |
| **VR 360° Preview** | 360° equirectangular panorama VR viewer powered by Pannellum |
| **System Monitor** | Live CPU/RAM/VRAM/GPU temperature & utilization dashboard via embedded HTTP server, with background VRAM auto-cleanup |
| **MiniMax H3 Reference-to-Video (@)** | Wraps official MiniMaxH3ReferenceToVideo with Seedance-style @ reference tags (@P1/@V1/@A1/@C1/@L1) auto-expanded to native tags, plus a storyboard director panel and a permanent global asset library |
| **Music Generation (YuE2)** | Whole YuE2 pipeline in ONE node: text-to-music, cover (reference song → melody → re-sing), AI lyric writing, voice conversion, LoRA. Type a one-line description, set a duration slider, press Run |

##### Music Generation Usage

One node replaces an 8-node graph. Everything — checkpoint loading, ABC scoring, music-token generation, sampling, decoding, saving — happens inside a single `execute()`; no wiring required.

**The duration control is honest.** `时长(秒)` is an **upper bound**, not a promise. Internally YuE2 turns it into a token budget (25 tokens/second), then silently shrinks that budget if your style/lyrics are long, and may finish early on its own. The real length only comes back from the model — so the node reads it back and prints each take's **actual** duration on the node panel. Expect output shorter than requested.

**Lyric length rule**: roughly **7 seconds per line**. A live `建议歌词 ≤ N 行` hint is drawn under the duration slider. Lyrics longer than the budget get truncated at the end.

| Control | Notes |
|---------|-------|
| `模式` | 文生音乐 (text-to-music) or 翻唱 (cover: transcribe a reference song's melody, then re-sing it with new style/lyrics) |
| `歌曲描述` | One line. Used when `AI 写词` is on and style/lyrics are empty |
| `时长(秒)` | 30–300 upper bound |
| `版本数` | 1–4 takes, seeds `base + i × 7919` |

**Style selectors** (plain backend COMBO inputs — dropdowns, no frontend code): `语言` / `人声` / `曲风` / `BPM` / `乐器1-3` / `制作质感`. They are composed server-side into the style string with the same formula as YuE Studio:

```
语言 + 人声 + 曲风 + BPM + 乐器 + 制作质感
→ Chinese, husky male vocal, folk, 78 BPM, fingerstyle acoustic guitar, harmonica, warm analog production
```

`风格提示词` is a manual override — fill it and it wins over the selectors. `歌词` uses `[Verse]`/`[Chorus]` tags; leave it empty with `AI 写词` on and the model writes it.

Resolution order: **manual `风格提示词` > `AI 写词` > the selectors.**

Remaining inputs: `AI 写词` (local llama.cpp + GGUF), `写词模型`, `参考音频`, `模型`, `YuE2 LoRA` (applied to both MODEL and CLIP — style/melody tokens come from the CLIP half), `输出格式`, `音色转换` (Demucs + Seed-VC) with `移调`/`转换步数`, `识别歌词文本` (Qwen3-ASR).

Sampling parameters (steps 32 / CFG 1.0 / `dpm_2` / `sgm_uniform`) are fixed to the values this pipeline is known to work with and are deliberately not exposed — CFG especially, since YuE2 does its CFG inside the text encoder.

This node is **backend-only**: no frontend JS.

**Outputs** land in `output/yue/<timestamp>/`: one FLAC per take (true length, never padded), plus `mp3/`, `vc/`, `meta.json`, `lyrics.txt`.

**When AI 写词 or 音色转换 are enabled**, the node serializes GPU use: the helper engine runs *entirely before* YuE2 loads or *entirely after* it unloads. Two GPU consumers are never resident at once.

> ⚠️ **Machine-specific absolute paths.** The external engines are referenced by absolute path and must be edited at the top of `nodes/zoey_yue2_studio.py` if they move: llama.cpp (`LLAMA_SERVER`, plus `CUDA_BIN` — without it on `PATH`, llama.cpp silently falls back to CPU, 10× slower with no error), ASR (`ASR_PYTHON`/`ASR_SCRIPT`/`ASR_MODEL_ROOT`), voice conversion (`VC_PYTHON`/`VC_SCRIPT`/`SEEDVC_DIR`).

> ⚠️ **Caching.** Like the rest of this pack, the node defines no `IS_CHANGED`. With `seed` on `control_after_generate` (the default) pressing Run again regenerates; but if you **pin the seed**, ComfyUI returns cached audio and silently **skips** the AI-writing and voice-conversion subprocesses. Bump the seed, or change any input, to force a real re-run.

##### MiniMax H3 Usage

Wraps the official `MiniMaxH3ReferenceToVideo` (text + image/video/audio references → video). Type `@` in the prompt to open a media picker with thumbnails; hover any `@` thumbnail for a large preview.

**@ Reference tags** — auto-expand to native `<Picture N>/<Video N>/<Audio N>`:
| Tag | Meaning |
|-----|---------|
| `@P1` | 1st connected reference image |
| `@V1` | 1st connected reference video |
| `@A1` | 1st connected reference audio |
| `@C1` | storyboard character slot (director mode) |
| `@L1` | global asset library entry |

**🧰 Global Asset Library** (permanent, cross-workflow): the **素材库** widget on the node lets you upload local images (character/prop/scene) and audio files to disk (`input/zoey_library/`). Characters support an **appearance note** and an optional **voice audio**. Reference any saved entry with `@L1…`; the node auto-loads the file and injects it into generation (no LoadImage node needed), and auto-writes purpose annotations such as ``<Picture K> 是{name}的人物参考（锁定脸和服装）。外貌：…，音色参考 <Audio N>``.

**🎬 Director panel** (故事板/分镜): enable `director_mode` to build multi-shot storyboards instead of a single prompt. Per shot:
- Camera & shot-size preset buttons (push/pull/pan/truck/arc/POV/static/… + close-up/medium/wide/long/…)
- Dialogue rows (speaker S1–S5 + language + original text) compiled to `(S1) says: <d>[lang] text</d>`; global speaker list for voice descriptions
- Character slots `@C` (assign a reference image → auto ``<Picture K> 是{name}的人物参考（锁定脸和服装）`` declaration)
- Transition presets, duration with one-click **auto-allocate by dialogue length**, timeline start timestamps, shot reorder/duplicate, shot templates (product ad / character story / transition rhythm / music MV)
- Reference-purpose quick annotation, sound/music fields (`overall_soundscape` / `non_diegetic_music`), cross-shot consistency toggle, and a live compiled-prompt preview

### Highlights

- **Mask Bounding Box Drawer**: HTML5 color picker, opacity/fill/width controls, light type + prompt output, optional rembg background removal
- **Translation**: Pure Translator (Helsinki-NLP / Baidu / Google) + Hunyuan Translator (20+ languages, auto model cache)
- **Video Pipeline**: Batch load → process → save complete workflow
- **Image Editor**: 15+ operations including flip, rotate, blur, edge detect, sharpen, blend, stylize
- **MiniMax H3**: Seedance-style @ reference tags, permanent global asset library (local upload + auto-inject), and a full storyboard director panel
- **System Monitor**: live CPU/GPU/VRAM dashboard with background VRAM auto-cleanup
- **Music Generation (YuE2)**: whole pipeline in one node — text-to-music + cover + AI lyrics + voice conversion, with honest actual-duration reporting

### Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| torch, Pillow, numpy | Yes | Core |
| opencv-python-headless | Yes | Image/video ops |
| transformers | For Hunyuan | Translator model |
| rembg | Optional | Mask node BG removal |
| av / PyAV | Optional | Video processor |
| psutil | Optional | System monitor node |

### License

MIT

---

<a name="chinese"></a>

## 中文

ComfyUI 多功能工具插件，提供图像/视频处理、翻译、提示词生成等实用节点。

### 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liangzoey/comfyui-ZoeyTool.git
cd comfyui-ZoeyTool
pip install -r requirements.txt
```

### 节点列表

#### 🖼️ 图像处理

| 节点 | 功能 |
|------|------|
| **真尺寸图像加载器** | 批量逐张加载文件夹中的图像，保持原始尺寸，自然排序，输出 Alpha 通道作为遮罩 |
| **智能图像存储器** | 批量保存图像 + 文本文件，自定义序号位数、起始索引、覆盖模式 |
| **遮罩边界框绘制** | 根据遮罩绘制矩形框，弹出调色板选色，透明度/填充/线宽调节，预设配色、灯光类型与提示词输出、主体后合成，可选 rembg 背景移除 |
| **图像批量裁剪器** | 批量裁剪图片（左/右/上/下自由裁剪），覆盖模式，保留原文件名 |
| **灯光手柄控制** | 交互式灯光方向控制，拖拽手柄生成圆形渐变遮罩与灯光提示词（16 种灯光类型），支持主体后合成、多种手柄形状 |
| **文本叠加** | 在图像上叠加文字，支持自定义或系统字体，可调字号、颜色、位置 |
| **多功能图像编辑器** | 翻转、旋转、分割、边缘检测、模糊、锐化、二值化、颜色反转、灰度化、透视变换、融合、风格化 |
| **图像编辑提示词生成器** | 自动生成编辑提示词，支持文字/对象/风格/背景编辑、虚拟试穿、对象添加/移除 |
| **框选裁剪/外扩** | 交互式框选裁剪与画布外扩，实时预览，自定义填充色 |
| **多功能画布** | 5 层图像叠加合成，拖拽移动、滚轮缩放、旋转手柄、水平/垂直翻转 |

#### 🎥 视频处理

| 节点 | 功能 |
|------|------|
| **批量视频加载器** | 从目录加载视频文件列表，支持多格式（mp4/avi/mov），自然排序 |
| **视频批处理器** | 批量处理视频（格式转换、帧率调整），自动安装依赖，跳过已处理文件 |
| **智能视频存储器** | 保存处理后视频，支持按日期分类、元数据写入 |

#### 📝 文本工具

| 节点 | 功能 |
|------|------|
| **纯净翻译器** | 多引擎翻译：Helsinki-NLP 内置模型 / 百度API / 谷歌API，支持中/英/日/韩 |
| **混元翻译器 (HY-MT1.5)** | 腾讯混元翻译模型本地部署，20+ 语言，自动下载缓存，术语干预和上下文翻译 |
| **Wan2.2提示词生成器** | 影视级视频提示词生成，16 个控制维度：主体、场景、动作、光源、构图、镜头、风格 |

#### 🔧 其他

| 节点 | 功能 |
|------|------|
| **多文件批量重命名** | 批量重命名文件，正则查找替换，自然排序，自定义起始序号 |
| **ZOEYTextEncodeQwenImageEditPlus** | Qwen 图像编辑编码器，支持多张参考图 |
| **VR 360° 预览** | 360° 全景图 VR 预览，基于 Pannellum 全屏查看器 |
| **系统监控** | 实时监控 CPU/内存/显存/GPU 温度与利用率，内置 HTTP 看板，后台自动清理显存 |
| **MiniMax H3 参考转视频 (@)** | 包装官方 MiniMaxH3ReferenceToVideo，支持 Seedance 风格 @P1/@V1/@A1/@C1/@L1 引用语法自动展开为原生标签，附带导演台分镜面板与永久全局素材库 |
| **音乐生成 (YuE2)** | 把整条 YuE2 管线压进一个节点：文生音乐、翻唱（参考曲→旋律→重唱）、AI 写词、音色转换、LoRA。填一句描述、拖一下时长、按运行 |

##### 音乐生成 使用方法

一个节点顶掉原本 8 个节点的连线图。检查点加载、ABC 写谱、音乐 token 生成、采样、解码、保存全部在一个 `execute()` 里完成，不需要连任何线。

**时长是诚实的。** 「时长(秒)」是**上限**而不是承诺。YuE2 内部先把它换算成 token 预算（25 token/秒），风格或歌词太长时这个预算会被**静默压小**，模型自己也可能提前收尾。真实长度只有模型知道——所以节点会把它读回来，在节点面板上打印每一版的**实际**时长。成品通常比设定值短，属正常。

**歌词长度经验值**：约 **7 秒/行**。时长滑块下方会实时显示「建议歌词 ≤ N 行」。超出预算的歌词，结尾会被截断。

| 控件 | 说明 |
|------|------|
| `模式` | 文生音乐 / 翻唱（翻唱=先转写参考曲的旋律，再用新的风格和歌词重唱） |
| `歌曲描述` | 一句话。开启 `AI 写词` 且风格/歌词为空时用它 |
| `时长(秒)` | 30–300，是上限 |
| `版本数` | 1–4 版，种子为 `基准 + i × 7919` |

**风格选择器**（后端 COMBO，就是下拉框，不需要任何前端代码）：`语言` / `人声` / `曲风` / `BPM` / `乐器1-3` / `制作质感`。节点内部按和 YuE Studio 一样的公式拼成风格串：

```
语言 + 人声 + 曲风 + BPM + 乐器 + 制作质感
→ Chinese, husky male vocal, folk, 78 BPM, fingerstyle acoustic guitar, harmonica, warm analog production
```

`风格提示词` 是手填覆盖项——填了就以它为准。`歌词` 用 `[Verse]`/`[Chorus]` 分段；开 `AI 写词` 且歌词留空时由模型写。

优先级：**手填 `风格提示词` > `AI 写词` > 选择器拼串**。

其余输入：`AI 写词`（本地 llama.cpp + GGUF）、`写词模型`、`参考音频`、`模型`、`YuE2 LoRA`（同时作用于 MODEL 和 CLIP——曲风/旋律 token 是 CLIP 那半边产生的）、`输出格式`、`音色转换`（Demucs + Seed-VC）含 `移调`/`转换步数`、`识别歌词文本`（Qwen3-ASR）。

采样参数（步数 32 / CFG 1.0 / `dpm_2` / `sgm_uniform`）固定成这条管线验证过的那套，刻意不暴露——尤其 CFG，YuE2 的 CFG 是在文本编码器内部做的，这里改大只会炸。

本节点是**纯后端**：没有前端 JS。

**产物**落在 `output/yue/<时间戳>/`：每版一个 FLAC（真实长度，不做补零），另有 `mp3/`、`vc/`、`meta.json`、`lyrics.txt`。

**开了 AI 写词或音色转换时**，节点会把显存使用串行化：辅助引擎要么全在 YuE2 加载**之前**跑完，要么全在 YuE2 卸载**之后**跑。任何时刻都不会有两个 GPU 消费者同时驻留。

> ⚠️ **绝对路径与本机绑定。** 外部引擎按绝对路径调用，换位置需要改 `nodes/zoey_yue2_studio.py` 顶部的常量：llama.cpp（`LLAMA_SERVER`，以及 `CUDA_BIN`——这个目录不上 `PATH`，llama.cpp 会静默回落 CPU，慢 10 倍且不报错）、ASR（`ASR_PYTHON`/`ASR_SCRIPT`/`ASR_MODEL_ROOT`）、变声（`VC_PYTHON`/`VC_SCRIPT`/`SEEDVC_DIR`）。

> ⚠️ **缓存。** 和本仓库其它节点一样，本节点不定义 `IS_CHANGED`。`随机种子` 默认是 `control_after_generate`，重按运行会重新生成；但如果你把种子**固定**住，ComfyUI 会直接返回缓存音频，并**静默跳过** AI 写词与变声子进程。改一下种子或任意输入即可强制重跑。

##### MiniMax H3 使用方法

包装官方 `MiniMaxH3ReferenceToVideo`（文字 + 图片/视频/音频参考 → 视频）。提示词里输入 `@` 弹出带缩略图的素材选择器；鼠标悬停任意 `@` 缩略图可看大图。

**@ 引用标签** —— 自动展开为原生 `<Picture N>/<Video N>/<Audio N>`：
| 标签 | 含义 |
|------|------|
| `@P1` | 第 1 张已连接的参考图 |
| `@V1` | 第 1 段已连接的参考视频 |
| `@A1` | 第 1 段已连接的参考音频 |
| `@C1` | 导演台角色槽（导演模式） |
| `@L1` | 全局素材库条目 |

**🧰 永久全局素材库**（跨工作流）：节点上的「素材库」控件可本地上传图片（角色/道具/场景）与音频文件到磁盘（`input/zoey_library/`）。角色支持**外貌备注**与可选**语音音频**。提示词里用 `@L1…` 调用，节点自动加载文件并注入生成（无需再连 LoadImage 节点），并自动补用途标注，如 ``<Picture K> 是{name}的人物参考（锁定脸和服装）。外貌：…，音色参考 <Audio N>``。

**🎬 导演台（分镜/故事板）**：开启 `director_mode` 用多镜头分镜替代单条提示词。每镜支持：
- 景别/运镜快捷按钮（特写/近景/中景/全景/远景 + 推/拉/摇/横移/环绕/POV/静态…）
- 对白行（说话人 S1–S5 + 语言 + 台词原文）自动拼成 `(S1) says: <d>[语言] 原文</d>`；全局说话人列表写音色描述
- 角色槽 `@C`（分配参考图 → 自动生成 ``<Picture K> 是{name}的人物参考（锁定脸和服装）`` 声明）
- 转场预设、时长 + 一键**按台词分配时长**、镜头起始时间轴、镜头复制/上下移、镜头模板（产品广告/角色剧情/转场节奏/音乐MV）
- 参考用途快捷标注、音效/配乐字段（`overall_soundscape` / `non_diegetic_music`）、跨镜一致开关、实时编译预览

### 功能亮点

- **遮罩边界框绘制**：弹出式调色板、透明度/填充/线宽调节、灯光类型与提示词输出、可选 rembg 背景移除
- **翻译**：纯净翻译器(Helsinki-NLP/百度/谷歌) + 混元翻译器(20+语言、自动缓存)
- **视频流水线**：批量加载→处理→保存完整工作流
- **图像编辑器**：15+ 种操作（翻转、旋转、模糊、边缘检测、锐化、融合、风格化）
- **MiniMax H3**：Seedance 风格 @ 引用标签、永久全局素材库（本地上传 + 自动注入）、完整的导演台分镜面板
- **系统监控**：实时 CPU/GPU/显存看板，后台自动清理显存
- **音乐生成 (YuE2)**：整条管线收进一个节点——文生音乐 + 翻唱 + AI 写词 + 音色转换，并诚实回报实际时长

### 依赖

| 包 | 必需 | 说明 |
|----|------|------|
| torch, Pillow, numpy | 是 | 核心 |
| opencv-python-headless | 是 | 图像/视频处理 |
| transformers | 混元翻译器 | 翻译模型 |
| rembg | 可选 | 遮罩节点背景移除 |
| av / PyAV | 可选 | 视频处理器 |
| psutil | 可选 | 系统监控节点 |

### 许可

MIT
