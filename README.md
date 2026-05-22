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
| **Mask Bounding Box Drawer** | Draw bounding boxes from mask with HTML5 color picker, opacity/fill/width controls, optional rembg background removal |
| **Batch Image Cropper** | Batch crop images (left/right/top/bottom), overwrite mode, filename preservation |
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

### Highlights

- **Mask Bounding Box Drawer**: HTML5 color picker, opacity/fill/width controls, optional rembg background removal
- **Translation**: Pure Translator (Helsinki-NLP / Baidu / Google) + Hunyuan Translator (20+ languages, auto model cache)
- **Video Pipeline**: Batch load → process → save complete workflow
- **Image Editor**: 15+ operations including flip, rotate, blur, edge detect, sharpen, blend, stylize

### Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| torch, Pillow, numpy | Yes | Core |
| opencv-python-headless | Yes | Image/video ops |
| transformers | For Hunyuan | Translator model |
| rembg | Optional | Mask node BG removal |
| av / PyAV | Optional | Video processor |

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
| **遮罩边界框绘制** | 根据遮罩绘制矩形框，弹出调色板选色，透明度/填充/线宽调节，可选 rembg 背景移除 |
| **图像批量裁剪器** | 批量裁剪图片（左/右/上/下自由裁剪），覆盖模式，保留原文件名 |
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

### 功能亮点

- **遮罩边界框绘制**：弹出式调色板、透明度/填充/线宽调节、可选 rembg 背景移除
- **翻译**：纯净翻译器(Helsinki-NLP/百度/谷歌) + 混元翻译器(20+语言、自动缓存)
- **视频流水线**：批量加载→处理→保存完整工作流
- **图像编辑器**：15+ 种操作（翻转、旋转、模糊、边缘检测、锐化、融合、风格化）

### 依赖

| 包 | 必需 | 说明 |
|----|------|------|
| torch, Pillow, numpy | 是 | 核心 |
| opencv-python-headless | 是 | 图像/视频处理 |
| transformers | 混元翻译器 | 翻译模型 |
| rembg | 可选 | 遮罩节点背景移除 |
| av / PyAV | 可选 | 视频处理器 |

### 许可

MIT
