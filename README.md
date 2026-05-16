# Zoey Tool - ComfyUI Custom Nodes

A multi-functional ComfyUI plugin providing image/video processing, translation, prompt generation, and other utility nodes.

ComfyUI 多功能工具插件，提供图像/视频处理、翻译、提示词生成等实用节点。

---

## Installation / 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liangzoey/comfyui-ZoeyTool.git
cd comfyui-ZoeyTool
pip install -r requirements.txt
```

---

## Nodes / 节点列表

### 🖼️ Image Processing / 图像处理

| Node | 节点 | Description |
|------|------|-------------|
| **Zoey - True Size Image Loader** | 真尺寸图像加载器 | Load images from folder at original resolution, batch sequential output, natural sorting, alpha channel as mask |
| **Zoey - Batch Image Saver** | 智能图像存储器 | Save images + text files in batch with custom index, digits, overwrite mode, batch name prefix |
| **Zoey - Mask Bounding Box Drawer** | 遮罩边界框绘制 | Draw bounding boxes from mask with HTML5 color picker, opacity/fill/width controls, optional rembg background removal |
| **Zoey - Batch Image Cropper** | 图像批量裁剪器 | Batch crop images (left/right/top/bottom), overwrite mode, filename preservation |
| **Zoey - Multi-function Image Editor** | 多功能图像编辑器 | Flip, rotate, split, edge detect, blur, sharpen, threshold, invert, grayscale, perspective warp, blend, stylize |
| **Zoey - Image Edit Prompt Generator** | 图像编辑提示词生成器 | Auto-generate edit prompts for text/object/style/background replacement, virtual try-on, object add/remove |

### 🎥 Video Processing / 视频处理

| Node | 节点 | Description |
|------|------|-------------|
| **Zoey - Batch Video Loader** | 批量视频加载器 | Load video file list from directory, multi-format support (mp4/avi/mov), natural sort, limit |
| **Zoey - Video Batch Processor** | 视频批处理器 | Batch process videos (format conversion, frame rate), auto-install deps, skip existing |
| **Zoey - Smart Video Saver** | 智能视频存储器 | Save processed videos with date folder, metadata writing |

### 📝 Text Tools / 文本工具

| Node | 节点 | Description |
|------|------|-------------|
| **Zoey - Pure Translator** | 纯净翻译器 | Multi-engine translation: Helsinki-NLP models / Baidu API / Google API (free). Supports ZH/EN/JA/KO |
| **Zoey - Hunyuan Translator (HY-MT1.5)** | 混元翻译器 | Tencent Hunyuan MT model local deployment, 20+ languages, auto-download, terminology & context support |
| **Zoey - Wan2.2 Prompt Generator** | Wan2.2提示词生成器 | Cinematic video prompt generator with 16 dimensions: subject, scene, action, lighting, composition, lens, style |

### 🔧 Other / 其他

| Node | 节点 | Description |
|------|------|-------------|
| **Zoey - Multi File Batch Renamer** | 多文件批量重命名 | Batch rename files with regex find/replace, natural sort, custom start index and digit count |
| **ZOEYTextEncodeQwenImageEditPlus** | Qwen 图像编辑编码器 | Qwen image edit encoder with multi-reference image support for Qwen2-VL |
| **Zoey VR 360° Preview** | VR 360° 嵌入式预览 | 360° equirectangular panorama VR preview powered by Pannellum, fullscreen interactive viewer |

---

## Highlights / 功能亮点

### 🎨 Mask Bounding Box Drawer / 遮罩边界框绘制
- HTML5 color picker popup for color selection / 弹出式调色板自由选色
- Opacity / Fill (solid/hollow) / Line width controls / 透明度、填充、线宽调节
- Optional rembg background removal (auto-downloads RMBG-1.4 model) / 可选 rembg 背景移除
- Margin percentage for box expansion / 边距百分比控制扩展范围

### 🌐 Translation / 翻译
- **Pure Translator**: Lightweight, supports Helsinki-NLP / Baidu API / Google API / 轻量级多引擎翻译
- **Hunyuan Translator**: Full Hunyuan HY-MT1.5 with 20+ languages, auto model caching, terminology & context / 全量语言支持，自动下载缓存

### 📹 Video Pipeline / 视频处理流水线
Batch Load → Batch Process → Smart Save, complete video batch workflow / 批量加载→处理→保存，完整工作流

---

## Dependencies / 依赖

| Package | Required | Notes |
|---------|----------|-------|
| torch, Pillow, numpy | Yes | Core |
| opencv-python-headless | Yes | Image/video processing |
| transformers | For Hunyuan Translator | 混元翻译器 |
| rembg | Optional | Mask node background removal / 遮罩背景移除 |
| av / PyAV | Optional | Video processor / 视频处理器 |

---

## License / 许可

MIT
