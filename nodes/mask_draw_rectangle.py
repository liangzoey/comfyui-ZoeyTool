import torch
import numpy as np
from PIL import Image, ImageDraw
import colorsys
import logging

logger = logging.getLogger("ZoeyTool")


class ZoeyMaskDrawBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "线宽": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "填充": (["否", "是"], {"default": "否"}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # ===== 颜色模式切换 =====
                "颜色模式": (["预设", "调色盘", "HEX"], {"default": "预设"}),
                # 预设颜色（当颜色模式=预设时使用）
                "颜色预设": (
                    ["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色", "白色", "黑色", "灰色"],
                    {"default": "红色"}
                ),
                # 调色盘（当颜色模式=调色盘时使用）—— 使用 ComfyUI 原生 COLOR 滑块
                "自定义颜色": ("COLOR", {"default": (1.0, 0.0, 0.0)}),  # RGB in [0.0, 1.0]
                # HEX 输入（当颜色模式=HEX时使用）
                "HEX颜色": ("STRING", {"default": "#ff0000", "multiline": False}),
                # 其他参数
                "亮度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "饱和度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "边距百分比": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "绘制方框"
    CATEGORY = "Zoey工具集/图像编辑"

    def 绘制方框(self, 图像, 遮罩, 线宽, 填充="否", 不透明度=1.0,
               颜色模式="预设", 颜色预设="红色", 自定义颜色=(1.0, 0.0, 0.0), HEX颜色="#ff0000",
               亮度=1.0, 饱和度=1.0, 边距百分比=5.0):

        # 默认颜色：红色
        r, g, b = 1.0, 0.0, 0.0

        if 颜色模式 == "预设":
            颜色映射 = {
                "红色": (0.0, 1.0, 1.0),
                "橙色": (0.08, 1.0, 1.0),
                "黄色": (0.16, 1.0, 1.0),
                "绿色": (0.33, 1.0, 1.0),
                "青色": (0.5, 1.0, 1.0),
                "蓝色": (0.66, 1.0, 1.0),
                "紫色": (0.83, 1.0, 1.0),
                "粉色": (0.92, 1.0, 1.0),
                "白色": (0.0, 0.0, 1.0),
                "黑色": (0.0, 0.0, 0.0),
                "灰色": (0.0, 0.0, 0.5),
            }
            h, s, v = 颜色映射.get(颜色预设, (0.0, 1.0, 1.0))
            s = min(1.0, max(0.0, s * 饱和度))
            v = min(1.0, max(0.0, v * 亮度))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)

        elif 颜色模式 == "调色盘":
            # 使用原 COLOR 滑块（RGB in [0,1]）
            try:
                r = float(自定义颜色[0])
                g = float(自定义颜色[1])
                b = float(自定义颜色[2])
            except (TypeError, IndexError, ValueError):
                logger.warning("自定义颜色格式无效，使用默认红色")
                r, g, b = 1.0, 0.0, 0.0

        elif 颜色模式 == "HEX":
            # 解析 HEX 字符串，如 "#aabbcc" 或 "aabbcc"
            try:
                hex_clean = HEX颜色.strip().lstrip('#')
                if len(hex_clean) != 6:
                    raise ValueError("HEX 长度必须为6位")
                r = int(hex_clean[0:2], 16) / 255.0
                g = int(hex_clean[2:4], 16) / 255.0
                b = int(hex_clean[4:6], 16) / 255.0
            except Exception as e:
                logger.warning(f"HEX颜色 '{HEX颜色}' 格式无效，使用默认红色。错误: {e}")
                r, g, b = 1.0, 0.0, 0.0

        # 统一应用亮度和饱和度（通过 HSV）
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, max(0.0, s * 饱和度))
        v = min(1.0, max(0.0, v * 亮度))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgba = (int(r * 255), int(g * 255), int(b * 255), int(不透明度 * 255))

        batch_size, height, width, _ = 图像.shape
        结果图像 = []

        for i in range(batch_size):
            img_tensor = 图像[i] * 255.0
            img_array = np.clip(img_tensor.cpu().numpy().astype(np.uint8), 0, 255)
            img = Image.fromarray(img_array).convert('RGBA')

            mask = 遮罩[i] if 遮罩.dim() == 3 and i < 遮罩.shape[0] else 遮罩.squeeze()
            bbox = self.获取遮罩外接矩形(mask, 边距百分比)

            if bbox is None:
                logger.warning(f"图像 {i} 无有效遮罩区域")
                结果图像.append(torch.from_numpy(np.array(img.convert('RGB'))).float() / 255.0)
                continue

            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            if 填充 == "是":
                draw.rectangle(bbox, fill=rgba, outline=rgba[:3], width=线宽)
            else:
                draw.rectangle(bbox, outline=rgba, width=线宽)

            img = Image.alpha_composite(img, overlay).convert('RGB')
            tensor = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
            结果图像.append(tensor)

        return (torch.stack(结果图像),)

    def 获取遮罩外接矩形(self, 遮罩张量, 边距百分比=5.0):
        try:
            mask_arr = 遮罩张量.cpu().numpy().squeeze()
            if mask_arr.ndim != 2:
                return None

            coords = np.where(mask_arr > 0)
            if len(coords[0]) == 0:
                return None

            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            h, w = mask_arr.shape
            pad_x = int((x_max - x_min) * 边距百分比 / 100)
            pad_y = int((y_max - y_min) * 边距百分比 / 100)

            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w - 1, x_max + pad_x)
            y_max = min(h - 1, y_max + pad_y)

            return (x_min, y_min, x_max, y_max)
        except Exception as e:
            logger.error(f"计算边界框失败: {e}")
            return None


# === 注册节点 ===
NODE_CLASS_MAPPINGS = {
    "ZoeyMaskDrawBox": ZoeyMaskDrawBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMaskDrawBox": "🎨 Zoey - 遮罩边界框绘制"
}
