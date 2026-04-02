import torch
import numpy as np
from PIL import Image, ImageDraw
import logging
import os
import requests

logger = logging.getLogger("ZoeyTool")

# ===== 背景移除支持 =====
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    logger.warning("未安装 rembg，背景移除功能将不可用。请运行: pip install rembg")

# 模型路径与下载
RMBG_MODEL_DIR = os.path.join("models", "rembg")
RMBG_MODEL_PATH = os.path.join(RMBG_MODEL_DIR, "RMBG-1.4.pth")
RMBG_MODEL_URL = "https://huggingface.co/zhengchong/RMBG-1.4/resolve/main/RMBG-1.4.pth"

def ensure_rmbg_model():
    if not os.path.exists(RMBG_MODEL_PATH):
        logger.info(f"RMBG-1.4 模型未找到，正在自动下载至: {RMBG_MODEL_PATH}")
        os.makedirs(RMBG_MODEL_DIR, exist_ok=True)
        try:
            resp = requests.get(RMBG_MODEL_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with open(RMBG_MODEL_PATH, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("✅ RMBG-1.4 模型下载完成！")
        except Exception as e:
            logger.error(f"❌ 下载 RMBG-1.4 模型失败: {e}")
            raise RuntimeError("无法获取背景移除模型，请检查网络或手动放置模型文件。")


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
                # ===== 简化后：仅保留自定义颜色 =====
                "自定义颜色": ("COLOR", {"default": "#ff0000"}), # 默认红色
                "边距百分比": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "启用背景移除": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "绘制方框"
    CATEGORY = "Zoey工具集/图像编辑"

    def 绘制方框(self, 图像, 遮罩, 线宽, 填充="否", 不透明度=1.0, 自定义颜色="#ff0000", 边距百分比=5.0, 启用背景移除=False):
        
        # ===== 简化后的颜色处理逻辑 =====
        # 1. 自定义颜色通常传入的是 HEX 字符串 (如 "#ff0000")
        # 2. 将 HEX 转换为 RGB 数值
        try:
            hex_clean = 自定义颜色.strip().lstrip('#')
            if len(hex_clean) != 6:
                raise ValueError("HEX 长度必须为6位")
            r = int(hex_clean[0:2], 16) 
            g = int(hex_clean[2:4], 16) 
            b = int(hex_clean[4:6], 16)
        except Exception as e:
            logger.warning(f"自定义颜色 '{自定义颜色}' 格式无效，使用默认红色。错误: {e}")
            r, g, b = 255, 0, 0

        # 3. 结合不透明度生成 RGBA 元组
        alpha = int(不透明度 * 255)
        rgba_color = (r, g, b, alpha)

        batch_size, height, width, _ = 图像.shape
        结果图像 = []

        if 启用背景移除 and HAS_REMBG:
            ensure_rmbg_model()

        for i in range(batch_size):
            img_tensor = 图像[i] * 255.0
            img_array = np.clip(img_tensor.cpu().numpy().astype(np.uint8), 0, 255)
            original_img = Image.fromarray(img_array).convert('RGB')

            mask = 遮罩[i] if 遮罩.dim() == 3 and i < 遮罩.shape[0] else 遮罩.squeeze()
            bbox = self.获取遮罩外接矩形(mask, 边距百分比)

            if bbox is None:
                logger.warning(f"图像 {i} 无有效遮罩区域")
                结果图像.append(torch.from_numpy(np.array(original_img).astype(np.float32)) / 255.0)
                continue

            # ===== 绘制逻辑 =====
            overlay = Image.new('RGBA', original_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            effective_fill = None
            if 填充 == "是":
                effective_fill = rgba_color # 填充颜色同样受不透明度控制

            # 绘制矩形
            draw.rectangle(
                bbox,
                fill=effective_fill,
                outline=rgba_color, # 边框颜色
                width=线宽
            )

            # 合成图像
            base_with_box = Image.alpha_composite(original_img.convert('RGBA'), overlay)

            if 启用背景移除 and HAS_REMBG:
                removed_bg_img = remove(
                    original_img,
                    model_path=RMBG_MODEL_PATH,
                    only_mask=False,
                    post_process_mask=True
                ).convert("RGBA")
                final_img = Image.alpha_composite(base_with_box, removed_bg_img).convert('RGB')
            else:
                final_img = base_with_box.convert('RGB')

            tensor = torch.from_numpy(np.array(final_img).astype(np.float32)) / 255.0
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
