import torch
import numpy as np
from PIL import Image, ImageDraw
import logging
import os
import requests
import math

logger = logging.getLogger("ZoeyTool")

# ===== 背景移除支持（用于灯光后置自动模式） =====
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    logger.warning("未安装 rembg，灯光后置自动模式将不可用。请运行: pip install rembg")

RMBG_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "rembg")
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
            logger.info("RMBG-1.4 模型下载完成！")
        except Exception as e:
            logger.error(f"下载 RMBG-1.4 模型失败: {e}")


class ZoeyMaskDrawBox:
    _LIGHT_TYPE_DESC = {
        "摄影棚灯光": "摄影棚灯光，柔和均匀的棚拍布光",
        "丁达尔光": "丁达尔光，光线在空气中穿透形成可见光柱，带有朦胧散射效果",
        "光斑": "光斑，光线透过缝隙形成斑驳的光点效果",
        "束光": "束光，光线收束成集中的光束，方向感强烈",
        "栅栏光": "栅栏光，光线透过百叶窗或格栅形成明暗相间的条状光影，带有规则几何纹理",
        "散射光": "散射光，光线经过漫射变得柔和均匀，阴影柔和",
        "柔光": "柔光，光线经过柔化处理，阴影边缘模糊过渡自然",
        "硬光": "硬光，光线直接照射，明暗对比强烈，阴影边缘锐利",
        "点光": "点光，从点状光源发出呈放射状扩散，有明显光源感",
        "面光": "面光，从大面积发光面发出，光线均匀柔和",
        "条光": "条光，光线呈条带状分布，有方向性的线性照明",
        "光晕": "光晕，光线在边缘扩散形成柔和的辉光发光效果",
        "漏光": "漏光，光线从缝隙或边缘渗入形成不规则光带",
        "轮廓光": "轮廓光，从侧后方勾勒主体边缘的照明光线",
        "斑驳光": "斑驳光，光线穿过遮挡物形成明暗交错的光影效果",
        "逆光": "逆光，光线从主体背后照射形成剪影效果",
    }

    _PRESET_COLORS = {
        "红色": "#ff0000",
        "橙色": "#ff8800",
        "黄色": "#ffff00",
        "绿色": "#00ff00",
        "青色": "#00ffff",
        "蓝色": "#0000ff",
        "紫色": "#ff00ff",
        "粉色": "#ff66b2",
        "白色": "#ffffff",
        "灰色": "#888888",
        "黑色": "#000000",
    }

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
                "强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "启用预设颜色": ("BOOLEAN", {"default": True}),
                "预设颜色": (list(cls._PRESET_COLORS.keys()), {"default": "红色"}),
                "自定义颜色": ("COLOR", {"default": "#ff0000"}),
                "边距百分比": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "灯光后置": ("BOOLEAN", {"default": False}),
                "光照类型": (["摄影棚灯光", "丁达尔光", "光斑", "束光", "栅栏光", "散射光", "柔光",
                              "硬光", "点光", "面光", "条光", "光晕", "漏光",
                              "轮廓光", "斑驳光", "逆光"], {"default": "摄影棚灯光"}),
                "主体遮罩": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "位置提示词")
    FUNCTION = "绘制方框"
    CATEGORY = "Zoey工具集/图像编辑"

    def 绘制方框(self, 图像, 遮罩, 线宽, 填充="否", 强度=1.0, 启用预设颜色=True, 预设颜色="红色", 自定义颜色="#ff0000", 边距百分比=5.0, 灯光后置=False, 光照类型="摄影棚灯光", 主体遮罩=None):
        # 颜色解析（预设颜色优先）
        if 启用预设颜色:
            color_hex = self._PRESET_COLORS.get(预设颜色, "#ff0000")
        else:
            color_hex = 自定义颜色
        try:
            hex_clean = color_hex.strip().lstrip('#')
            if len(hex_clean) != 6:
                raise ValueError("HEX 长度必须为6位")
            r = int(hex_clean[0:2], 16)
            g = int(hex_clean[2:4], 16)
            b = int(hex_clean[4:6], 16)
        except Exception as e:
            logger.warning(f"颜色 '{color_hex}' 格式无效，使用默认红色。错误: {e}")
            r, g, b = 255, 0, 0
            color_hex = "#ff0000"

        alpha = int(强度 * 0.5 * 255)
        rgba_color = (r, g, b, alpha)

        batch_size, height, width, _ = 图像.shape
        结果图像 = []
        last_bbox = None

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

            last_bbox = bbox

            # 绘制边界框
            overlay = Image.new('RGBA', original_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            effective_fill = rgba_color if 填充 == "是" else None
            draw.rectangle(bbox, fill=effective_fill, outline=rgba_color, width=线宽)
            base_with_box = Image.alpha_composite(original_img.convert('RGBA'), overlay)

            # 灯光后置：将主体重新合成到边框之上
            if 灯光后置:
                sub_mask_tensor = None
                if 主体遮罩 is not None:
                    sm = 主体遮罩[i] if 主体遮罩.dim() == 3 and i < 主体遮罩.shape[0] else 主体遮罩
                    sub_mask_tensor = sm.squeeze()
                elif HAS_REMBG:
                    ensure_rmbg_model()
                    img_np_for_mask = (255. * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
                    pil_rgb = Image.fromarray(img_np_for_mask)
                    mask_img = remove(pil_rgb, model_path=RMBG_MODEL_PATH,
                                      only_mask=True, post_process_mask=True).convert("L")
                    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                    sub_mask_tensor = torch.from_numpy(mask_arr)

                if sub_mask_tensor is not None:
                    if sub_mask_tensor.shape != (height, width):
                        sub_np = (sub_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                        sub_pil = Image.fromarray(sub_np).resize((width, height), Image.BILINEAR)
                        sub_mask_tensor = torch.from_numpy(np.array(sub_pil).astype(np.float32)) / 255.0

                    sub_np = sub_mask_tensor.cpu().numpy()
                    base_np = np.array(base_with_box.convert('RGB')).astype(np.float32)
                    orig_np = np.array(original_img).astype(np.float32)
                    final_np = base_np * (1.0 - sub_np[..., None]) + orig_np * sub_np[..., None]
                    final_img = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
                else:
                    final_img = base_with_box.convert('RGB')
            else:
                final_img = base_with_box.convert('RGB')

            tensor = torch.from_numpy(np.array(final_img).astype(np.float32)) / 255.0
            结果图像.append(tensor)

        # 构建提示词（与灯光手柄格式一致）
        if last_bbox is not None:
            x_min, y_min, x_max, y_max = last_bbox
            hx = (x_min + x_max) / 2.0 / width
            hy = (y_min + y_max) / 2.0 / height
            bbox_area = (x_max - x_min) * (y_max - y_min)
            img_area = width * height
            coverage = (bbox_area / img_area) * 100

            equiv_ball_size = math.sqrt((coverage / 100.0) / math.pi)

            direction = self._direction_text(hx, hy, 灯光后置)
            range_desc = self._range_text(equiv_ball_size)
            intens_desc = self._intensity_text(强度)
            type_desc = self._LIGHT_TYPE_DESC.get(光照类型, 光照类型)
            px = int(round(hx * width))
            py = int(round(hy * height))

            prompt = f"{type_desc}，{color_hex}色光光源来自{direction}，{range_desc}，{intens_desc} (坐标: {px},{py})"
            prompt = "根据图中色块方向和颜色打光，并移除色块，保持主体清晰，" + prompt
        else:
            prompt = "无有效遮罩区域"

        return (torch.stack(结果图像), prompt)

    def _direction_text(self, hx, hy, behind_subject):
        """与灯光手柄完全一致的方向描述"""
        th = 0.35
        x_raw = "左" if hx < th else ("右" if hx > (1 - th) else "")
        y_raw = "上" if hy < th else ("下" if hy > (1 - th) else "")

        side = "后" if behind_subject else "前"

        if not x_raw and not y_raw:
            return f"主体正{side}方"
        if x_raw and not y_raw:
            return f"{x_raw}侧主体{side}方"
        if not x_raw and y_raw:
            return f"{y_raw}主体{side}方"
        return f"{x_raw}{y_raw}主体{side}方"

    def _range_text(self, ball_size):
        """与灯光手柄完全一致的范围描述"""
        if ball_size < 0.12:
            return "点光源"
        elif ball_size < 0.25:
            return "小范围"
        elif ball_size < 0.45:
            return "中范围"
        else:
            return "大范围"

    def _intensity_text(self, intensity):
        """强度描述（0.0~1.0 范围）"""
        if intensity < 0.2:
            return "强度微弱"
        elif intensity < 0.4:
            return "强度较弱"
        elif intensity < 0.6:
            return "强度适中"
        elif intensity < 0.8:
            return "强度较强"
        else:
            return "强度强烈"

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
