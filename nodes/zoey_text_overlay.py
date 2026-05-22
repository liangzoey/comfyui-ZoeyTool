import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os


def _find_font(preferred=""):
    candidates = [
        preferred,
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


class ZoeyTextOverlay:
    """文本叠加 - 在图像上添加可拖拽定位的文字"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "text_config": ("STRING", {"multiline": False, "default": "{}", "dynamicPrompts": False}),
            },
            "optional": {
                "font_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "render"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def render(self, image, text, text_config="{}", font_path=""):
        try:
            cfg = json.loads(text_config)
        except Exception:
            cfg = {}

        B, H, W, C = image.shape
        results = []

        fp = _find_font(font_path)
        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(img_np).convert("RGBA")

            overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            size = int(cfg.get("size", 48))
            ox = float(cfg.get("x", 0.5))
            oy = float(cfg.get("y", 0.5))
            rot = float(cfg.get("r", 0))
            opacity = max(0, min(1, float(cfg.get("o", 1))))
            align = cfg.get("align", "center")
            hex_color = cfg.get("color", "#ffffff")

            try:
                hex_c = hex_color.lstrip("#")
                if len(hex_c) == 3:
                    hex_c = "".join(c * 2 for c in hex_c)
                fr = int(hex_c[0:2], 16)
                fg = int(hex_c[2:4], 16)
                fb = int(hex_c[4:6], 16)
            except Exception:
                fr, fg, fb = 255, 255, 255

            fa = int(opacity * 255)
            fill_color = (fr, fg, fb, fa)

            try:
                font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            px = int(ox * W - tw / 2)
            py = int(oy * H - th / 2)

            text_layer = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
            td = ImageDraw.Draw(text_layer)
            td.text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=fill_color)

            if rot != 0:
                text_layer = text_layer.rotate(rot, expand=True, center=(text_layer.width // 2, text_layer.height // 2),
                                                fillcolor=(0, 0, 0, 0))

            overlay.paste(text_layer, (px - 4, py - 4), text_layer)
            result = Image.alpha_composite(pil, overlay).convert("RGB")
            result_t = torch.from_numpy(np.array(result).astype(np.float32) / 255.0)
            results.append(result_t)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ZoeyTextOverlay": ZoeyTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyTextOverlay": "Zoey - 文本叠加",
}
