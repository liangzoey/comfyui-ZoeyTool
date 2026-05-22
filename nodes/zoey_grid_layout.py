import torch
import numpy as np
from PIL import Image, ImageDraw
import json


class ZoeyGridLayout:
    """网格排版 - 多图网格排列布局"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "grid_config": ("STRING", {"multiline": False, "default": "{}", "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "compose"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def compose(self, image1, image2=None, image3=None, image4=None, image5=None,
                image6=None, image7=None, image8=None, image9=None, grid_config="{}"):
        try:
            cfg = json.loads(grid_config)
        except Exception:
            cfg = {}

        cols = max(1, int(cfg.get("cols", 3)))
        gap = int(cfg.get("gap", 4))
        border = int(cfg.get("border", 0))
        radius = int(cfg.get("radius", 0))
        bg_color = cfg.get("bg", "#222222")
        border_color = cfg.get("border_color", "#444444")

        try:
            bc = bg_color.lstrip("#")
            br_c = int(bc[0:2], 16) / 255.0 if len(bc) >= 2 else 0.133
            bg_r = int(bc[2:4], 16) / 255.0 if len(bc) >= 4 else 0.133
            bg_b = int(bc[4:6], 16) / 255.0 if len(bc) >= 6 else 0.133
        except Exception:
            br_c, bg_r, bg_b = 0.133, 0.133, 0.133

        images = [image1]
        for img in [image2, image3, image4, image5, image6, image7, image8, image9]:
            if img is not None:
                images.append(img)

        if not images:
            return (image1,)

        B = image1.shape[0]
        if len(images) == 1:
            return (images[0],)

        cell_w = max(img.shape[2] for img in images)
        cell_h = max(img.shape[1] for img in images)
        rows = (len(images) + cols - 1) // cols
        actual_cols = min(cols, len(images))

        grid_w = actual_cols * cell_w + (actual_cols - 1) * gap + border * 2
        grid_h = rows * cell_h + (rows - 1) * gap + border * 2

        result = torch.zeros((B, grid_h, grid_w, 3), dtype=image1.dtype, device=image1.device)
        result[:, :, :, 0] = br_c
        result[:, :, :, 1] = bg_r
        result[:, :, :, 2] = bg_b

        for i, img in enumerate(images):
            col = i % actual_cols
            row = i // actual_cols
            _, ih, iw, _ = img.shape

            x = border + col * (cell_w + gap) + (cell_w - iw) // 2
            y = border + row * (cell_h + gap) + (cell_h - ih) // 2

            if border > 0:
                try:
                    bc_hex = border_color.lstrip("#")
                    b_r = int(bc_hex[0:2], 16) / 255.0
                    b_g = int(bc_hex[2:4], 16) / 255.0
                    b_b = int(bc_hex[4:6], 16) / 255.0
                except Exception:
                    b_r, b_g, b_b = 0.267, 0.267, 0.267
                result[:, y - border:y + ih + border, x - border:x + iw + border, 0] = b_r
                result[:, y - border:y + ih + border, x - border:x + iw + border, 1] = b_g
                result[:, y - border:y + ih + border, x - border:x + iw + border, 2] = b_b

            if radius > 0:
                self._paste_rounded(result, img, x, y, iw, ih, radius)
            else:
                result[:, y:y + ih, x:x + iw, :3] = img[:, :, :, :3]

        return (result,)

    def _paste_rounded(self, canvas, img, x, y, w, h, r):
        B = img.shape[0]
        mask = torch.zeros((h, w), dtype=torch.float32, device=canvas.device)
        for i in range(h):
            for j in range(w):
                # Distance to nearest corner
                d = min(
                    ((j) ** 2 + (i) ** 2) ** 0.5,
                    ((w - 1 - j) ** 2 + (i) ** 2) ** 0.5,
                    ((j) ** 2 + (h - 1 - i) ** 2) ** 0.5,
                    ((w - 1 - j) ** 2 + (h - 1 - i) ** 2) ** 0.5,
                )
                mask[i, j] = 1.0 if d >= r else d / r
        mask = mask.clamp(0, 1)[None, :, :, None]  # (1, h, w, 1)
        for b in range(B):
            existing = canvas[b, y:y + h, x:x + w, :3]
            src = img[b, :, :, :3]
            canvas[b, y:y + h, x:x + w, :3] = src * mask + existing * (1 - mask)


NODE_CLASS_MAPPINGS = {
    "ZoeyGridLayout": ZoeyGridLayout,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyGridLayout": "Zoey - 网格排版",
}
