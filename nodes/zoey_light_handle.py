"""
Zoey Light Handle Node
Interactive lighting direction control with circular mask generation.
- Draggable handle sets light position
- Generates circular gradient mask at handle position
- Generates lighting direction prompt from handle position
- Optional behind-subject compositing with auto remove-bg or external mask
- IMAGE output with handle overlay drawn using light color
- Multiple handle shapes: circle, square, diamond, triangle
- Handle opacity controlled by intensity
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import math
import logging
import os
import requests

logger = logging.getLogger("ZoeyTool")

# ===== Background removal support (same as mask_draw_rectangle) =====
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    logger.warning("rembg not installed. behind-subject auto mode requires: pip install rembg")

RMBG_MODEL_DIR = os.path.join("models", "rembg")
RMBG_MODEL_PATH = os.path.join(RMBG_MODEL_DIR, "RMBG-1.4.pth")
RMBG_MODEL_URL = "https://huggingface.co/zhengchong/RMBG-1.4/resolve/main/RMBG-1.4.pth"


def ensure_rmbg_model():
    if not os.path.exists(RMBG_MODEL_PATH):
        logger.info(f"Downloading RMBG-1.4 model to {RMBG_MODEL_PATH}...")
        os.makedirs(RMBG_MODEL_DIR, exist_ok=True)
        try:
            resp = requests.get(RMBG_MODEL_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with open(RMBG_MODEL_PATH, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("RMBG-1.4 model downloaded!")
        except Exception as e:
            logger.error(f"Failed to download RMBG-1.4 model: {e}")
            raise RuntimeError("Cannot get background removal model. Check network or place model manually.")


class ZoeyLightHandle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "azimuth": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1, "display": "slider"
                }),
                "elevation": ("FLOAT", {
                    "default": 30.0, "min": -90.0, "max": 90.0, "step": 1, "display": "slider"
                }),
                "ball_size": ("FLOAT", {
                    "default": 0.3, "min": 0.02, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "handle_shape": (["圆形", "方形", "菱形", "三角形"], {"default": "圆形"}),
                "light_color": ("STRING", {"default": "#FFFFFF"}),
                "intensity": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"
                }),
            },
            "optional": {
                "subject_mask": ("MASK",),
                "behind_subject": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("light_mask", "light_prompt", "preview_image")
    FUNCTION = "generate"
    CATEGORY = "Zoey工具集/图像编辑"
    OUTPUT_NODE = True

    def _build_prompt(self, handle_x, handle_y, behind_subject, light_color):
        th = 0.35
        if handle_x < th:
            x_dir = "左"
        elif handle_x > (1 - th):
            x_dir = "右"
        else:
            x_dir = ""

        if handle_y < th:
            y_dir = "上"
        elif handle_y > (1 - th):
            y_dir = "下"
        else:
            y_dir = ""

        if behind_subject:
            if x_dir == "" and y_dir == "":
                direction = "正后方"
            elif x_dir == "":
                direction = f"后{y_dir}方"
            elif y_dir == "":
                direction = f"{x_dir}后方"
            else:
                direction = f"{x_dir}后{y_dir}方"
        else:
            if x_dir == "" and y_dir == "":
                direction = "正前方"
            elif x_dir == "":
                direction = f"前{y_dir}方"
            elif y_dir == "":
                direction = f"{x_dir}前方"
            else:
                direction = f"{x_dir}前{y_dir}方"

        return f"根据图中色块方向和颜色打光，并移除色块，{light_color}色光光源来自{direction}"

    def _generate_shape_mask(self, h, w, cx, cy, radius, shape):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )

        if shape == "方形":
            dist = torch.max(torch.abs(x_coords - cx), torch.abs(y_coords - cy))
        elif shape == "菱形":
            dist = torch.abs(x_coords - cx) + torch.abs(y_coords - cy) * 0.6
        elif shape == "三角形":
            dx = x_coords - cx
            dy = y_coords - cy
            # Direction from image center to handle
            angle = math.atan2(cy - h / 2, cx - w / 2)
            # Rotate coordinates to align triangle
            rx = dx * math.cos(-angle) - dy * math.sin(-angle)
            ry = dx * math.sin(-angle) + dy * math.cos(-angle)
            # Triangle distance metric
            tri_dist = torch.max(
                torch.abs(rx) * 0.5 + ry * 0.4,
                torch.abs(rx) * 0.5 - ry * 0.3
            )
            tri_dist = torch.max(tri_dist, -ry * 0.3)
            dist = tri_dist + 0.3 * torch.abs(rx)
        else:
            # 圆形 - Euclidean distance
            dist = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

        t = torch.clamp(dist / radius, 0, 1)
        mask = 1.0 - t
        mask = mask * mask * (3 - 2 * mask)
        return mask

    def _draw_shape_on_canvas(self, draw, cx, cy, r, cr, cg, cb, alpha, shape, img_cx, img_cy):
        """Draw shape with opacity scaled by alpha (0-255)."""
        fill = (cr, cg, cb, max(5, int(180 * alpha / 255)))
        outline = (cr, cg, cb, max(5, int(255 * alpha / 255)))

        if shape == "方形":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
        elif shape == "菱形":
            pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            draw.polygon(pts, fill=fill, outline=outline)
        elif shape == "三角形":
            angle = math.atan2(cy - img_cy, cx - img_cx)
            tip = (cx + r * math.cos(angle), cy + r * math.sin(angle))
            b1 = (cx + r * 0.5 * math.cos(angle + 2.094), cy + r * 0.5 * math.sin(angle + 2.094))
            b2 = (cx + r * 0.5 * math.cos(angle - 2.094), cy + r * 0.5 * math.sin(angle - 2.094))
            draw.polygon([tip, b1, b2], fill=fill, outline=outline)
        else:  # 圆形
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)

    def _draw_handle_overlay(self, img_tensor, width, height, cx, cy, radius, light_color, intensity=5.0, handle_shape="圆形", behind_subject=False, subject_mask=None):
        img_np = (255. * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).convert('RGBA')

        h = light_color.lstrip('#')
        cr = int(h[0:2], 16) if len(h) >= 2 else 255
        cg = int(h[2:4], 16) if len(h) >= 4 else 255
        cb = int(h[4:6], 16) if len(h) >= 6 else 255

        # Intensity → opacity: 0→fully transparent, 10→fully opaque
        opacity = max(0.05, min(1.0, intensity / 10.0))
        alpha = int(opacity * 255)

        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        r = max(6, int(radius))

        # Draw the selected shape (opacity controlled by intensity)
        self._draw_shape_on_canvas(draw, cx, cy, r, cr, cg, cb, alpha, handle_shape, width/2, height/2)

        # Center dot (no crosshair)
        if alpha > 5:
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(cr, cg, cb, alpha))

        # Composite handle onto image
        base_with_handle = Image.alpha_composite(pil_img, overlay)

        # Place subject on top so handle appears behind the subject
        if behind_subject:
            subject_only = None
            if subject_mask is not None:
                # Use provided mask
                sm = subject_mask
                while sm.dim() > 2:
                    sm = sm[0]
                mask_np = (sm.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).resize((width, height), Image.BILINEAR)
                subject_only = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                subject_only.paste(pil_img, (0, 0), mask_pil)
            elif HAS_REMBG:
                # Auto remove background (same approach as mask_draw_rectangle)
                ensure_rmbg_model()
                subject_only = remove(
                    pil_img.convert('RGB'),
                    model_path=RMBG_MODEL_PATH,
                    only_mask=False,
                    post_process_mask=True
                ).convert("RGBA")

            if subject_only is not None:
                result = Image.alpha_composite(base_with_handle, subject_only)
            else:
                result = base_with_handle
        else:
            result = base_with_handle

        return torch.from_numpy(np.array(result.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)

    def generate(self, image, azimuth, elevation, ball_size, handle_shape="圆形", light_color="#FFFFFF", intensity=5.0, subject_mask=None, behind_subject=False):
        batch_size, height, width, channels = image.shape
        img = image[0]

        # Project 3D angles to 2D handle position on the image
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        handle_x = max(0.0, min(1.0, 0.5 + 0.5 * math.cos(el_rad) * math.sin(az_rad)))
        handle_y = max(0.0, min(1.0, 0.5 - 0.5 * math.sin(el_rad)))

        cx = handle_x * width
        cy = handle_y * height
        radius = max(2.0, ball_size * max(width, height))

        # Shape gradient mask
        light_mask = self._generate_shape_mask(height, width, cx, cy, radius, handle_shape)

        # Behind-subject compositing for mask
        if behind_subject:
            sub_mask_tensor = None
            if subject_mask is not None:
                # Use provided mask
                sm = subject_mask
                while sm.dim() > 2:
                    sm = sm[0]
                sub_mask_tensor = sm
            elif HAS_REMBG:
                # Auto generate subject mask via rembg
                ensure_rmbg_model()
                img_np_for_mask = (255. * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
                pil_rgb = Image.fromarray(img_np_for_mask)
                mask_img = remove(
                    pil_rgb,
                    model_path=RMBG_MODEL_PATH,
                    only_mask=True,
                    post_process_mask=True
                ).convert("L")
                mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                sub_mask_tensor = torch.from_numpy(mask_arr)

            if sub_mask_tensor is not None:
                if sub_mask_tensor.shape != (height, width):
                    sub_np = (sub_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    sub_pil = Image.fromarray(sub_np).resize((width, height), Image.BILINEAR)
                    sub_mask_tensor = torch.from_numpy(np.array(sub_pil).astype(np.float32)) / 255.0
                light_mask = light_mask * (1.0 - sub_mask_tensor)

        light_mask = light_mask.unsqueeze(0).to(torch.float32)

        # Lighting prompt
        prompt = self._build_prompt(handle_x, handle_y, behind_subject, light_color)

        # Annotated image output
        annotated = self._draw_handle_overlay(
            img, width, height, cx, cy, radius, light_color,
            intensity, handle_shape, behind_subject, subject_mask
        )

        # Frontend preview: ORIGINAL image (JS draws interactive handle)
        image_base64 = ""
        try:
            img_np = (255. * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil_original = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_original.save(buf, format="PNG")
            image_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

        return {"ui": {"image_base64": [image_base64]}, "result": (light_mask, prompt, annotated)}


NODE_CLASS_MAPPINGS = {
    "ZoeyLightHandle": ZoeyLightHandle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyLightHandle": "Zoey - 灯光手柄控制"
}
