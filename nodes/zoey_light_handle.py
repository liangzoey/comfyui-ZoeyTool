"""
Zoey Light Handle Node
Interactive lighting direction control with circular mask generation.
- Draggable handle sets light position
- Generates circular gradient mask at handle position
- Generates lighting direction prompt from handle position
- Optional behind-subject compositing with subject mask (alpha composite)
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


class ZoeyLightHandle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "handle_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "handle_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "ball_size": ("FLOAT", {
                    "default": 0.15, "min": 0.02, "max": 0.5, "step": 0.01, "display": "slider"
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

    def _build_prompt(self, behind_subject):
        if behind_subject:
            return "根据图中色块方向和颜色打光，并移除色块，光源在主体后方"
        return "根据图中色块方向和颜色打光，并移除色块"

    def _generate_circular_mask(self, h, w, cx, cy, radius):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        dist = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        t = torch.clamp(dist / radius, 0, 1)
        mask = 1.0 - t
        mask = mask * mask * (3 - 2 * mask)
        return mask

    def _draw_shape_on_canvas(self, draw, cx, cy, r, cr, cg, cb, alpha, shape, img_cx, img_cy):
        """Draw shape with opacity scaled by alpha (0-255)."""
        fill = (cr, cg, cb, max(5, int(60 * alpha / 255)))
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

        # Outer glow ring (only visible when opacity > 0)
        if alpha > 5:
            draw.ellipse([cx - r*2, cy - r*2, cx + r*2, cy + r*2], outline=(cr, cg, cb, max(5, alpha // 4)), width=1)

        # Draw the selected shape (opacity controlled by intensity)
        self._draw_shape_on_canvas(draw, cx, cy, r, cr, cg, cb, alpha, handle_shape, width/2, height/2)

        # Center dot (no crosshair)
        if alpha > 5:
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(cr, cg, cb, alpha))

        # Composite handle onto image
        base_with_handle = Image.alpha_composite(pil_img, overlay)

        # Place subject on top so handle appears behind the subject
        if behind_subject and subject_mask is not None:
            if subject_mask.dim() == 3:
                sub_mask = subject_mask[0]
            else:
                sub_mask = subject_mask

            mask_np = (sub_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np).resize((width, height), Image.BILINEAR)

            subject_only = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            subject_only.paste(pil_img, (0, 0), mask_pil)

            result = Image.alpha_composite(base_with_handle, subject_only)
        else:
            result = base_with_handle

        return torch.from_numpy(np.array(result.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)

    def generate(self, image, handle_x, handle_y, ball_size, handle_shape="圆形", light_color="#FFFFFF", intensity=5.0, subject_mask=None, behind_subject=False):
        batch_size, height, width, channels = image.shape
        img = image[0]

        cx = handle_x * width
        cy = handle_y * height
        radius = max(2.0, ball_size * max(width, height))

        # Circular gradient mask
        light_mask = self._generate_circular_mask(height, width, cx, cy, radius)

        # Behind-subject compositing for mask
        if behind_subject and subject_mask is not None:
            if subject_mask.dim() == 3:
                sub_mask = subject_mask[0]
            else:
                sub_mask = subject_mask

            if sub_mask.shape != (height, width):
                sub_np = (sub_mask.cpu().numpy() * 255).astype(np.uint8)
                sub_pil = Image.fromarray(sub_np).resize((width, height), Image.BILINEAR)
                sub_mask = torch.from_numpy(np.array(sub_pil).astype(np.float32)) / 255.0

            light_mask = light_mask * (1.0 - sub_mask)

        light_mask = light_mask.unsqueeze(0).to(torch.float32)

        # Lighting prompt
        prompt = self._build_prompt(behind_subject)

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
