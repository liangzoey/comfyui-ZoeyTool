"""
Zoey Light Handle Node
Interactive lighting direction control with circular mask generation.
- Draggable handle sets light position
- Generates circular gradient mask at handle position
- Generates lighting direction prompt from handle position
- Optional behind-subject compositing with subject mask
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

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("light_mask", "light_prompt")
    FUNCTION = "generate"
    CATEGORY = "Zoey工具集/图像编辑"
    OUTPUT_NODE = True

    def _build_prompt(self, handle_x, handle_y, intensity, color_hex):
        """Generate lighting direction prompt from 2D handle position."""
        # Map handle position to azimuth (0-360) and elevation (-90 to 90)
        azimuth = (handle_x * 360) % 360
        elevation = 90 - handle_y * 180
        elevation = max(-90, min(90, elevation))

        # Azimuth (horizontal) description
        if (azimuth >= 337.5) or (azimuth < 22.5):
            pos_desc = "light source in front"
        elif 22.5 <= azimuth < 67.5:
            pos_desc = "light source from the front-right"
        elif 67.5 <= azimuth < 112.5:
            pos_desc = "light source from the right"
        elif 112.5 <= azimuth < 157.5:
            pos_desc = "light source from the back-right"
        elif 157.5 <= azimuth < 202.5:
            pos_desc = "light source from behind"
        elif 202.5 <= azimuth < 247.5:
            pos_desc = "light source from the back-left"
        elif 247.5 <= azimuth < 292.5:
            pos_desc = "light source from the left"
        else:
            pos_desc = "light source from the front-left"

        # Elevation (vertical) description
        e = elevation
        if -90 <= e < -30:
            elev_desc = "uplighting, light source positioned below, light shining upwards"
        elif -30 <= e < -10:
            elev_desc = "low-angle light source from below, upward illumination"
        elif -10 <= e < 20:
            elev_desc = "horizontal level light source"
        elif 20 <= e < 60:
            elev_desc = "high-angle light source"
        else:
            elev_desc = "overhead top-down light source"

        # Intensity
        if intensity < 3.0:
            int_desc = "soft"
        elif intensity < 7.0:
            int_desc = "bright"
        else:
            int_desc = "intense"

        constraints = "SCENE LOCK, FIXED VIEWPOINT, maintaining character consistency and pose. RELIGHTING ONLY: "
        return f"{constraints}{pos_desc}, {elev_desc}, {int_desc} colored light ({color_hex})"

    def _generate_circular_mask(self, h, w, cx, cy, radius):
        """Generate a smooth circular gradient mask centered at (cx, cy)."""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        dist = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        t = torch.clamp(dist / radius, 0, 1)
        # Smoothstep: 1.0 at center, 0.0 at radius
        mask = 1.0 - t
        # Smooth falloff: 3t^2 - 2t^3 applied to (1-t)
        mask = mask * mask * (3 - 2 * mask)
        return mask

    def generate(self, image, handle_x, handle_y, ball_size, light_color="#FFFFFF", intensity=5.0, subject_mask=None, behind_subject=False):
        batch_size, height, width, channels = image.shape
        img = image[0]

        # Handle position in pixel coords
        cx = handle_x * width
        cy = handle_y * height
        radius = max(2.0, ball_size * max(width, height))

        # Generate circular gradient mask
        light_mask = self._generate_circular_mask(height, width, cx, cy, radius)

        # Behind-subject compositing
        if behind_subject and subject_mask is not None:
            if subject_mask.dim() == 3:
                sub_mask = subject_mask[0]
            else:
                sub_mask = subject_mask

            if sub_mask.shape != (height, width):
                sub_np = (sub_mask.cpu().numpy() * 255).astype(np.uint8)
                sub_pil = Image.fromarray(sub_np).resize((width, height), Image.BILINEAR)
                sub_mask = torch.from_numpy(np.array(sub_pil).astype(np.float32)) / 255.0

            # Light falls behind subject: mask is occluded where subject blocks
            light_mask = light_mask * (1.0 - sub_mask)

        light_mask = light_mask.unsqueeze(0).to(torch.float32)

        # Generate prompt
        prompt = self._build_prompt(handle_x, handle_y, intensity, light_color)

        # Preview: draw handle indicator on image for frontend
        image_base64 = ""
        try:
            img_np = (255. * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            draw = ImageDraw.Draw(pil_image)
            r = max(6, int(radius))
            # Outer glow ring
            draw.ellipse([cx - r*2, cy - r*2, cx + r*2, cy + r*2],
                         outline="rgba(255,215,0,100)", width=1)
            # Handle circle
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline="#FFD700", width=2)
            # Crosshair
            ch = max(8, int(r * 0.3))
            draw.line([cx - ch, cy, cx + ch, cy], fill="white", width=1)
            draw.line([cx, cy - ch, cx, cy + ch], fill="white", width=1)
            # Center dot
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="#FFD700")

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

        return {"ui": {"image_base64": [image_base64]}, "result": (light_mask, prompt)}


NODE_CLASS_MAPPINGS = {
    "ZoeyLightHandle": ZoeyLightHandle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyLightHandle": "Zoey - 灯光手柄控制"
}
