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

RMBG_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "rembg")
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
                "light_type": (["摄影棚灯光", "丁达尔光", "光斑", "束光", "栅栏光", "散射光", "柔光",
                                "硬光", "点光", "面光", "条光", "光晕", "漏光",
                                "轮廓光", "斑驳光", "逆光"], {"default": "摄影棚灯光"}),
            },
            "optional": {
                "subject_mask": ("MASK",),
                "behind_subject": ("BOOLEAN", {"default": False}),
                "handles_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("light_mask", "light_prompt", "preview_image")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "Zoey工具集/图像编辑"

    def _direction_text(self, hx, hy, behind_subject):
        """Convert normalized coordinates to direction description.
        Format: 左侧主体前方 / 左下主体后方 / 主体正前方"""
        th = 0.35
        x_raw = "左" if hx < th else ("右" if hx > (1 - th) else "")
        y_raw = "上" if hy < th else ("下" if hy > (1 - th) else "")

        side = "后" if behind_subject else "前"

        if not x_raw and not y_raw:
            return f"主体正{side}方"
        # x-only: add 侧, e.g. "左侧" → "左侧主体前方"
        if x_raw and not y_raw:
            return f"{x_raw}侧主体{side}方"
        # y-only: e.g. "上" → "上主体前方"
        if not x_raw and y_raw:
            return f"{y_raw}主体{side}方"
        # both x and y: e.g. "左下" → "左下主体后方"
        return f"{x_raw}{y_raw}主体{side}方"

    def _range_text(self, ball_size):
        """Describe light range based on ball_size (0.02~1.0)."""
        if ball_size < 0.12:
            return "点光源"
        elif ball_size < 0.25:
            return "小范围"
        elif ball_size < 0.45:
            return "中范围"
        else:
            return "大范围"

    def _intensity_text(self, intensity):
        """Describe light intensity (0~10)."""
        if intensity < 2:
            return "强度微弱"
        elif intensity < 4:
            return "强度较弱"
        elif intensity < 6:
            return "强度适中"
        elif intensity < 8:
            return "强度较强"
        else:
            return "强度强烈"

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

    def _build_coord_string(self, handles, image_width, image_height):
        """Build prompt with original format, detailed light type description."""
        parts = []
        for h in handles:
            color = h.get('color', h.get('light_color', '#FFFFFF'))
            hx = h.get('x', h.get('handle_x', 0.5))
            hy = h.get('y', h.get('handle_y', 0.5))
            behind = h.get('behind_subject', False)
            bsize = h.get('ball_size', 0.3)
            ltype = h.get('light_type', '摄影棚灯光')
            intens = h.get('intensity', 5.0)
            px = int(round(hx * image_width))
            py = int(round(hy * image_height))
            direction = self._direction_text(hx, hy, behind)
            range_desc = self._range_text(bsize)
            intens_desc = self._intensity_text(intens)
            type_desc = self._LIGHT_TYPE_DESC.get(ltype, ltype)
            parts.append(f"{type_desc}，{color}色光光源来自{direction}，{range_desc}，{intens_desc} (坐标: {px},{py})")
        return "根据图中色块方向和颜色打光，并移除色块，保持主体清晰，" + "; ".join(parts)

    def _compute_handle_xy(self, azimuth, elevation):
        """Convert 3D angles to 2D normalized coordinates (0-1) on image."""
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        hx = max(0.0, min(1.0, 0.5 + 0.5 * math.cos(el_rad) * math.sin(az_rad)))
        hy = max(0.0, min(1.0, 0.5 - 0.5 * math.sin(el_rad)))
        return hx, hy

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

    def _parse_handles(self, handles_json, azimuth, elevation, ball_size, handle_shape, light_color, intensity, behind_subject, light_type):
        """Parse handles_json or create default single handle."""
        import json
        if handles_json and handles_json.strip() not in ("", "[]"):
            try:
                raw = json.loads(handles_json)
                if isinstance(raw, list) and len(raw) > 0:
                    # Ensure all handles have required fields
                    return [{
                        "x": h.get("x", 0.5),
                        "y": h.get("y", 0.5),
                        "azimuth": h.get("azimuth", azimuth),
                        "elevation": h.get("elevation", elevation),
                        "ball_size": h.get("ball_size", ball_size),
                        "handle_shape": h.get("handle_shape", handle_shape),
                        "light_color": h.get("light_color", light_color),
                        "intensity": h.get("intensity", intensity),
                        "behind_subject": h.get("behind_subject", behind_subject),
                        "light_type": h.get("light_type", light_type),
                    } for h in raw]
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: single handle from current widget values
        return [{
            "azimuth": azimuth,
            "elevation": elevation,
            "ball_size": ball_size,
            "handle_shape": handle_shape,
            "light_color": light_color,
            "intensity": intensity,
            "behind_subject": behind_subject,
            "light_type": light_type,
        }]

    def _draw_all_handles(self, img_tensor, width, height, handles):
        """Draw all handles on the image preview."""
        img_np = (255. * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).convert('RGBA')
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for h in handles:
            if 'x' in h and 'y' in h:
                hx, hy = h['x'], h['y']
            else:
                hx, hy = self._compute_handle_xy(
                    h.get('azimuth', 0), h.get('elevation', 30))
            cx = hx * width
            cy = hy * height
            radius = max(6, int(h.get('ball_size', 0.3) * max(width, height)))

            color = h.get('light_color', '#FFFFFF')
            color_s = color.lstrip('#')
            cr = int(color_s[0:2], 16) if len(color_s) >= 2 else 255
            cg = int(color_s[2:4], 16) if len(color_s) >= 4 else 255
            cb = int(color_s[4:6], 16) if len(color_s) >= 6 else 255

            intensity = h.get('intensity', 5.0)
            opacity = max(0.05, min(1.0, intensity / 10.0))
            alpha = int(opacity * 255)
            shape = h.get('handle_shape', '圆形')

            self._draw_shape_on_canvas(draw, int(cx), int(cy), radius,
                cr, cg, cb, alpha, shape, width / 2, height / 2)
            if alpha > 5:
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3],
                    fill=(cr, cg, cb, alpha))

        base_with_handle = Image.alpha_composite(pil_img, overlay)

        # Behind-subject compositing (uses first handle's behind_subject flag)
        behind = handles[0].get('behind_subject', False) if handles else False
        if behind:
            # Simplified: behind-subject for preview requires subject_mask
            # (the actual mask computation handles this in generate())
            pass

        return torch.from_numpy(
            np.array(base_with_handle.convert('RGB')).astype(np.float32) / 255.0
        ).unsqueeze(0)

    def generate(self, image, azimuth, elevation, ball_size, handle_shape="圆形", light_color="#FFFFFF", intensity=5.0, subject_mask=None, behind_subject=False, handles_json="[]", light_type="摄影棚灯光"):
        batch_size, height, width, channels = image.shape
        img = image[0]

        # Parse multiple handles
        handles = self._parse_handles(handles_json, azimuth, elevation,
            ball_size, handle_shape, light_color, intensity, behind_subject, light_type)

        # Generate combined mask from all handles
        combined_mask = None
        coord_items = []
        for h in handles:
            if 'x' in h and 'y' in h:
                hx, hy = h['x'], h['y']
            else:
                hx, hy = self._compute_handle_xy(
                    h.get('azimuth', 0), h.get('elevation', 30))
            cx = hx * width
            cy = hy * height
            bsize = h.get('ball_size', 0.3)
            radius = max(2.0, bsize * max(width, height))
            shape = h.get('handle_shape', '圆形')

            mask = self._generate_shape_mask(height, width, cx, cy, radius, shape)
            combined_mask = mask if combined_mask is None else torch.max(combined_mask, mask)
            coord_items.append({'x': hx, 'y': hy, 'ball_size': bsize, 'intensity': h.get('intensity', 5.0), 'light_type': h.get('light_type', '摄影棚灯光'), 'color': h.get('light_color', '#FFFFFF'), 'behind_subject': h.get('behind_subject', False)})

        # Behind-subject compositing for mask (use first handle's flag)
        h_behind = handles[0].get('behind_subject', False) if handles else behind_subject
        if h_behind:
            sub_mask_tensor = None
            if subject_mask is not None:
                sm = subject_mask
                while sm.dim() > 2:
                    sm = sm[0]
                sub_mask_tensor = sm
            elif HAS_REMBG:
                ensure_rmbg_model()
                img_np_for_mask = (255. * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
                pil_rgb = Image.fromarray(img_np_for_mask)
                mask_img = remove(pil_rgb, model_path=RMBG_MODEL_PATH,
                    only_mask=True, post_process_mask=True).convert("L")
                mask_arr = np.array(mask_img).astype(np.float32) / 255.0
                sub_mask_tensor = torch.from_numpy(mask_arr)

            if sub_mask_tensor is not None:
                if sub_mask_tensor.shape != (height, width):
                    sub_np = (sub_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    sub_pil = Image.fromarray(sub_np).resize((width, height), Image.BILINEAR)
                    sub_mask_tensor = torch.from_numpy(
                        np.array(sub_pil).astype(np.float32)) / 255.0
                combined_mask = combined_mask * (1.0 - sub_mask_tensor)

        light_mask = combined_mask.unsqueeze(0).to(torch.float32)

        # Coordinate prompt
        prompt = self._build_coord_string(coord_items, width, height)

        # Annotated image with all handles drawn
        annotated = self._draw_all_handles(img, width, height, handles)

        # Frontend preview: ORIGINAL image (JS draws interactive handles)
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
