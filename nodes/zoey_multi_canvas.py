import torch
import json
import math
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F

# ── Rembg 背景移除集成 ──
_REMBG_SESSIONS = {}

def _get_rembg_model_dir():
    """获取 models/rembg 目录（ComfyUI 标准路径）"""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "rembg")
    except Exception:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        models_dir = os.path.join(base, "models", "rembg")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def _get_rembg_session(model_name):
    if model_name not in _REMBG_SESSIONS:
        try:
            from rembg import new_session
            os.environ["REMBG_MODEL_DIR"] = _get_rembg_model_dir()
            _REMBG_SESSIONS[model_name] = new_session(model_name)
        except ImportError:
            print("=" * 60)
            print("[ZoeyMultiCanvas] rembg 未安装，背景移除不可用。")
            print("  请运行: pip install rembg")
            print("=" * 60)
            return None
        except Exception as e:
            print(f"[ZoeyMultiCanvas] 加载 rembg 模型 '{model_name}' 失败: {e}")
            return None
    return _REMBG_SESSIONS[model_name]

def _remove_bg(img_tensor, model_name="u2net"):
    """
    对 (B, H, W, 3) 图像执行背景移除。
    返回: (rgb: B,H,W,3, alpha: B,H,W,1)
    """
    try:
        from rembg import remove
    except ImportError:
        return img_tensor, None

    session = _get_rembg_session(model_name)
    if session is None:
        return img_tensor, None

    rgb_list = []
    alpha_list = []
    for i in range(img_tensor.shape[0]):
        img_np = (img_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, "RGB")
        out_pil = remove(img_pil, session=session)  # → RGBA
        out_np = np.array(out_pil).astype(np.float32) / 255.0
        out_t = torch.from_numpy(out_np).to(img_tensor.device)
        rgb_list.append(out_t[:, :, :3])
        alpha_list.append(out_t[:, :, 3:4])

    return torch.stack(rgb_list), torch.stack(alpha_list)

# ── 常用 rembg 模型列表 ──
REMBG_MODELS = [
    "u2net", "u2netp", "u2net_human_seg",
    "isnet-general-use", "isnet-anime", "silueta",
    "birefnet-general", "birefnet-general-lite",
    "birefnet-dis", "birefnet-dis-lite",
    "birefnet-massive", "birefnet-massive-lite",
]


class ZoeyMultiCanvas:
    """多功能画布 - 多层图像叠加合成（支持旋转/翻转/透明通道/背景移除）"""
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
                "layer_config": ("STRING", {
                    "multiline": False,
                    "default": "{}",
                    "dynamicPrompts": False,
                }),
                "bg_model": (REMBG_MODELS, {"default": "u2net"}),
                "bg_remove_1": ("BOOLEAN", {"default": False, "label_on": "图层1去背景", "label_off": "关闭"}),
                "bg_remove_2": ("BOOLEAN", {"default": False, "label_on": "图层2去背景", "label_off": "关闭"}),
                "bg_remove_3": ("BOOLEAN", {"default": False, "label_on": "图层3去背景", "label_off": "关闭"}),
                "bg_remove_4": ("BOOLEAN", {"default": False, "label_on": "图层4去背景", "label_off": "关闭"}),
                "bg_remove_5": ("BOOLEAN", {"default": False, "label_on": "图层5去背景", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("合成图像",)
    FUNCTION = "composite"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def composite(self, image1, image2=None, image3=None, image4=None, image5=None,
                  layer_config="{}", bg_model="u2net",
                  bg_remove_1=False, bg_remove_2=False, bg_remove_3=False,
                  bg_remove_4=False, bg_remove_5=False):
        try:
            config = json.loads(layer_config)
        except Exception:
            config = {}

        layers = []
        inputs = [
            ("layer1", image1),
            ("layer2", image2),
            ("layer3", image3),
            ("layer4", image4),
            ("layer5", image5),
        ]
        bg_toggles = [bg_remove_1, bg_remove_2, bg_remove_3, bg_remove_4, bg_remove_5]

        for i, (key, img) in enumerate(inputs):
            if img is None:
                continue

            cfg = config.get(key, {})

            # 提取 RGB + 原生 alpha（来自 RGBA 输入）
            img_rgb = img[:, :, :, :3]
            native_alpha = img[:, :, :, 3:4] if img.shape[-1] == 4 else None

            # 背景移除：UI 开关 || JSON 配置
            bg_remove = bg_toggles[i] or bool(cfg.get("bgr", False))
            if bg_remove:
                # 优先使用图层级模型名，否则用节点级默认
                m = str(cfg.get("bgm", bg_model))
                removed_rgb, removed_alpha = _remove_bg(img_rgb, m)
                if removed_alpha is not None:
                    img_rgb = removed_rgb
                    native_alpha = removed_alpha if native_alpha is None else native_alpha * removed_alpha

            layers.append({
                "img": img_rgb,
                "native_alpha": native_alpha,  # (B, H, W, 1) or None
                "ox": float(cfg.get("x", 0)),
                "oy": float(cfg.get("y", 0)),
                "scale": max(0.01, float(cfg.get("s", 1))),
                "opacity": max(0, min(1, float(cfg.get("o", 1)))),
                "visible": bool(cfg.get("v", True)),
                "rotation": float(cfg.get("r", 0)) % 360,
                "flip_h": bool(cfg.get("fh", False)),
                "flip_v": bool(cfg.get("fv", False)),
            })

        if not layers:
            return (image1,)

        if len(layers) == 1:
            return (layers[0]["img"],)

        B, baseH, baseW, C = layers[0]["img"].shape
        dtype = layers[0]["img"].dtype
        device = layers[0]["img"].device

        result = torch.zeros((B, baseH, baseW, C), dtype=dtype, device=device)

        for layer in layers:
            if not layer["visible"]:
                continue

            img = layer["img"]
            native_alpha = layer.get("native_alpha")
            _, H, W, _ = img.shape
            s = layer["scale"]
            rot = layer["rotation"]
            flip_h = layer["flip_h"]
            flip_v = layer["flip_v"]

            # 1. Flip
            if flip_h or flip_v:
                dims = []
                if flip_h: dims.append(2)
                if flip_v: dims.append(1)
                img = torch.flip(img, dims=dims)
                if native_alpha is not None:
                    native_alpha = torch.flip(native_alpha, dims=dims)

            # 2. Scale
            newH = max(1, int(H * s))
            newW = max(1, int(W * s))

            if s != 1.0:
                img_p = img.permute(0, 3, 1, 2)
                resized = F.interpolate(
                    img_p, size=(newH, newW), mode="bilinear", antialias=True
                ).permute(0, 2, 3, 1)
                if native_alpha is not None:
                    na_p = native_alpha.permute(0, 3, 1, 2)
                    native_alpha = F.interpolate(
                        na_p, size=(newH, newW), mode="bilinear", antialias=True
                    ).permute(0, 2, 3, 1)
            else:
                resized = img

            # 3. Rotation
            if rot != 0:
                theta_rad = math.radians(rot)
                cos_t = abs(math.cos(theta_rad))
                sin_t = abs(math.sin(theta_rad))
                rotW = math.ceil(newW * cos_t + newH * sin_t)
                rotH = math.ceil(newW * sin_t + newH * cos_t)
                if (rotW - newW) % 2 != 0:
                    rotW += 1
                if (rotH - newH) % 2 != 0:
                    rotH += 1

                pad_l = max(0, (rotW - newW) // 2)
                pad_r = max(0, rotW - newW - pad_l)
                pad_t = max(0, (rotH - newH) // 2)
                pad_b = max(0, rotH - newH - pad_t)

                img_p = resized.permute(0, 3, 1, 2).contiguous()
                img_p = F.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

                # alpha 遮罩（旋转 padding 区域）
                mask = torch.ones((B, 1, newH, newW), dtype=resized.dtype, device=resized.device)
                mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

                cos_θ = math.cos(theta_rad)
                sin_θ = math.sin(theta_rad)

                affine = torch.tensor([[
                    [cos_θ, sin_θ, 0],
                    [-sin_θ, cos_θ, 0],
                ]], dtype=resized.dtype, device=resized.device).repeat(B, 1, 1)

                grid = F.affine_grid(affine, (B, C, rotH, rotW), align_corners=False)
                rotated = F.grid_sample(img_p, grid, mode="bilinear", align_corners=False)
                resized = rotated.permute(0, 2, 3, 1)

                # 旋转遮罩
                mask_rotated = F.grid_sample(mask, grid, mode="bilinear", align_corners=False)
                layer_mask = mask_rotated.permute(0, 2, 3, 1)

                # 旋转原生 alpha（如果有）
                if native_alpha is not None:
                    na_p = native_alpha.permute(0, 3, 1, 2).contiguous()
                    na_p = F.pad(na_p, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)
                    na_rot = F.grid_sample(na_p, grid, mode="bilinear", align_corners=False)
                    native_alpha = na_rot.permute(0, 2, 3, 1)

                newH, newW = rotH, rotW
            else:
                layer_mask = torch.ones((B, newH, newW, 1), dtype=resized.dtype, device=resized.device)

            # 4. Position and composite
            cx = baseW // 2 + int(layer["ox"] * baseW)
            cy = baseH // 2 + int(layer["oy"] * baseH)

            x1 = cx - newW // 2
            y1 = cy - newH // 2
            x2 = x1 + newW
            y2 = y1 + newH

            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            src_x2 = newW - max(0, x2 - baseW)
            src_y2 = newH - max(0, y2 - baseH)
            dst_x2 = min(baseW, x2)
            dst_y2 = min(baseH, y2)

            sw = src_x2 - src_x1
            sh = src_y2 - src_y1
            if sw <= 0 or sh <= 0:
                continue

            src_part = resized[:, src_y1:src_y2, src_x1:src_x2, :]
            mask_part = layer_mask[:, src_y1:src_y2, src_x1:src_x2, :]

            # 合并旋转遮罩 + 原生 alpha
            if native_alpha is not None:
                na_part = native_alpha[:, src_y1:src_y2, src_x1:src_x2, :]
                combined_mask = mask_part * na_part
            else:
                combined_mask = mask_part

            dst_part = result[:, dst_y1:dst_y2, dst_x1:dst_x2, :]
            op = layer["opacity"]

            alpha = combined_mask * op
            result[:, dst_y1:dst_y2, dst_x1:dst_x2, :] = \
                src_part * alpha + dst_part * (1 - alpha)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "ZoeyMultiCanvas": ZoeyMultiCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMultiCanvas": "Zoey - 多功能画布",
}
