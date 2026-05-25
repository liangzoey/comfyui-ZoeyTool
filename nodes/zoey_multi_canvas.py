import torch
import json
import math
import torch.nn.functional as F


class ZoeyMultiCanvas:
    """多功能画布 - 多层图像叠加合成（支持旋转/翻转）"""
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("合成图像",)
    FUNCTION = "composite"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def composite(self, image1, image2=None, image3=None, image4=None, image5=None, layer_config="{}"):
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

        for key, img in inputs:
            if img is None:
                continue
            cfg = config.get(key, {})
            layers.append({
                "img": img,
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

        B, baseH, baseW, C = layers[0]["img"].shape
        dtype = layers[0]["img"].dtype
        device = layers[0]["img"].device

        result = torch.zeros((B, baseH, baseW, C), dtype=dtype, device=device)

        for layer in layers:
            if not layer["visible"]:
                continue

            img = layer["img"]
            _, H, W, _ = img.shape
            s = layer["scale"]
            rot = layer["rotation"]
            flip_h = layer["flip_h"]
            flip_v = layer["flip_v"]

            # 1. Apply flip
            if flip_h or flip_v:
                dims = []
                if flip_h: dims.append(2)
                if flip_v: dims.append(1)
                img = torch.flip(img, dims=dims)

            # 2. Scale
            newH = max(1, int(H * s))
            newW = max(1, int(W * s))

            if s != 1.0:
                img_p = img.permute(0, 3, 1, 2)
                cur = F.interpolate(
                    img_p, size=(newH, newW), mode="bilinear", antialias=True
                ).permute(0, 2, 3, 1)
            else:
                cur = img

            # 3. Rotation (fixed: rot90 for 90° multiples, proper affine for others)
            if rot != 0:
                if abs(rot) % 90 == 0:
                    k = int(round(rot / 90)) % 4
                    cur = torch.rot90(cur, k=k, dims=[1, 2])
                    newH, newW = cur.shape[1], cur.shape[2]
                else:
                    theta_rad = math.radians(rot)
                    cos_t = abs(math.cos(theta_rad))
                    sin_t = abs(math.sin(theta_rad))
                    rotW = int(newW * cos_t + newH * sin_t + 0.5)
                    rotH = int(newW * sin_t + newH * cos_t + 0.5)

                    pad_l = max(0, (rotW - newW) // 2)
                    pad_r = max(0, rotW - newW - pad_l)
                    pad_t = max(0, (rotH - newH) // 2)
                    pad_b = max(0, rotH - newH - pad_t)

                    img_p = cur.permute(0, 3, 1, 2).contiguous()
                    img_p = F.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

                    cos_θ = math.cos(theta_rad)
                    sin_θ = math.sin(theta_rad)

                    affine = torch.tensor([[
                        [cos_θ, sin_θ, 0],
                        [-sin_θ, cos_θ, 0],
                    ]], dtype=cur.dtype, device=cur.device).repeat(B, 1, 1)

                    grid = F.affine_grid(affine, (B, C, rotH, rotW), align_corners=False)
                    rotated = F.grid_sample(img_p, grid, mode="bilinear", align_corners=False)
                    cur = rotated.permute(0, 2, 3, 1)
                    newH, newW = rotH, rotW

            # 4. Position and composite onto fixed canvas
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

            src_part = cur[:, src_y1:src_y2, src_x1:src_x2, :]
            dst_part = result[:, dst_y1:dst_y2, dst_x1:dst_x2, :]
            op = layer["opacity"]

            result[:, dst_y1:dst_y2, dst_x1:dst_x2, :] = \
                src_part * op + dst_part * (1 - op)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "ZoeyMultiCanvas": ZoeyMultiCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMultiCanvas": "Zoey - 多功能画布",
}
