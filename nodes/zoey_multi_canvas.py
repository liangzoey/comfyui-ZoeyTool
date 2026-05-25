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

        # ── First pass: compute global bounding box of all visible layers ──
        bx1, by1, bx2, by2 = 0, 0, baseW, baseH
        layer_render_info = []

        for layer in layers:
            info = {"visible": layer["visible"]}
            if not layer["visible"]:
                layer_render_info.append(info)
                continue

            img = layer["img"]
            _, H, W, _ = img.shape
            s = layer["scale"]
            rot = layer["rotation"]

            scW = max(1, int(W * s))
            scH = max(1, int(H * s))

            if rot != 0:
                if abs(rot) % 90 == 0:
                    k = (-int(round(rot / 90))) % 4
                    if k in (1, 3):
                        bbW, bbH = scH, scW
                    else:
                        bbW, bbH = scW, scH
                else:
                    rad = math.radians(rot)
                    ct = abs(math.cos(rad))
                    st = abs(math.sin(rad))
                    bbW = int(scW * ct + scH * st + 0.5)
                    bbH = int(scW * st + scH * ct + 0.5)
            else:
                bbW, bbH = scW, scH

            info["bbW"] = bbW
            info["bbH"] = bbH

            # Position in original canvas coords
            cx = baseW // 2 + int(layer["ox"] * baseW)
            cy = baseH // 2 + int(layer["oy"] * baseH)

            info["cx"] = cx
            info["cy"] = cy

            lx1 = cx - bbW // 2
            ly1 = cy - bbH // 2
            lx2 = lx1 + bbW
            ly2 = ly1 + bbH

            bx1 = min(bx1, lx1)
            by1 = min(by1, ly1)
            bx2 = max(bx2, lx2)
            by2 = max(by2, ly2)

            layer_render_info.append(info)

        canvasW = bx2 - bx1
        canvasH = by2 - by1
        offset_x = -bx1
        offset_y = -by1

        # Always output 3-channel RGB
        result = torch.zeros((B, canvasH, canvasW, 3), dtype=dtype, device=device)

        # ── Second pass: render each layer ──
        for idx, layer in enumerate(layers):
            if not layer["visible"]:
                continue

            img = layer["img"]
            _, H, W, _ = img.shape
            s = layer["scale"]
            rot = layer["rotation"]
            flip_h = layer["flip_h"]
            flip_v = layer["flip_v"]
            info = layer_render_info[idx]
            bbW = info["bbW"]
            bbH = info["bbH"]

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

            # 3. Rotation (CW positive, matching preview)
            cur_mask = None
            if rot != 0:
                if abs(rot) % 90 == 0:
                    k = (-int(round(rot / 90))) % 4
                    cur = torch.rot90(cur, k=k, dims=[1, 2])
                else:
                    theta_rad = math.radians(rot)
                    cos_t = abs(math.cos(theta_rad))
                    sin_t = abs(math.sin(theta_rad))

                    cos_θ = math.cos(theta_rad)
                    sin_θ = math.sin(theta_rad)

                    affine = torch.tensor([[
                        [cos_θ, sin_θ, 0],
                        [-sin_θ, cos_θ, 0],
                    ]], dtype=cur.dtype, device=cur.device).repeat(B, 1, 1)

                    grid = F.affine_grid(affine, (B, C, bbH, bbW), align_corners=False)

                    img_p = cur.permute(0, 3, 1, 2).contiguous()
                    rotated = F.grid_sample(img_p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                    cur = rotated.permute(0, 2, 3, 1)

                    # Validity mask: 1 where grid coords are within original image bounds
                    cur_mask = ((grid[..., 0:1] >= -1.0) & (grid[..., 0:1] <= 1.0) &
                                (grid[..., 1:2] >= -1.0) & (grid[..., 1:2] <= 1.0)).float()

            # 4. Position and composite onto expanded canvas
            cx = info["cx"] + offset_x
            cy = info["cy"] + offset_y

            x1 = cx - bbW // 2
            y1 = cy - bbH // 2
            x2 = x1 + bbW
            y2 = y1 + bbH

            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            src_x2 = bbW - max(0, x2 - canvasW)
            src_y2 = bbH - max(0, y2 - canvasH)
            dst_x2 = min(canvasW, x2)
            dst_y2 = min(canvasH, y2)

            sw = src_x2 - src_x1
            sh = src_y2 - src_y1
            if sw <= 0 or sh <= 0:
                continue

            src_part = cur[:, src_y1:src_y2, src_x1:src_x2, :]
            dst_part = result[:, dst_y1:dst_y2, dst_x1:dst_x2, :]
            op = layer["opacity"]

            # Per-pixel alpha from 4-channel RGBA input
            cur_alpha = src_part[:, :, :, 3:4] if cur.shape[3] >= 4 else 1.0
            cur_rgb = src_part[:, :, :, :3]

            if cur_mask is not None:
                mask_part = cur_mask[:, src_y1:src_y2, src_x1:src_x2, :]
                alpha = cur_alpha * op * mask_part
            else:
                alpha = cur_alpha * op

            result[:, dst_y1:dst_y2, dst_x1:dst_x2, :] = \
                cur_rgb * alpha + dst_part * (1 - alpha)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "ZoeyMultiCanvas": ZoeyMultiCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMultiCanvas": "Zoey - 多功能画布",
}
