import torch


def parse_hex_color(hex_str):
    try:
        h = hex_str.lstrip('#').strip()
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError
        return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    except Exception:
        return 0.5, 0.5, 0.5


def make_feather_mask(h, w, feather, device, dtype):
    """Create a distance-to-edge alpha mask for feathering."""
    if feather <= 0:
        return None
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    dist = torch.minimum(
        torch.minimum(y[:, None], (h - 1 - y)[:, None]),
        torch.minimum(x[None, :], (w - 1 - x)[None, :]),
    )
    alpha = torch.clamp(dist / feather, 0, 1)
    return alpha[None, :, :, None]  # (1, h, w, 1)


def blend_feather(dst, src, feather, dst_y1, dst_x1, h, w):
    """Blend src into dst with feather at edges."""
    if feather <= 0:
        dst[:, dst_y1:dst_y1 + h, dst_x1:dst_x1 + w, :] = src
        return
    alpha = make_feather_mask(h, w, feather, dst.device, dst.dtype)
    dst_part = dst[:, dst_y1:dst_y1 + h, dst_x1:dst_x1 + w, :]
    dst[:, dst_y1:dst_y1 + h, dst_x1:dst_x1 + w, :] = src * alpha + dst_part * (1 - alpha)


class ZoeyOutpaintFrame:
    """外扩画布框架 - 通过拖拽交互控制画布外扩/裁剪"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_left": ("FLOAT", {"default": -0.1, "min": -5.0, "max": 1.0, "step": 0.001}),
                "frame_top": ("FLOAT", {"default": -0.1, "min": -5.0, "max": 1.0, "step": 0.001}),
                "frame_right": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 5.0, "step": 0.001}),
                "frame_bottom": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 5.0, "step": 0.001}),
                "填充颜色": ("STRING", {"multiline": False, "default": "#808080"}),
                "fill_mode": ("BOOLEAN", {"default": True}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "outpaint"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def outpaint(self, image, frame_left, frame_top, frame_right, frame_bottom, 填充颜色, fill_mode=True, feather=0):
        fill_rgb = parse_hex_color(填充颜色)
        B, H, W, C = image.shape
        l = int(round(frame_left * W))
        t = int(round(frame_top * H))
        r = int(round(frame_right * W))
        b = int(round(frame_bottom * H))

        fw = r - l
        fh = b - t

        # 填充模式 + 框在图像内部 → 输出完整图像尺寸，框外填充颜色
        if fill_mode and l >= 0 and t >= 0 and r <= W and b <= H:
            out = torch.empty((B, H, W, 3), dtype=image.dtype, device=image.device)
            out[:, :, :, 0] = fill_rgb[0]
            out[:, :, :, 1] = fill_rgb[1]
            out[:, :, :, 2] = fill_rgb[2]
            if feather > 0:
                blend_feather(out, image[:, t:b, l:r, :3], feather, t, l, fh, fw)
            else:
                out[:, t:b, l:r, :] = image[:, t:b, l:r, :3]
            return (out,)

        out_w = r - l
        out_h = b - t
        if out_w <= 0 or out_h <= 0:
            return (image,)

        out = torch.empty((B, out_h, out_w, 3), dtype=image.dtype, device=image.device)
        out[:, :, :, 0] = fill_rgb[0]
        out[:, :, :, 1] = fill_rgb[1]
        out[:, :, :, 2] = fill_rgb[2]

        src_x1 = max(0, l)
        src_y1 = max(0, t)
        src_x2 = min(W, r)
        src_y2 = min(H, b)
        dst_x1 = max(0, -l)
        dst_y1 = max(0, -t)

        if src_x2 > src_x1 and src_y2 > src_y1:
            sh = src_y2 - src_y1
            sw = src_x2 - src_x1
            src_part = image[:, src_y1:src_y2, src_x1:src_x2, :3]
            blend_feather(out, src_part, feather, dst_y1, dst_x1, sh, sw)

        return (out,)


NODE_CLASS_MAPPINGS = {
    "ZoeyOutpaintFrame": ZoeyOutpaintFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyOutpaintFrame": "Zoey - 框选裁剪/外扩",
}
