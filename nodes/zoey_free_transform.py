import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json


class ZoeyFreeTransform:
    """自由变换 - 拖拽四个角点进行透视/扭曲变换"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "corners": ("STRING", {"multiline": False, "default": "{}", "dynamicPrompts": False}),
                "grid_overlay": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "warp"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def warp(self, image, corners="{}", grid_overlay=True):
        try:
            cfg = json.loads(corners)
        except Exception:
            cfg = {}

        tl = cfg.get("tl", {"x": 0, "y": 0})
        tr = cfg.get("tr", {"x": 1, "y": 0})
        br = cfg.get("br", {"x": 1, "y": 1})
        bl = cfg.get("bl", {"x": 0, "y": 1})

        pts = [tl["x"], tl["y"], tr["x"], tr["y"], br["x"], br["y"], bl["x"], bl["y"]]
        return (self._warp_perspective(image, pts),)

    def _warp_perspective(self, img, corners_flat):
        B, H, W, C = img.shape
        src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        c = np.array(corners_flat, dtype=np.float32).reshape(4, 2)
        c[:, 0] *= W
        c[:, 1] *= H
        dst = c.astype(np.float32)

        try:
            M = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return img
        M_inv = np.linalg.inv(M)

        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ones = np.ones_like(gx)
        sx = M_inv[0, 0] * gx + M_inv[0, 1] * gy + M_inv[0, 2] * ones
        sy = M_inv[1, 0] * gx + M_inv[1, 1] * gy + M_inv[1, 2] * ones
        sw = M_inv[2, 0] * gx + M_inv[2, 1] * gy + M_inv[2, 2] * ones
        mx = (sx / sw / W * 2 - 1).astype(np.float32)
        my = (sy / sw / H * 2 - 1).astype(np.float32)

        grid = torch.from_numpy(np.stack([mx, my], axis=-1)).unsqueeze(0)
        grid = grid.to(img.device).repeat(B, 1, 1, 1)

        img_p = img.permute(0, 3, 1, 2)
        warped = F.grid_sample(img_p, grid, mode="bilinear", align_corners=False)
        return warped.permute(0, 2, 3, 1)


NODE_CLASS_MAPPINGS = {
    "ZoeyFreeTransform": ZoeyFreeTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyFreeTransform": "Zoey - 自由变换",
}
