# -*- coding: utf-8 -*-
"""
Zoey 局部重绘（面部 / 手部）
检测面部、手部区域 → 按框裁剪并放大 → 局部采样重绘 → 羽化贴回原图。

检测优先级：
  面部 = ultralytics 里的 face_*yolo*.pt（有就用），否则 OpenCV Haar 级联
  手部 = ultralytics 里的 hand_*yolo*.pt（有就用），否则 mediapipe Hands
采样用 ComfyUI 自带的 common_ksampler，模型 / VAE / 条件全部从外部接入。
"""

import logging
import os

import numpy as np
import torch

import comfy.utils
import comfy.model_management
import folder_paths
from comfy.samplers import KSAMPLER_NAMES, SCHEDULER_NAMES

logger = logging.getLogger("ZoeyTool")

try:
    import cv2
except ImportError:
    cv2 = None

_YOLO_CACHE = {}
_HANDS_CACHE = {}
_HAAR_CACHE = {}

DETECTORS = ["面部", "手部", "面部 + 手部"]


# ────────────────────────── 检测 ──────────────────────────

def _iter_ultralytics():
    roots = []
    try:
        roots.extend(folder_paths.get_folder_paths("ultralytics"))
    except Exception:
        pass
    roots.append(os.path.join(folder_paths.models_dir, "ultralytics"))
    seen = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".pt"):
                    yield os.path.join(dirpath, f)


def _find_yolo(kind):
    for path in _iter_ultralytics():
        name = os.path.basename(path).lower()
        if kind in name and "yolo" in name:
            return path
    return None


def _clamp_box(box, W, H):
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
    y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


def _detect_yolo(kind, rgb, conf):
    """返回检测框列表；未安装对应 yolo 模型时返回 None。"""
    path = _find_yolo(kind)
    if path is None:
        return None
    try:
        from PIL import Image
        from ultralytics import YOLO
        yolo = _YOLO_CACHE.get(path)
        if yolo is None:
            yolo = YOLO(path)
            _YOLO_CACHE[path] = yolo
        res = yolo.predict(Image.fromarray(rgb), conf=float(conf), verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        H, W = rgb.shape[:2]
        out = [_clamp_box(b, W, H) for b in boxes.xyxy.cpu().numpy().tolist()]
        return [b for b in out if b is not None]
    except Exception as e:
        logger.warning(f"[ZoeyLocalDetailer] YOLO 检测失败({kind}): {e}")
        return []


def _detect_haar(rgb, conf):
    if cv2 is None:
        return []
    casc = _HAAR_CACHE.get("frontal")
    if casc is None:
        casc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _HAAR_CACHE["frontal"] = casc
    if casc.empty():
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = casc.detectMultiScale(
        gray, scaleFactor=1.12,
        minNeighbors=max(3, int(round(float(conf) * 8))),
        minSize=(24, 24))
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]


def _detect_faces(rgb, conf):
    boxes = _detect_yolo("face", rgb, conf)
    if boxes is not None:
        return boxes
    return _detect_haar(rgb, conf)


def _detect_hands(rgb, conf):
    boxes = _detect_yolo("hand", rgb, conf)
    if boxes is not None:
        return boxes
    try:
        import mediapipe as mp
    except ImportError:
        return []
    key = (round(float(conf), 2), 2)
    hands = _HANDS_CACHE.get(key)
    if hands is None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=float(conf))
        _HANDS_CACHE[key] = hands
    try:
        res = hands.process(np.ascontiguousarray(rgb))
    except Exception as e:
        logger.warning(f"[ZoeyLocalDetailer] mediapipe 手部检测失败: {e}")
        return []
    if not res.multi_hand_landmarks:
        return []
    H, W = rgb.shape[:2]
    out = []
    for lm in res.multi_hand_landmarks:
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]
        b = _clamp_box((min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H), W, H)
        if b is not None:
            out.append(b)
    return out


# ────────────────────────── 框处理 ──────────────────────────

def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _merge_boxes(boxes, thr=0.55):
    kept = []
    for b in sorted(boxes, key=_box_area, reverse=True):
        if all(_iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def _expand_box(box, pad, W, H):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1 = max(0, int(round(cx - bw * (1.0 + 2.0 * pad) / 2.0)))
    ny1 = max(0, int(round(cy - bh * (1.0 + 2.0 * pad) / 2.0)))
    nx2 = min(W, int(round(cx + bw * (1.0 + 2.0 * pad) / 2.0)))
    ny2 = min(H, int(round(cy + bh * (1.0 + 2.0 * pad) / 2.0)))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return (nx1, ny1, nx2, ny2)


def _boxes_from_mask(mask_np, min_area=64):
    if cv2 is None:
        return []
    binary = (mask_np > 0.5).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and w >= 4 and h >= 4:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _feather_mask(ch, cw, f, device, dtype):
    f = min(int(f), cw // 2, ch // 2)
    if f <= 0:
        return None
    ramp = torch.linspace(0.0, 1.0, f + 2, device=device, dtype=dtype)[1:-1]
    m = torch.ones(ch, cw, device=device, dtype=dtype)
    m[:f, :] *= ramp[:, None]
    m[-f:, :] *= ramp.flip(0)[:, None]
    m[:, :f] *= ramp[None, :]
    m[:, -f:] *= ramp.flip(0)[None, :]
    return m


# ────────────────────────── 节点 ──────────────────────────

class ZoeyLocalDetailer:
    """检测面部/手部 → 裁剪放大 → 局部采样 → 羽化贴回。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "detector": (DETECTORS, {"default": "面部 + 手部"}),
                "conf": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                                   "tooltip": "检测置信度阈值"}),
                "max_regions": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1,
                                        "tooltip": "每张图最多重绘几处，按面积从大到小取；0 = 不限"}),
                "padding": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "检测框四周外扩比例，给模型留上下文"}),
                "work_size": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 8,
                                      "tooltip": "裁剪区长边放大到的采样分辨率"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (KSAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "0.3–0.6 保结构；太高会换脸/改手型"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1,
                                    "tooltip": "贴回时边缘羽化像素，0 为硬贴"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "detail"
    CATEGORY = "Zoey工具集/图像编辑"
    DESCRIPTION = ("检测面部/手部（可从模型目录读 face_yolo*.pt / hand_yolo*.pt，"
                   "没有则用 Haar / mediapipe），裁剪放大后局部采样，再羽化贴回原图。\n"
                   "可接 mask：遮罩的连通区域会作为额外重绘区域。")

    def _sample(self, model, vae, positive, negative, crop, denoise, seed,
                steps, cfg, sampler_name, scheduler, work_size):
        import nodes as comfy_nodes  # common_ksampler

        _, ch, cw, _ = crop.shape
        k = float(work_size) / max(ch, cw)
        # 取 16 的倍数，兼容 Flux / Qwen-Image 这类 2x2 patch 的模型
        ww = max(16, int(round(cw * k / 16)) * 16)
        wh = max(16, int(round(ch * k / 16)) * 16)

        work = comfy.utils.common_upscale(
            crop.movedim(-1, 1), ww, wh, "lanczos", "disabled").movedim(1, -1)

        latent = {"samples": vae.encode(work)}
        (out,) = comfy_nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=denoise)

        dec = vae.decode(out["samples"])
        if dec.shape[1:3] != (ch, cw):
            dec = comfy.utils.common_upscale(
                dec.movedim(-1, 1), cw, ch, "lanczos", "disabled").movedim(1, -1)
        return dec.to(device=crop.device, dtype=crop.dtype)

    def detail(self, image, model, vae, positive, negative, detector="面部 + 手部",
               conf=0.5, max_regions=4, padding=0.5, work_size=1024, steps=20, cfg=7.0,
               sampler_name="euler", scheduler="normal", denoise=0.45, seed=0,
               feather=12, mask=None):
        B, H, W, _ = image.shape
        out = image.clone()
        out_mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

        pbar = comfy.utils.ProgressBar(B)
        processed = 0
        seed_cursor = int(seed)

        for b in range(B):
            comfy.model_management.throw_exception_if_processing_interrupted()

            frame = image[b:b + 1]
            rgb = (frame[0].detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            boxes = []
            if detector in ("面部", "面部 + 手部"):
                boxes += _detect_faces(rgb, conf)
            if detector in ("手部", "面部 + 手部"):
                boxes += _detect_hands(rgb, conf)

            if mask is not None:
                if mask.dim() == 3:
                    m = mask[b if b < mask.shape[0] else 0]
                else:
                    m = mask
                boxes += _boxes_from_mask(m.detach().float().cpu().numpy().squeeze())

            regions = []
            for box in _merge_boxes(boxes):
                eb = _expand_box(box, padding, W, H)
                if eb is not None:
                    regions.append(eb)
            regions.sort(key=_box_area, reverse=True)
            if max_regions > 0:
                regions = regions[:max_regions]

            for x1, y1, x2, y2 in regions:
                comfy.model_management.throw_exception_if_processing_interrupted()
                ch, cw = y2 - y1, x2 - x1
                crop = frame[:, y1:y2, x1:x2, :]

                dec = self._sample(
                    model, vae, positive, negative, crop, denoise,
                    seed_cursor, steps, cfg, sampler_name, scheduler, work_size)
                seed_cursor += 1

                fm = _feather_mask(ch, cw, feather, crop.device, crop.dtype)
                if fm is None:
                    out[b, y1:y2, x1:x2, :] = dec[0]
                    out_mask[b, y1:y2, x1:x2] = 1.0
                else:
                    w = fm.unsqueeze(-1)
                    out[b, y1:y2, x1:x2, :] = dec[0] * w + crop[0] * (1.0 - w)
                    out_mask[b, y1:y2, x1:x2] = torch.maximum(
                        out_mask[b, y1:y2, x1:x2], fm)

                processed += 1
                del crop, dec, fm

            pbar.update(1)

        info = f"{detector}：重绘 {processed} 处"
        logger.info(f"[ZoeyLocalDetailer] {info}")
        return (out, out_mask, info)


NODE_CLASS_MAPPINGS = {
    "ZoeyLocalDetailer": ZoeyLocalDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyLocalDetailer": "Zoey - 局部重绘（面部/手部）",
}
