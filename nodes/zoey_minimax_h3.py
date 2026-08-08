# -*- coding: utf-8 -*-
"""Zoey MiniMax H3 ref2va 包装节点。

在 MiniMaxH3ReferenceToVideo 基础上增加 Seedance 2.0 风格的 @ 引用语法：
提示词里用 @P1/@V1/@A1 引用已连接的参考素材，自动展开为
<Picture N>/<Video N>/<Audio N>，并可选生成引用声明行。其余行为与官方节点一致。
"""

import json
import math

from .zoey_minimax_h3_tags import build_declaration, count_refs, expand_at_tags

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    _H3_AVAILABLE = True
except Exception:
    _H3_AVAILABLE = False

_IMAGE_SLOTS = 9
_VIDEO_SLOTS = 3
_AUDIO_SLOTS = 3

# 分辨率档位：像素标签（16:9 参考尺寸，来自官方 megapixels 表，已去重）
_RESOLUTION_TABLE = {
    "608*352": (608, 352), "736*416": (736, 416), "864*480": (864, 480),
    "960*544": (960, 544), "1056*608": (1056, 608), "1152*640": (1152, 640),
    "1216*672": (1216, 672), "1280*736": (1280, 736), "1344*768": (1344, 768),
    "1376*768": (1376, 768), "1504*832": (1504, 832), "1664*928": (1664, 928),
    "1824*1024": (1824, 1024), "1920*1088": (1920, 1088),
}
_ASPECT_RATIO = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0,
                 "4:3": 4 / 3, "3:4": 3 / 4, "2:3": 2 / 3, "3:2": 3 / 2}
_CANVAS_MULTIPLE = 32


def _first_ref_image(ref_images):
    for value in (ref_images or {}).values():
        if value is not None:
            return value
    return None


def _round_to_multiple(value, multiple):
    # 四舍五入到 multiple 的整数倍（round half up，与前端 Math.round 一致）
    return max(multiple, int(value / multiple + 0.5) * multiple)


def _compute_canvas(ref_images, resolution, aspect):
    """根据分辨率档位/比例/参考图算出宽高（32 倍数）。"""
    img = _first_ref_image(ref_images)
    img_h = img_w = None
    if img is not None:
        img_h, img_w = img.shape[1], img.shape[2]

    if aspect == "自动" and img is not None:
        ratio = img_w / img_h
    elif aspect in _ASPECT_RATIO:
        ratio = _ASPECT_RATIO[aspect]
    else:
        ratio = 16 / 9

    if resolution in _RESOLUTION_TABLE:
        # 像素档：16:9 时精确用参考尺寸，其他比例按面积换算
        base_w, base_h = _RESOLUTION_TABLE[resolution]
        if abs(ratio - 16 / 9) < 1e-6:
            cw, ch = base_w, base_h
        else:
            area = base_w * base_h
            cw = math.sqrt(area * ratio)
            ch = math.sqrt(area / ratio)
    else:  # 分辨率自动
        short = min(img_w, img_h) if img is not None else 720
        if ratio >= 1.0:
            cw, ch = short * ratio, short
        else:
            cw, ch = short, short / ratio

    return _round_to_multiple(cw, _CANVAS_MULTIPLE), _round_to_multiple(ch, _CANVAS_MULTIPLE)


def _frame_count(duration):
    """秒 -> 帧数，对齐到模型 17k+5 网格（24fps）。"""
    n = max(5, round(duration * 24))
    while n % 17 != 5:
        n += 1
    return n


def _compose_director(shots_json, counts):
    """导演台：把分镜列表拼成 CUT/TRANSITION 提示词 + 总时长（秒）+ 参考素材说明。

    数据结构：{"ref_decl": "参考素材说明(可空)", "shots": [{"prompt", "duration", "transition"}]}
    兼容旧格式：直接传镜头数组。
    """
    try:
        data = json.loads(shots_json or "{}")
    except Exception:
        data = {}
    ref_decl = ""
    if isinstance(data, dict):
        shots = data.get("shots") or []
        ref_decl = str(data.get("ref_decl", "")).strip()
    elif isinstance(data, list):
        shots = data
    else:
        shots = []
    if not isinstance(shots, list) or not shots:
        raise ValueError("导演台：请至少添加一个镜头（点击「＋ 添加镜头」）")

    parts = []
    total = 0.0
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict) or not str(shot.get("prompt", "")).strip():
            raise ValueError(f"导演台：CUT {i + 1} 缺少提示词")
        prompt_text = expand_at_tags(str(shot.get("prompt", "")).strip(), counts)
        if i == 0:
            parts.append(f"CUT 1: {prompt_text}")
        else:
            transition = str(shot.get("transition", "")).strip()
            if transition:
                parts.append(f"TRANSITION: {transition}")
            parts.append(f"CUT {i + 1}: {prompt_text}")
        try:
            total += float(shot.get("duration", 5.0))
        except Exception:
            total += 5.0

    if ref_decl:
        ref_decl = expand_at_tags(ref_decl, counts)
    return "\n".join(parts), min(total, 15.0), ref_decl


class ZoeyMiniMaxH3ReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(_IMAGE_SLOTS):
            optional[f"ref_image_{i}"] = ("IMAGE",)
        for i in range(_VIDEO_SLOTS):
            optional[f"ref_video_{i}"] = ("IMAGE",)
            optional[f"ref_video_audio_{i}"] = ("AUDIO",)
        for i in range(_AUDIO_SLOTS):
            optional[f"ref_audio_{i}"] = ("AUDIO",)
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "placeholder": "用 @P1/@V1/@A1 引用参考素材，如：@P1 的男人…",
                }),
                "resolution": (["自动", "608*352", "736*416", "864*480", "960*544", "1056*608",
                                "1152*640", "1216*672", "1280*736", "1344*768", "1376*768",
                                "1504*832", "1664*928", "1824*1024", "1920*1088"], {"default": "1280*736"}),
                "aspect": (["自动", "16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2"], {"default": "自动"}),
                "duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "auto_declaration": ("BOOLEAN", {
                    "default": True,
                    "label_on": "生成",
                    "label_off": "关闭",
                }),
                "director_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "开启",
                    "label_off": "关闭",
                }),
                "director_shots": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "placeholder": "导演台分镜数据（前端自动生成）",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "LATENT", "提示词(已转换)")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"

    def process(self, clip, vae, audio_vae, prompt, resolution, aspect, duration,
                ref_image_size, auto_declaration, director_mode, director_shots, **refs):
        ref_images = {}
        for i in range(_IMAGE_SLOTS):
            ref_images[f"ref_image_{i}"] = refs.get(f"ref_image_{i}")

        ref_videos = {}
        ref_video_audios = {}
        for i in range(_VIDEO_SLOTS):
            ref_videos[f"ref_video_{i}"] = refs.get(f"ref_video_{i}")
            ref_video_audios[f"ref_video_audio_{i}"] = refs.get(f"ref_video_audio_{i}")

        ref_audios = {}
        for i in range(_AUDIO_SLOTS):
            ref_audios[f"ref_audio_{i}"] = refs.get(f"ref_audio_{i}")

        counts = count_refs(
            image_count=sum(1 for i in range(_IMAGE_SLOTS) if refs.get(f"ref_image_{i}") is not None),
            video_count=sum(1 for i in range(_VIDEO_SLOTS) if refs.get(f"ref_video_{i}") is not None),
            video_audio_count=sum(1 for i in range(_VIDEO_SLOTS) if refs.get(f"ref_video_audio_{i}") is not None),
            audio_count=sum(1 for i in range(_AUDIO_SLOTS) if refs.get(f"ref_audio_{i}") is not None),
        )

        width, height = _compute_canvas(ref_images, resolution, aspect)

        if director_mode:
            cut_text, total_duration, ref_decl = _compose_director(director_shots, counts)
            if ref_decl:
                # 用户写了参考素材说明（@ 已展开），优先用它
                composed = ref_decl + "\n" + cut_text
            else:
                composed = cut_text
                if auto_declaration:
                    declaration = build_declaration(counts)
                    if declaration:
                        composed = declaration + "\n" + composed
            prompt_text = composed
            length = _frame_count(total_duration)
        else:
            prompt_text = expand_at_tags(prompt, counts)
            if auto_declaration:
                declaration = build_declaration(counts)
                if declaration:
                    prompt_text = declaration + "\n" + prompt_text
            length = _frame_count(duration)

        outputs = MiniMaxH3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt_text, width, height, length,
            ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
        )
        return (outputs.args[0], outputs.args[1], prompt_text)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if _H3_AVAILABLE:
    NODE_CLASS_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = ZoeyMiniMaxH3ReferenceToVideo
    NODE_DISPLAY_NAME_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = "Zoey - MiniMax H3 参考转视频 (@)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
