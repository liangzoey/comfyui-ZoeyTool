# -*- coding: utf-8 -*-
"""Zoey MiniMax H3 ref2va 包装节点。

在 MiniMaxH3ReferenceToVideo 基础上增加 Seedance 2.0 风格的 @ 引用语法：
提示词里用 @P1/@V1/@A1 引用已连接的参考素材，自动展开为
<Picture N>/<Video N>/<Audio N>，并可选生成引用声明行。其余行为与官方节点一致。
"""

import json
import math
import os
import re

from .zoey_minimax_h3_tags import build_declaration, count_refs, expand_at_tags

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo
    _H3_AVAILABLE = True
except Exception:
    _H3_AVAILABLE = False

# SageAttention：是否可用与具体实现，来自核心 attention 模块（sageattention 包已装则启用）
try:
    from comfy.ldm.modules.attention import attention_sage, SAGE_ATTENTION_IS_AVAILABLE
    _SAGE_AVAILABLE = SAGE_ATTENTION_IS_AVAILABLE
except Exception:
    attention_sage = None
    _SAGE_AVAILABLE = False


def _apply_sage_attention(model):
    """在模型克隆上启用 SageAttention（仅影响该克隆，不改全局 optimized_attention）。

    通过 set_model_optimized_attention 写入 transformer_options 的
    optimized_attention_override，MiniMaxH3 Attention 层采样时会走 sageattn。
    """
    if not _SAGE_AVAILABLE or attention_sage is None:
        raise ValueError("已开启 sage_attention，但 sageattention 包不可用（请先 pip install sageattention）")
    m = model.clone()
    to = m.model_options.get("transformer_options", {}).copy()
    m.model_options["transformer_options"] = to
    m.set_model_optimized_attention(attention_sage)
    return m


_NONE = "无"


def _vae_names():
    """VAE 文件列表（复用官方 VAELoader 的分类，含 video_taes / pixel_space）。"""
    import nodes
    return nodes.VAELoader.vae_list(nodes.VAELoader)


def _load_unet(unet_name):
    import folder_paths
    import comfy.sd
    path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    return comfy.sd.load_diffusion_model(path)


def _load_clip(clip_name):
    import folder_paths
    import comfy.sd
    clip_type = getattr(comfy.sd.CLIPType, "MINIMAX", comfy.sd.CLIPType.STABLE_DIFFUSION)
    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    return comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )


def _load_vae(vae_name):
    import nodes
    return nodes.VAELoader().load_vae(vae_name)[0]


def _apply_lora(model, clip, lora_name, strength_model, strength_clip):
    if not lora_name or lora_name == _NONE or (strength_model == 0 and strength_clip == 0):
        return model, clip
    import folder_paths
    import comfy.utils
    import comfy.sd
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                         lora_metadata=metadata)


def _parse_loras(loras_json):
    """解析前端 LoRA 列表 JSON 为 [(lora, strength_model, strength_clip), ...]。"""
    try:
        data = json.loads(loras_json or "[]")
    except Exception:
        data = []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        lora = str(item.get("lora", "") or "").strip()
        if not lora or lora == _NONE:
            continue
        try:
            sm = float(item.get("model", 1.0) or 1.0)
        except (TypeError, ValueError):
            sm = 1.0
        try:
            sc = float(item.get("clip", 1.0) or 1.0)
        except (TypeError, ValueError):
            sc = 1.0
        out.append((lora, sm, sc))
    return out


def _parse_director_shots(director_shots, prompt):
    """解析分镜 JSON；空则退回单镜头 prompt。返回 shots: [{prompt,duration,...}]。"""
    data = None
    try:
        data = json.loads(director_shots or "[]")
    except Exception:
        data = None
    shots = None
    if isinstance(data, dict):
        shots = data.get("shots")
    elif isinstance(data, list):
        shots = data
    if isinstance(shots, list) and shots:
        return shots
    return [{"prompt": (prompt or "").strip() or "场景", "duration": 5.0}]


def _segment_shots(shots, segment_mode, segment_size, hard_max=15.0):
    """把 shots 分成 ≤segment_size 的段（长视频分段）。

    segment_mode:
      "手动(每镜一段)" -> 每镜成一段，时长 clamp 到 hard_max。
      "自动"        -> 超长镜(>segment_size)拆成多个子段，相邻短镜合并成≤segment_size 段。
    返回 [{"shots":[...], "duration": 秒}]。
    """
    if not shots:
        return []
    size = float(segment_size or hard_max)
    segs = []
    if segment_mode and str(segment_mode).startswith("手动"):
        for s in shots:
            d = float(s.get("duration", 5.0) or 5.0)
            segs.append({"shots": [s], "duration": min(d, hard_max)})
        return segs
    cur = {"shots": [], "duration": 0.0}
    for s in shots:
        d = float(s.get("duration", 5.0) or 5.0)
        if d > size:
            if cur["shots"]:
                segs.append(cur)
                cur = {"shots": [], "duration": 0.0}
            while d > size:
                segs.append({"shots": [s], "duration": size})
                d -= size
            if d > 0:
                segs.append({"shots": [s], "duration": d})
        else:
            if cur["shots"] and cur["duration"] + d > size:
                segs.append(cur)
                cur = {"shots": [], "duration": 0.0}
            cur["shots"].append(s)
            cur["duration"] += d
    if cur["shots"]:
        segs.append(cur)
    return segs


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


_CAST_RE = re.compile(r"@(?:[Cc]|char)(\d+)")
_LIB_RE = re.compile(r"@[Ll](\d+)")
_LIB_SUBFOLDER = "zoey_library"

# 成对引号（ASCII 双引号 / 弯双引号 / 「」 / 『』）内的内容视为台词
_DIALOGUE_QUOTE_RE = re.compile(
    '"([^"\n]*)"'
    '|“([^”\n]*)”'
    '|「([^」\n]*)」'
    '|『([^』\n]*)』'
)


def _detect_lang(text):
    """按文字系统粗识别语种，供 <d>[Lang] 使用。"""
    if re.search(r"[぀-ヿ]", text):       # 平假名/片假名
        return "Japanese"
    if re.search(r"[가-힣]", text):       # 谚文音节
        return "Korean"
    if re.search(r"[一-鿿]", text):       # CJK 汉字
        return "Chinese"
    if re.search(r"[Ѐ-ӿ]", text):       # 西里尔字母
        return "Russian"
    if re.search(r"[฀-๿]", text):       # 泰文
        return "Thai"
    if re.search(r"[؀-ۿ]", text):       # 阿拉伯文
        return "Arabic"
    return "English"


def _convert_quoted_dialogue(text):
    """把成对引号内容自动转成 <d>[语种] 内容</d>，语种按内容识别。"""
    if not text:
        return text

    def repl(match):
        content = next((g for g in match.groups() if g), "").strip()
        if not content:
            return match.group(0)
        return f"<d>[{_detect_lang(content)}] {content}</d>"

    return _DIALOGUE_QUOTE_RE.sub(repl, text)


# 参考图用途标注（官方手册：参考图必须标注用途，否则保主体不保背景）
_REF_PURPOSE_LINES = {
    "character": lambda k: f"<Picture {k}> 是人物参考（锁定脸和服装）",
    "scene": lambda k: f"<Picture {k}> 是场景参考（背景完全一致）",
    "style": lambda k: f"<Picture {k}> 是风格参考（匹配这种美术风格）",
    "composition": lambda k: f"<Picture {k}> 是构图参考（匹配这个取景）",
    "object": lambda k: f"<Picture {k}> 是物体参考（保持这件物品原样）",
    "first_frame": lambda k: f"<Picture {k}> 是首帧参考",
    "last_frame": lambda k: f"<Picture {k}> 是尾帧参考",
    "motion": lambda k: f"<Picture {k}> 是动作参考（沿用它的动作）",
}


def _build_purpose_lines(ref_purposes, image_slots):
    """按 ref_purposes JSON（{图片槽位: 用途key}）生成用途标注行。"""
    try:
        data = json.loads(ref_purposes or "{}")
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    lines = []
    for slot, key in data.items():
        try:
            slot = int(slot)
        except (TypeError, ValueError):
            continue
        if slot not in image_slots:
            continue
        tpl = _REF_PURPOSE_LINES.get(str(key))
        if tpl:
            lines.append(tpl(image_slots.index(slot) + 1))
    return lines


# ── 3D 白模导演台：布局 JSON → 空间关系文字（机位段 + 站位段） ──
# 中文描述给 H3，帮助它理解"谁在谁前/左右、朝向哪、机位多近多高多广"。
# 前端 composeLayoutProse 是本函数的 JS 镜像，两端算法必须逐字一致。

import math as _layout_math


def _wrap_deg(a):
    a = float(a or 0) % 360.0
    return a


_DIR_SECTORS = [
    (30, "正对目标"), (60, "目标右前方"), (120, "目标右侧"), (150, "目标右后方"),
    (210, "目标后方"), (240, "目标左后方"), (300, "目标左侧"), (330, "目标左前方"),
    (360, "正对目标"),
]


def _dir_word(az):
    az = _wrap_deg(az)
    for limit, word in _DIR_SECTORS:
        if az < limit:
            return word
    return "正对目标"


def _shot_size(dist, fov):
    fov = float(fov or 40)
    dist = float(dist or 2)
    # 水平可见高度近似 = 2*dist*tan(fov/2)，据此推断景别
    h = 2 * dist * _layout_math.tan(_layout_math.radians(fov / 2))
    if h <= 1.2:
        return "特写"
    if h <= 2.0:
        return "近景"
    if h <= 3.5:
        return "中景"
    if h <= 6.0:
        return "中全景"
    return "全景"


def _elev_word(el):
    el = float(el or 0)
    if el >= 12:
        return "高位俯拍"
    if el <= -12:
        return "低位仰拍"
    return "平视"


def _vec3(v):
    v = v or []
    if len(v) >= 3:
        try:
            return [float(v[0]), float(v[1]), float(v[2])]
        except (TypeError, ValueError):
            return None
    return None


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    n = _layout_math.sqrt(_dot(a, a))
    return a if n == 0 else [x / n for x in a]


def _project_side(entry_pos, camera_pos, forward):
    """返回 (深度档位, 左右) —— 把一点投影到摄像机画面：>0 偏右，<0 偏左。"""
    if camera_pos is None or entry_pos is None:
        return (None, None)
    forward = _norm(list(forward or [0, 0, -1]))
    up = [0, 1, 0]
    # 投影到水平面避免 y 轴倾斜干扰"左右"
    right = _cross(forward, up)
    if _length(right) < 1e-6:
        right = [1, 0, 0]
    right = _norm(right)
    forward_h = _norm([forward[0], 0, forward[2]] or forward)
    dv = _sub(entry_pos, camera_pos)
    depth = _dot(dv, forward_h)
    lateral = _dot(dv, right)
    if depth <= 0:
        return ("后方", None)
    lateral = lateral / max(abs(depth), 1e-6)
    if abs(lateral) < 0.35:
        side = "中间偏"
        side += "右" if lateral > 0 else "左"
    else:
        side = "右侧" if lateral > 0 else "左侧"
    return (depth, side)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _length(a):
    return _layout_math.sqrt(_dot(a, a))


def _compose_camera_prose(cam):
    """单机位 → 机位段文字（与前端 composeCameraProseJs 逐字一致）。"""
    if not isinstance(cam, dict):
        return ""
    az = cam.get("azimuth")
    if az is None:
        return ""
    target_name = str(cam.get("target") or "").strip() or "画面中心"
    dist = float(cam.get("distance") or 2)
    parts = [f"摄像机位于{target_name}" + _dir_word(az),
             _elev_word(cam.get("elevation")),
             f"距离≈{round(dist, 1)}m",
             f"视场≈{round(float(cam.get('fov') or 40))}°（{_shot_size(dist, cam.get('fov'))}）"]
    if cam.get("free") and cam.get("view_azimuth") is not None:
        parts.append(f"镜头朝向{_dir_word(cam.get('view_azimuth'))}，俯仰{round(float(cam.get('view_elevation') or 0))}°")
    else:
        parts.append(f"望向{target_name}")
    return "〔机位〕" + "，".join(parts) + "。"


def _layout_cameras(layout):
    """从 layout JSON 读机位数组（兼容旧单 camera → [camera]）。"""
    try:
        data = json.loads(layout or "{}")
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("cameras"), list) and data["cameras"]:
        return data["cameras"]
    legacy = data.get("camera")
    return [legacy] if isinstance(legacy, dict) and legacy else []


def _compose_layout_prose(layout):
    """布局 JSON → 空间关系文字；无有效内容返回空串。

    布局结构（与前端 JS 一致）：
      {"file","camera":{"pos":[x,y,z],"target_pos":[x,y,z],"azimuth","elevation",
                         "distance","fov","target"},
       "actors":[{"name","color","pos","rotY","pose","scale"}],
       "props":[{"name","color","type","pos","rotY","size"}],"activeCamera"}
    """
    try:
        data = json.loads(layout or "{}")
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    cams = None
    if isinstance(data.get("cameras"), list) and data.get("cameras"):
        cams = data["cameras"]
    else:
        legacy = data.get("camera")
        cams = [legacy] if isinstance(legacy, dict) and legacy else []
    active_selected = str(data.get("activeCamera") or "").strip()
    cam = next((c for c in cams if isinstance(c, dict) and str(c.get("name", "")) == active_selected), cams[0] if cams else {})
    if not isinstance(cam, dict):
        cam = {}
    actors = [a for a in (data.get("actors") or []) if isinstance(a, dict) and a.get("name")]
    props = [p for p in (data.get("props") or []) if isinstance(p, dict) and p.get("name")]
    if not cam and not actors and not props:
        return ""

    target_name = str(cam.get("target") or "").strip() or "画面中心"
    cam_pos = _vec3(cam.get("pos"))
    target_pos = _vec3(cam.get("target_pos"))
    forward = None
    if cam_pos is not None and target_pos is not None:
        forward = _sub(target_pos, cam_pos)

    segs = []
    # ── 机位段 ──
    cam_line = _compose_camera_prose(cam)
    if cam_line:
        segs.append(cam_line)

    # ── 站位段 ──
    stage_parts = []
    for a in actors:
        name = str(a.get("name"))
        pose = str(a.get("pose") or "").strip()
        depth, side = _project_side(_vec3(a.get("pos")), cam_pos, forward)
        loc = ""
        if side:
            if depth == "后方":
                loc = "在画面" + side + "后方"
            else:
                plane = "前景" if depth is not None and depth < 1.0 else (
                    "中景" if depth is not None and depth < 2.5 else "背景")
                loc = f"{side}画面{plane}"
        desc = name
        if loc:
            desc += f"（{loc}）"
        if pose and pose != "站立":
            desc += f" 呈{pose}姿势"
        stage_parts.append(desc)
    for p in props:
        name = str(p.get("name"))
        depth, side = _project_side(_vec3(p.get("pos")), cam_pos, forward)
        loc = ""
        if side:
            plane = "前景" if depth is not None and depth < 1.0 else (
                "中景" if depth is not None and depth < 2.5 else "背景")
            loc = f"{side}画面{plane}"
        stage_parts.append(f"{name}（{loc}）" if loc else name)
    if stage_parts:
        segs.append("〔站位〕" + "；".join(stage_parts) + "。")

    # 空间：有墙体 → 提示室内围合
    if any(isinstance(w, dict) and w.get("name") for w in (data.get("walls") or [])):
        segs.append("〔空间〕四周由墙体围合，构成室内空间。")

    # 视线轴：前两个可见人物
    kept = [a.get("name") for a in actors[:2] if a.get("name")]
    if len(kept) >= 2:
        segs.append(f"〔视线轴〕{kept[0]}→{kept[1]}。")

    if not segs:
        return ""
    return "空间关系（白模预视觉化）：" + " ".join(segs)


def _referenced_indexes(text):
    """扫描文本里用到的 @L 序号（转成 0 基索引），排序去重。"""
    return sorted({int(m) - 1 for m in _LIB_RE.findall(text or "")})


def _referenced_from_director(shots_json):
    """扫描导演台 JSON（ref_decl + 各镜头提示词）里用到的 @L 索引。"""
    idxs = set()
    try:
        data = json.loads(shots_json or "{}")
    except Exception:
        return []
    texts = []
    if isinstance(data, dict):
        texts.append(str(data.get("ref_decl", "")))
        for s in data.get("shots") or []:
            if isinstance(s, dict):
                texts.append(str(s.get("prompt", "")))
    elif isinstance(data, list):
        for s in data:
            if isinstance(s, dict):
                texts.append(str(s.get("prompt", "")))
    for t in texts:
        idxs.update(int(m) - 1 for m in _LIB_RE.findall(t))
    return sorted(idxs)


def _library_plan(entries, referenced, connected_img_slots, paired_audio, connected_audio_slots):
    """计算被引用素材库条目的 <Picture K>/<Audio K> 编号与注入顺序。

    Picture = 已连接图片（槽位升序）→ 被引用素材库图片（库顺序）
    Audio   = 视频音轨（槽位升序）→ 已连接独立音频（槽位升序）→ 被引用素材库 audio（库顺序）→ 被引用 character 语音（库顺序）
    返回 (pic_of, aud_of, img_order, aud_order)。
    """
    referenced = sorted(set(referenced))
    pic_of = {}
    aud_of = {}
    img_order = []
    aud_order = []
    n_img = len(connected_img_slots)
    n_aud = len(connected_audio_slots)
    for i in referenced:
        e = entries[i] if 0 <= i < len(entries) else None
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind", "")).lower()
        if kind in ("character", "prop", "scene") and e.get("file"):
            n_img += 1
            pic_of[i] = n_img
            img_order.append((i, kind))
    lib_audio = []
    char_voice = []
    for i in referenced:
        e = entries[i] if 0 <= i < len(entries) else None
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind", "")).lower()
        if kind == "audio" and e.get("file"):
            lib_audio.append(i)
        elif kind == "character" and e.get("audio_file"):
            char_voice.append(i)
    for rank, i in enumerate(lib_audio):
        aud_of[i] = paired_audio + n_aud + rank + 1
        aud_order.append((i, "audio"))
    for rank, i in enumerate(char_voice):
        aud_of[i] = paired_audio + n_aud + len(lib_audio) + rank + 1
        aud_order.append((i, "voice"))
    return pic_of, aud_of, img_order, aud_order


def _expand_library_tags(text, entries, pic_of, aud_of):
    """把 @L{n} 展开为 <Picture K>/<Audio K>，并收集标注行（去重、保持顺序）。

    character 标注含外貌与语音：...人物参考（锁定脸和服装）。外貌：…，音色参考 <Audio J>
    """
    annos = []
    seen = set()

    def add(line):
        if line and line not in seen:
            seen.add(line)
            annos.append(line)

    def repl(match):
        i = int(match.group(1)) - 1
        entry = entries[i] if 0 <= i < len(entries) else None
        if not isinstance(entry, dict):
            return match.group(0)
        if i in pic_of:
            tag = f"<Picture {pic_of[i]}>"
            kind = str(entry.get("kind", "")).lower()
            name = str(entry.get("name", "")).strip()
            if kind == "character":
                line = f"{tag} 是{name}的人物参考（锁定脸和服装）" if name else f"{tag} 是人物参考（锁定脸和服装）"
                appearance = str(entry.get("appearance", "")).strip()
                if appearance:
                    line += f"。外貌：{appearance}"
                if i in aud_of:
                    line += f"，音色参考 <Audio {aud_of[i]}>"
                add(line)
            elif kind == "prop":
                add(f"{tag} 是{name}的物体参考（保持原样）" if name else f"{tag} 是物体参考（保持这件物品原样）")
            else:  # scene / 其他图片类
                add(f"{tag} 是{name}的场景参考（背景完全一致）" if name else f"{tag} 是场景参考（背景完全一致）")
            return tag
        if i in aud_of:
            tag = f"<Audio {aud_of[i]}>"
            add(f"{tag} 原样复用这段音频")
            return tag
        return match.group(0)  # 未引用/无媒体 → 原样保留

    return _LIB_RE.sub(repl, text), annos


def _load_global_library():
    """从磁盘读永久素材库清单（input/zoey_library/library.json，与 server 一致）。"""
    try:
        import folder_paths
        path = os.path.join(folder_paths.get_input_directory(), _LIB_SUBFOLDER, "library.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _load_image(rel_path):
    """按 input 目录相对路径加载为 IMAGE 张量。"""
    import nodes
    img, _ = nodes.LoadImage().load_image(rel_path)
    return img


def _load_audio(rel_path):
    """按 input 目录相对路径加载为 {"waveform":[B,C,L],"sample_rate":sr}。

    get_full_path("input", ...) 在部分环境下搜不到子目录文件，直接按上传路径拼接。
    """
    import folder_paths
    import torchaudio
    abs_path = os.path.join(folder_paths.get_input_directory(), rel_path)
    if not os.path.isfile(abs_path):
        raise ValueError(f"素材库音频文件不存在：{rel_path}（{abs_path}）")
    waveform, sr = torchaudio.load(abs_path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}


def _parse_ref_tray(ref_tray):
    """解析前端参考素材托盘 JSON → [{type(file/image/audio), file, kind, name}, ...]。"""
    try:
        data = json.loads(ref_tray or "[]")
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        typ = str(it.get("type", "")).lower()
        f = str(it.get("file", "") or "").strip()
        if typ not in ("image", "video", "audio") or not f:
            continue
        out.append({"type": typ, "file": f,
                    "kind": str(it.get("kind", "") or "").strip(),
                    "name": str(it.get("name", "") or "").strip()})
    return out


def _load_tray_video(rel_path, max_frames=128):
    """视频文件（input 相对路径）→ IMAGE 帧张量 [N,H,W,C]，PyAV 解码、均匀抽样降密度。"""
    import folder_paths
    import numpy as np
    import torch
    import av
    abs_path = os.path.join(folder_paths.get_input_directory(), rel_path)
    if not os.path.isfile(abs_path):
        raise ValueError(f"托盘视频不存在：{rel_path}（{abs_path}）")
    container = av.open(abs_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"托盘视频解码不到帧：{rel_path}")
    n = len(frames)
    arr = np.stack(frames).astype(np.float32) / 255.0   # N,H,W,3
    return torch.from_numpy(arr)


def _inject_tray(ref_images, ref_videos, ref_audios, tray):
    """把托盘素材加载进 ref 字典（内部读取，无需外接端口）；返回 (img, vid, aud) 数量。"""
    img = vid = aud = 0
    for e in tray:
        t, f = e["type"], e["file"]
        if t == "image":
            ref_images[f"tray_img_{img}"] = _load_image(f)
            img += 1
        elif t == "video":
            ref_videos[f"tray_vid_{vid}"] = _load_tray_video(f)
            vid += 1
        elif t == "audio":
            ref_audios[f"tray_aud_{aud}"] = _load_audio(f)
            aud += 1
    return img, vid, aud


def _inject_library(ref_images, ref_audios, img_order, aud_order, entries):
    """把被引用的素材库媒体 load 进 ref_images/ref_audios（保持编号顺序）。"""
    for idx, kind in img_order:
        e = entries[idx]
        rel = f"{_LIB_SUBFOLDER}/{e['file']}"
        try:
            ref_images[f"lib_img_{idx}"] = _load_image(rel)
        except Exception as err:
            raise ValueError(f"素材库图片加载失败：{e.get('file')}（{err}）") from err
    for idx, kind in aud_order:
        e = entries[idx]
        rel = f"{_LIB_SUBFOLDER}/{e['audio_file'] if kind == 'voice' else e['file']}"
        try:
            ref_audios[f"lib_aud_{idx}"] = _load_audio(rel)
        except Exception as err:
            raise ValueError(f"素材库音频加载失败：{e.get('audio_file') if kind == 'voice' else e.get('file')}（{err}）") from err


def _expand_char_tags(text, char_pic):
    """把 @C1/@c1/@char1... 展开为 <Picture N>（char_pic: {角色序号: Picture 编号}）。"""
    def repl(match):
        pic = char_pic.get(int(match.group(1)))
        if pic is None:
            return match.group(0)  # 未分配角色，原样保留
        return f"<Picture {pic}>"
    return _CAST_RE.sub(repl, text)


def _compose_director(shots_json, counts, image_slots, audio_slots, paired_audio, library, pic_of, aud_of,
                      dialogue_convert=True, layout="{}"):
    """导演台：分镜拼成 CUT/TRANSITION + 对白 + 角色人物参考 + 音效/配乐 + 总时长。

    数据结构：
      {"ref_decl", "characters":[{"slot","name"}], "speakers":[{"id","desc"}],
       "soundscape", "music", "shots":[{"prompt","duration","transition","dialogue":[{"speaker","lang","text"}]}]}
    兼容旧格式：直接传镜头数组 / 无新增字段。
    返回 (提示词文本, 总时长(秒), 是否已写用户参考说明)。
    """
    try:
        data = json.loads(shots_json or "{}")
    except Exception:
        data = {}
    ref_decl = ""
    characters = []
    speakers = {}
    soundscape = ""
    music = ""
    consistent = True
    if isinstance(data, dict):
        shots = data.get("shots") or []
        ref_decl = str(data.get("ref_decl", "")).strip()
        characters = data.get("characters") or []
        for sp in (data.get("speakers") or []):
            if isinstance(sp, dict) and str(sp.get("id", "")).strip():
                speakers[str(sp.get("id")).strip()] = str(sp.get("desc", "")).strip()
        soundscape = str(data.get("soundscape", "")).strip()
        music = str(data.get("music", "")).strip()
        consistent = bool(data.get("consistent", True))
    elif isinstance(data, list):
        shots = data
    else:
        shots = []
    if not isinstance(shots, list) or not shots:
        raise ValueError("导演台：请至少添加一个镜头（点击「＋ 添加镜头」）")

    # 角色槽 -> <Picture K>（图片按已连接槽位顺序编号，与前端 collectEntries 一致）
    image_slots = sorted(image_slots or [])
    char_pic = {}
    for i, ch in enumerate(characters, start=1):
        if not isinstance(ch, dict):
            continue
        try:
            slot = int(ch.get("slot"))
        except (TypeError, ValueError):
            continue
        if slot in image_slots:
            char_pic[i] = image_slots.index(slot) + 1

    # 声明块：用户参考说明（可空）+ 角色人物参考行 + 素材库用途标注
    decl_parts = []
    lib_annos = []
    if ref_decl:
        ref_decl, a = _expand_library_tags(ref_decl, library, pic_of, aud_of)
        lib_annos.extend(a)
        decl_parts.append(expand_at_tags(ref_decl, counts))
    for i, ch in enumerate(characters, start=1):
        pic = char_pic.get(i)
        if pic is None:
            continue
        name = str(ch.get("name", "")).strip() if isinstance(ch, dict) else ""
        if name:
            decl_parts.append(f"<Picture {pic}> 是{name}的人物参考（锁定脸和服装）")
        else:
            decl_parts.append(f"<Picture {pic}> 是人物参考（锁定脸和服装）")

    parts = []
    total = 0.0
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict) or not str(shot.get("prompt", "")).strip():
            raise ValueError(f"导演台：CUT {i + 1} 缺少提示词")
        shot_text = str(shot.get("prompt", "")).strip()
        if dialogue_convert:
            shot_text = _convert_quoted_dialogue(shot_text)
        shot_text, a = _expand_library_tags(_expand_char_tags(shot_text, char_pic),
                                            library, pic_of, aud_of)
        lib_annos.extend(a)
        shot_text = expand_at_tags(shot_text, counts)
        # 该镜机位（来自布局 cameras，按 shot.camera 匹配）
        _cam = next((c for c in _layout_cameras(layout)
                     if isinstance(c, dict) and str(c.get("name", "")) == str(shot.get("camera", ""))), None)
        _cam_line = _compose_camera_prose(_cam)
        if _cam_line:
            shot_text = _cam_line + "\n" + shot_text
        # 跨镜一致性约束：从第二镜起补一句保持角色/场景/服装一致 + 逐镜一致性清单
        if i > 0 and consistent:
            shot_text += "\n保持与上一镜相同的角色、场景、服装与光线。"
            if str(shot.get("continuity") or "").strip():
                shot_text += "\n保持与上一镜一致：" + str(shot.get("continuity")).strip() + "。"
        # 对白：身份描述（可空）+ (Sx) + <d>[语言] 原文</d>
        for d in shot.get("dialogue") or []:
            if not isinstance(d, dict):
                continue
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            spk = str(d.get("speaker", "")).strip() or "S1"
            lang = str(d.get("lang", "")).strip() or "English"
            desc = speakers.get(spk, "")
            if desc:
                shot_text += f"\n{desc} ({spk}) says: <d>[{lang}] {text}.</d>"
            else:
                shot_text += f"\n({spk}) says: <d>[{lang}] {text}.</d>"
        if i == 0:
            parts.append(f"CUT 1: {shot_text}")
        else:
            transition = str(shot.get("transition", "")).strip()
            if transition:
                parts.append(f"TRANSITION: {transition}")
            parts.append(f"CUT {i + 1}: {shot_text}")
        try:
            total += float(shot.get("duration", 5.0))
        except Exception:
            total += 5.0

    if soundscape:
        parts.append(f"overall_soundscape: {soundscape}")
    if music:
        parts.append(f"non_diegetic_music: {music}")

    if lib_annos:
        # 跨镜头/ref_decl 出现的 @L 标注合并去重
        decl_parts.extend(list(dict.fromkeys(lib_annos)))
    if decl_parts:
        parts.insert(0, "\n".join(decl_parts))
    return "\n".join(parts), min(total, 15.0), bool(ref_decl)


class ZoeyMiniMaxH3ReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import comfy.samplers
        loras = [_NONE] + folder_paths.get_filename_list("loras")
        optional = {}
        for i in range(_IMAGE_SLOTS):
            optional[f"ref_image_{i}"] = ("IMAGE",)
        optional["negative"] = ("CONDITIONING", {"tooltip": "长视频分段采样用的负向条件；接一个空/负面 ClipTextEncode"})
        for i in range(_VIDEO_SLOTS):
            optional[f"ref_video_{i}"] = ("IMAGE",)
            optional[f"ref_video_audio_{i}"] = ("AUDIO",)
        for i in range(_AUDIO_SLOTS):
            optional[f"ref_audio_{i}"] = ("AUDIO",)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "placeholder": "参考模式用 @P1/@V1/@A1 引用素材；T2V 纯文本；I2V 用首张参考图作首帧；自动按连接选择",
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
                "library": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "永久全局素材库（前端渲染，值不存工作流）",
                }),
                "mode": (["参考", "T2V", "I2V", "自动"], {"default": "参考"}),
                "sage_attention": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用 SageAttention",
                    "label_off": "关闭",
                    "tooltip": "对该模型启用 SageAttention 加速（仅影响此模型克隆；需已安装 sageattention）。",
                }),
                "dialogue_convert": ("BOOLEAN", {
                    "default": True,
                    "label_on": "转换",
                    "label_off": "关闭",
                }),
                "ref_purposes": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "参考图用途标注 JSON（前端渲染，如 {\"0\":\"scene\"}）",
                }),
                # ── 内置加载器（追加在末尾：不动上方已有控件的索引，避免旧工作流 widgets_values 错位） ──
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "MiniMax H3 的 DiT 模型（diffusion_models 文件夹）。"}),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "MiniMax H3 的 Qwen3-VL CLIP（text_encoders 文件夹，type=minimax）。"}),
                "vae_name": (_vae_names(), {"tooltip": "MiniMax H3 视频 VAE。"}),
                "audio_vae_name": ([_NONE] + _vae_names(), {"default": _NONE, "tooltip": "MiniMax H3 音频 VAE（引用音频素材时才需要）。"}),
                "lora_list": (loras, {"default": _NONE, "tooltip": "LoRA 下拉选项源（前端隐藏，仅提供文件名列表）。"}),
                "loras": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "placeholder": '[{"lora":"名字.safetensors","model":1.0,"clip":1.0}]',
                    "tooltip": "LoRA 列表（前端渲染；每个条目含 lora 文件名 + model/clip 强度）。",
                }),
                "layout": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "placeholder": "3D 白模导演台布局数据（前端渲染；含机位/人物/物体 + 导出的预视觉化截图）",
                    "tooltip": "3D 白模导演台：摆人物姿势/物体/机位 → 导出布局截图作构图参考 + 生成空间关系文字。",
                }),
                "layout_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用",
                    "label_off": "关闭",
                    "tooltip": "是否让 3D 白模布局参与生成：关闭=不注入布局构图参考图与空间关系文字，逐镜机位也失效。",
                }),
                # 追加在末尾：不动上方已有控件的索引，避免旧工作流 widgets_values 错位
                "tray": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "placeholder": '[{"type":"image","file":"a.png","kind":"character","name":"主图"}]',
                    "tooltip": "参考素材托盘：节点内部按 input 相对路径读取图片/音频/视频，替代外接端口（前端渲染，@P1/@V1/@A1 引用托盘顺序）。",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                # ── 最后N步注入LoRA（两段采样，追加在末尾） ──
                "last_lora_name": (loras, {"default": _NONE, "tooltip": "只在最后 N 步注入的 LoRA（如皮肤质感）；无=不启用两段"}),
                "last_lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "last_lora_clip": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "该 LoRA 的 CLIP 强度；纹理/皮肤 LoRA 一般 0"}),
                "last_steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "最后多少步改用合成该LoRA的模型采样；0=不启用"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "VAE", "VAE", "CLIP", "CONDITIONING", "LATENT", "STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("model", "vae", "audio_vae", "clip", "positive", "LATENT", "提示词(已转换)", "视频帧", "音频", "设置")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"
    OUTPUT_NODE = True  # 输出节点：即使单独跑（靠"保存mp4"出片）也会被 ComfyUI 执行，否则报 Prompt has no outputs

    def process(self, unet_name, clip_name, vae_name, audio_vae_name,
                lora_list, loras,
                prompt, mode, resolution, aspect, duration,
                ref_image_size, auto_declaration, director_mode, director_shots,
                library="", tray="[]", sage_attention=False,
                dialogue_convert=True, ref_purposes="{}", layout="{}", layout_enabled=True,
                negative=None, seed=0, steps=25,
                cfg=4.0, sampler_name="res_multistep", scheduler="simple",
                last_lora_name=_NONE, last_lora_strength=1.0, last_lora_clip=0.0, last_steps=0, **refs):
        # ── 内置模型加载：UNet + CLIP + VAE + 音频 VAE，再叠 LoRA ──
        model = _load_unet(unet_name)
        clip = _load_clip(clip_name)
        vae = _load_vae(vae_name)
        audio_vae = _load_vae(audio_vae_name) if audio_vae_name and audio_vae_name != _NONE else None
        parsed_loras = _parse_loras(loras)
        for lora_name, lora_model, lora_clip in parsed_loras:
            model, clip = _apply_lora(model, clip, lora_name, lora_model, lora_clip)

        return self._segment(
            model, clip, vae, audio_vae,
            prompt, mode, resolution, aspect, duration, ref_image_size,
            auto_declaration, director_mode, director_shots, library, tray,
            sage_attention, dialogue_convert, ref_purposes, layout, layout_enabled,
            negative, seed, steps, cfg, sampler_name, scheduler,
            last_lora_name, last_lora_strength, last_lora_clip, last_steps, **refs)

    def _segment(self, model, clip, vae, audio_vae, prompt, mode, resolution, aspect, duration,
                 ref_image_size, auto_declaration, director_mode, director_shots, library="",
                 tray="[]", sage_attention=False, dialogue_convert=True, ref_purposes="{}",
                 layout="{}", layout_enabled=True, negative=None, seed=0, steps=25,
                 cfg=4.0, sampler_name="res_multistep", scheduler="simple",
                 last_lora_name=_NONE, last_lora_strength=1.0, last_lora_clip=0.0, last_steps=0,
                 composed_prompt=None, width_override=None, height_override=None, **refs):
        # 用已加载的 model/clip/vae 做单段生成（不重复加载）：process 与长视频节点都调这里
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

        # 参考素材托盘：节点内部按路径读取图片/视频/音频（替代/叠加外接端口）
        tray_entries = _parse_ref_tray(tray)
        tray_img, tray_vid, tray_aud = _inject_tray(ref_images, ref_videos, ref_audios, tray_entries)

        port_img = sum(1 for i in range(_IMAGE_SLOTS) if refs.get(f"ref_image_{i}") is not None)
        port_vid = sum(1 for i in range(_VIDEO_SLOTS) if refs.get(f"ref_video_{i}") is not None)
        port_vid_aud = sum(1 for i in range(_VIDEO_SLOTS) if refs.get(f"ref_video_audio_{i}") is not None)
        port_aud = sum(1 for i in range(_AUDIO_SLOTS) if refs.get(f"ref_audio_{i}") is not None)

        counts = count_refs(
            image_count=port_img + tray_img,
            video_count=port_vid + tray_vid,
            video_audio_count=port_vid_aud,
            audio_count=port_aud + tray_aud,
        )

        # 长视频末帧续接：把上一段末帧注入为参考图（不计 @P，纯衔接）
        prev_frame = refs.get("prev_frame")
        prev_line = ""
        if prev_frame is not None:
            ref_images["prev_frame"] = prev_frame
            prev_line = "保持与上一镜相同的角色、场景、服装与光线（以上一镜尾帧为续接参考）。"
        image_slots = [i for i in range(_IMAGE_SLOTS) if refs.get(f"ref_image_{i}") is not None]
        image_slots += list(range(_IMAGE_SLOTS, _IMAGE_SLOTS + tray_img))
        has_video = any(v is not None for v in ref_videos.values())
        has_audio = any(a is not None for a in ref_audios.values()) \
            or any(a is not None for a in ref_video_audios.values())

        # 托盘图片的用途（人物/场景/风格…）并入 ref_purposes，供 _build_purpose_lines 生成标注行
        if any((e["type"] == "image" and e.get("kind")) for e in tray_entries):
            try:
                merged_purposes = json.loads(ref_purposes or "{}")
            except Exception:
                merged_purposes = {}
            if not isinstance(merged_purposes, dict):
                merged_purposes = {}
            ti = 0
            for e in tray_entries:
                if e["type"] == "image":
                    if e.get("kind"):
                        merged_purposes[str(_IMAGE_SLOTS + ti)] = e["kind"]
                    ti += 1
            ref_purposes = json.dumps(merged_purposes, ensure_ascii=False)

        # 引号台词自动转 <d>[语种] 内容</d>（覆盖非导演台的原始 prompt）
        if dialogue_convert:
            prompt = _convert_quoted_dialogue(prompt or "")

        # 自动模式：按连接情况选后端
        eff = mode
        if mode == "自动":
            if has_video or has_audio:
                eff = "参考"
            elif len(image_slots) == 1:
                eff = "I2V"
            elif image_slots:
                eff = "参考"
            else:
                eff = "T2V"

        _cw, _ch = _compute_canvas(ref_images, resolution, aspect)
        width = width_override or _cw
        height = height_override or _ch

        if eff in ("T2V", "I2V"):
            # I2V：第一张已连接参考图作首帧；≥2 张时最后一张作尾帧（首尾帧）
            def _slot_img(slot):
                return ref_images.get(f"ref_image_{slot}") if slot < _IMAGE_SLOTS else ref_images.get(f"tray_img_{slot - _IMAGE_SLOTS}")
            first_frame = _slot_img(image_slots[0]) if (eff == "I2V" and image_slots) else None
            last_frame = _slot_img(image_slots[-1]) if (eff == "I2V" and len(image_slots) >= 2) else None
            cond, latent, prompt_text = self._run_text_image(
                clip, vae, prompt, width, height, duration,
                director_mode, director_shots, eff == "I2V",
                first_frame, dialogue_convert, last_frame,
                (layout if layout_enabled else ""), layout_enabled, composed_prompt)
        else:
            cond, latent, prompt_text = self._run_ref2va(
                clip, vae, audio_vae, prompt, ref_images, ref_videos,
                ref_video_audios, ref_audios, counts, width, height,
                duration, ref_image_size, auto_declaration, director_mode,
                director_shots, library, image_slots, dialogue_convert,
                ref_purposes, (layout if layout_enabled else ""), layout_enabled,
                composed_prompt)

        if prev_line and prompt_text:
            prompt_text = prompt_text + "\n" + prev_line

        settings = json.dumps({
            "steps": int(steps), "cfg": float(cfg),
            "sampler_name": sampler_name, "scheduler": scheduler,
            "seed": int(seed),
        }, ensure_ascii=False)

        # 集成 SageAttention：在模型克隆上启用，随 CONDITIONING/LATENT 一并返回
        if sage_attention:
            model = _apply_sage_attention(model)

        # 最后N步注入LoRA（两段采样）：设置后节点自己采样，走 base→合成LoRA 续跑，输出画面
        if last_steps > 0 and last_lora_name != _NONE:
            import nodes as _nodes
            if negative is None:
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            out_latent, two_info = ZoeyMiniMaxH3LastStepsLora().process(
                model, cond, negative, latent, last_lora_name, last_lora_strength, last_lora_clip,
                seed, steps, last_steps, cfg, sampler_name, scheduler)
            # 完整 AV 嵌套 latent 直接给 VAEDecode/VAEDecodeAudio，由它们按 VAE 拆流解码（不再拆 tensors[0]）
            frames = _nodes.VAEDecode().decode(vae, {"samples": out_latent["samples"]})[0]
            audio = None
            if audio_vae is not None:
                from comfy_extras.nodes_audio import VAEDecodeAudio
                audio = VAEDecodeAudio().decode(audio_vae, {"samples": out_latent["samples"]})[0]
            return (model, vae, audio_vae, clip, cond, out_latent,
                    (prompt_text or "") + " ｜ " + two_info, frames, audio, settings)

        # 未启用两段(low last_steps/无 last_lora)时也做单段采样，让输出节点总能出片
        import comfy.sample as _cs
        import nodes as _n3
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        latent_image = latent["samples"]
        latent_image = _cs.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = _cs.prepare_noise(latent_image, seed, batch_inds)
        import latent_preview
        samples = _cs.sample(model, noise, steps, cfg, sampler_name, scheduler, cond, negative, latent_image,
                             denoise=1.0, force_full_denoise=True, noise_mask=latent.get("noise_mask"),
                             callback=latent_preview.prepare_callback(model, steps), seed=seed)
        frames = _n3.VAEDecode().decode(vae, {"samples": samples})[0]
        audio = None
        if audio_vae is not None:
            from comfy_extras.nodes_audio import VAEDecodeAudio
            audio = VAEDecodeAudio().decode(audio_vae, {"samples": samples})[0]
        return (model, vae, audio_vae, clip, cond, latent, prompt_text, frames, audio, settings)

    def _compose_plain_director(self, shots_json, dialogue_convert=True, layout="{}"):
        """T2V/I2V 分镜组合：只拼 CUT/TRANSITION/对白/音效/配乐，不做 @ 引用展开。"""
        try:
            data = json.loads(shots_json or "{}")
        except Exception:
            data = {}
        shots = []
        speakers = {}
        soundscape = ""
        music = ""
        consistent = True
        if isinstance(data, dict):
            shots = data.get("shots") or []
            soundscape = str(data.get("soundscape", "")).strip()
            music = str(data.get("music", "")).strip()
            consistent = bool(data.get("consistent", True))
            for sp in data.get("speakers") or []:
                if isinstance(sp, dict) and str(sp.get("id", "")).strip():
                    speakers[str(sp.get("id")).strip()] = str(sp.get("desc", "")).strip()
        elif isinstance(data, list):
            shots = data
        if not isinstance(shots, list) or not shots:
            raise ValueError("导演台：请至少添加一个镜头（点击「＋ 添加镜头」）")

        parts = []
        total = 0.0
        for i, shot in enumerate(shots):
            if not isinstance(shot, dict) or not str(shot.get("prompt", "")).strip():
                raise ValueError(f"导演台：CUT {i + 1} 缺少提示词")
            shot_text = str(shot.get("prompt", "")).strip()
            if dialogue_convert:
                shot_text = _convert_quoted_dialogue(shot_text)
            # 该镜机位（来自布局 cameras，按 shot.camera 匹配）
            _cam = next((c for c in _layout_cameras(layout)
                         if isinstance(c, dict) and str(c.get("name", "")) == str(shot.get("camera", ""))), None)
            _cam_line = _compose_camera_prose(_cam)
            if _cam_line:
                shot_text = _cam_line + "\n" + shot_text
            if i > 0 and consistent:
                shot_text += "\n保持与上一镜相同的角色、场景、服装与光线。"
                if str(shot.get("continuity") or "").strip():
                    shot_text += "\n保持与上一镜一致：" + str(shot.get("continuity")).strip() + "。"
            for d in shot.get("dialogue") or []:
                if not isinstance(d, dict):
                    continue
                text = str(d.get("text", "")).strip()
                if not text:
                    continue
                spk = str(d.get("speaker", "")).strip() or "S1"
                lang = str(d.get("lang", "")).strip() or "English"
                desc = speakers.get(spk, "")
                if desc:
                    shot_text += f"\n{desc} ({spk}) says: <d>[{lang}] {text}.</d>"
                else:
                    shot_text += f"\n({spk}) says: <d>[{lang}] {text}.</d>"
            if i == 0:
                parts.append(f"CUT 1: {shot_text}")
            else:
                transition = str(shot.get("transition", "")).strip()
                if transition:
                    parts.append(f"TRANSITION: {transition}")
                parts.append(f"CUT {i + 1}: {shot_text}")
            try:
                total += float(shot.get("duration", 5.0))
            except Exception:
                total += 5.0
        if soundscape:
            parts.append(f"overall_soundscape: {soundscape}")
        if music:
            parts.append(f"non_diegetic_music: {music}")
        return "\n".join(parts), min(total, 15.0)

    def _run_text_image(self, clip, vae, prompt, width, height, duration,
                        director_mode, director_shots, use_first_frame, first_frame,
                        dialogue_convert=True, last_frame=None, layout="{}", layout_enabled=True,
                        composed_prompt=None):
        """T2V / I2V：走官方 MiniMaxH3ImageToVideo，不做 @ 引用展开。"""
        if composed_prompt is not None:
            prompt_text = composed_prompt
            length = _frame_count(duration)
        elif director_mode:
            prompt_text, total = self._compose_plain_director(director_shots, dialogue_convert, layout)
            length = _frame_count(total)
        else:
            prompt_text = (prompt or "").strip()
            length = _frame_count(duration)
        # 布局空间关系文字（纯文字，T2V/I2V 不注入预视觉化图）
        layout_prose = _compose_layout_prose(layout)
        if layout_prose:
            prompt_text = layout_prose + "\n" + prompt_text
        outputs = MiniMaxH3ImageToVideo.execute(
            clip, vae, prompt_text, width, height, length,
            first_frame=first_frame if use_first_frame else None,
            last_frame=last_frame if use_first_frame else None,
        )
        return (outputs.args[0], outputs.args[1], prompt_text)

    def _run_ref2va(self, clip, vae, audio_vae, prompt, ref_images, ref_videos,
                    ref_video_audios, ref_audios, counts, width, height, duration,
                    ref_image_size, auto_declaration, director_mode, director_shots,
                    library, image_slots, dialogue_convert=True, ref_purposes="{}",
                    layout="{}", layout_enabled=True, composed_prompt=None):
        # 长视频续接：主节点已组合好提示词，这里直接复用（跳过二次 @ 展开/用途/布局注入）
        if composed_prompt is not None:
            prompt_text = composed_prompt
            length = _frame_count(duration)
            if audio_vae is None and (any(v is not None for v in ref_video_audios.values())
                                      or any(a is not None for a in ref_audios.values())):
                raise ValueError("参考模式引用了音频素材，请选择 audio_vae_name")
            outputs = MiniMaxH3ReferenceToVideo.execute(
                clip, vae, audio_vae, prompt_text, width, height, length,
                ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios)
            return (outputs.args[0], outputs.args[1], prompt_text)
        # 永久全局素材库：从磁盘读清单，仅对提示词里 @L 引用的条目注入媒体
        library = _load_global_library()
        audio_slots = sorted(i for i in range(_AUDIO_SLOTS) if ref_audios.get(f"ref_audio_{i}") is not None)
        paired_audio = sum(
            1 for i in range(_VIDEO_SLOTS)
            if ref_videos.get(f"ref_video_{i}") is not None
            and ref_video_audios.get(f"ref_video_audio_{i}") is not None)

        pic_of = aud_of = {}
        img_order = aud_order = []
        if director_mode:
            referenced = _referenced_from_director(director_shots)
            pic_of, aud_of, img_order, aud_order = _library_plan(
                library, referenced, image_slots, paired_audio, audio_slots)
            composed, total_duration, has_user_decl = _compose_director(
                director_shots, counts, image_slots, audio_slots, paired_audio, library,
                pic_of, aud_of, dialogue_convert, layout)
            if not has_user_decl and auto_declaration:
                declaration = build_declaration(counts)
                if declaration:
                    composed = declaration + "\n" + composed
            prompt_text = composed
            length = _frame_count(total_duration)
        else:
            prompt_text = expand_at_tags(prompt, counts)
            pic_of, aud_of, img_order, aud_order = _library_plan(
                library, _referenced_indexes(prompt), image_slots, paired_audio, audio_slots)
            prompt_text, lib_annos = _expand_library_tags(prompt_text, library, pic_of, aud_of)
            decl = []
            if auto_declaration:
                declaration = build_declaration(counts)
                if declaration:
                    decl.append(declaration)
            decl.extend(lib_annos)
            if decl:
                prompt_text = "\n".join(decl) + "\n" + prompt_text
            length = _frame_count(duration)

        # 参考图用途标注（简单模式/导演台共用）：按 ref_purposes 生成 <Picture K> 是…行
        purpose_lines = _build_purpose_lines(ref_purposes, image_slots)
        if purpose_lines:
            prompt_text = "\n".join(purpose_lines) + "\n" + prompt_text

        if img_order or aud_order:
            _inject_library(ref_images, ref_audios, img_order, aud_order, library)

        # 3D 白模导演台：布局预视觉化截图作构图参考（排所有图之后）+ 空间关系文字到顶部
        layout_data = None
        try:
            _ls = (layout or "").strip()
            layout_data = json.loads(_ls) if _ls and _ls != "{}" else None
        except Exception:
            layout_data = None
        layout_prose = _compose_layout_prose(layout)
        layout_pic = len(image_slots) + len(img_order) + 1
        layout_screen = ""
        if isinstance(layout_data, dict) and layout_data.get("file"):
            try:
                ref_images[f"layout_img_0"] = _load_image(f"zoey_layout/{layout_data['file']}")
                layout_screen = f"<Picture {layout_pic}> 是站位机位参考图（人物站位与摄像机方位）"
            except Exception:
                layout_screen = ""
        layout_prepend = []
        if layout_prose:
            layout_prepend.append(layout_prose)
        if layout_screen:
            layout_prepend.append(layout_screen)
        if layout_prepend:
            prompt_text = "\n".join(layout_prepend) + "\n" + prompt_text

        if audio_vae is None and (paired_audio or any(a is not None for a in ref_audios.values())):
            raise ValueError("参考模式引用了音频素材，请选择 audio_vae_name")

        outputs = MiniMaxH3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt_text, width, height, length,
            ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
        )
        return (outputs.args[0], outputs.args[1], prompt_text)


# ── 风格下拉选择器（内置多风格 → 输出对应提示词；可选把模板里的 [STYLE] 替换掉） ──
# 12 种风格整段提示词（数据在 zoey_minimax_h3_styles.py，便于单独维护）
from .zoey_minimax_h3_styles import STYLE_MAP as _STYLE_MAP
from .zoey_minimax_h3_styles_dual import DUAL_STYLE_MAP as _DUAL_STYLE_MAP


class ZoeyStylePrompt:
    """下拉选风格 → 输出该风格整段 MiniMax H3 提示词；可选替换占位符。

    占位符：[MORPH] / [ESCAPE_ROUTE] / [FINAL] / [MOOD]，不填则原样保留。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (list(_STYLE_MAP.keys()), {"default": "水墨 sumi-e"}),
            },
            "optional": {
                "morph": ("STRING", {"default": "", "multiline": False, "tooltip": "[MORPH] 变形成什么，留空保留占位符"}),
                "escape_route": ("STRING", {"default": "", "multiline": False, "tooltip": "[ESCAPE_ROUTE] 逃跑路线，留空保留占位符"}),
                "final": ("STRING", {"default": "", "multiline": False, "tooltip": "[FINAL] 最终形态/落点，留空保留占位符"}),
                "mood": ("STRING", {"default": "", "multiline": False, "tooltip": "[MOOD] 情绪，留空保留占位符"}),
                "lock_background": (["true", "false"], {"default": "true", "tooltip": "开启后强化提示词：锁定首帧背景/场景/光线/构图不变，防背景漂移"}),
                "dual_ref": ("BOOLEAN", {"default": False, "tooltip": "双图版：图1真实主体不变，图2被重渲染成所选风格并与之互动；关=单图版(凭空手绘角色)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "style_label")
    FUNCTION = "run"
    CATEGORY = "Zoey/Minimax H3"

    _LOCK = (" ENTIRE FRAME STAYS FIXED: the uploaded first frame is the one true reference — its real subject "
             "(whatever it is, a person or an object) AND its whole background, scene, layout, lighting and "
             "camera framing all stay 100% identical for the entire video. The real subject keeps exactly the "
             "same identity, face, clothing, proportions and appearance it had in that first frame — no identity "
             "drift, no facial or clothing change, no re-cast, no new object added or removed, no scene or "
             "background change, no camera re-framing, pan or pull-back; the camera holds one steady locked shot. "
             "The only thing that appears and moves is the small hand-drawn animated character, which never "
             "replaces, covers or alters the real subject.")

    def run(self, style, morph="", escape_route="", final="", mood="", lock_background="true", dual_ref=False):
        src = _DUAL_STYLE_MAP if dual_ref else _STYLE_MAP
        p = src.get(style) or _STYLE_MAP.get(style, "")
        if lock_background == "true":
            if dual_ref:
                p = p.replace("nothing on walls.",
                              "nothing on walls. <Picture 1> subject and its whole real scene, background, lighting and camera framing stay 100% identical for the entire video; the stylized <Picture 2> subject never replaces, covers or alters the <Picture 1> subject or the real scene.")
            else:
                p = p.replace("never changes appearance, never vanishes, never switches.",
                              "never changes appearance, never vanishes, never switches." + self._LOCK)
                p = p.replace("nothing on walls.",
                              "nothing on walls, no identity drift, no facial change, no clothing change, no background change, no scene change, no camera re-framing, no revealing new scenery.")
        for key, val in (("[MORPH]", morph), ("[ESCAPE_ROUTE]", escape_route),
                         ("[FINAL]", final), ("[MOOD]", mood)):
            if val:
                p = p.replace(key, val)
        return (p, style)

class ZoeyMiniMaxH3LastStepsLora:
    """同轨迹两段采样：前段用基座模型（贴合参考），最后 few 步切到合成皮肤/质感 LoRA 的模型续跑。

    解决"全程叠 LoRA 会带偏参考、但纯基座又缺皮肤质感"的矛盾。
    关键：H3 是 FLOW 模型（`class ModelSampling(ModelSamplingAV, CONST)`，
    noise_scaling = σ*noise + (1-σ)*latent，inverse = latent/(1-σ)）。
    因此可做**真正不重加噪的同轨迹续采**：
      段1：基座跑 0..split（停在 σ_split）→ 输出 = x_t/(1-σ)；
      恢复 x_t = 输出*(1-σ)；
      段2：把噪声和 latent_image 都设为 x_t 喂回 → noise_scaling(σ,x_t,x_t)=x_t（恒等，不重加噪），
      合成 LoRA 模型接着 x_t 续跑到末步。→ 同一轨迹，只在最后 N 步换模型，无整帧重渲、无色块。
    之前基于 comfy.hooks 的关键帧调度会强制 DiT 在文本编码期驻留显存而爆 OOM，此实现改走 merge（add_patches），
    彻底绕开那个问题。
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import comfy.samplers
        loras = [_NONE] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "来自 ZoeyMiniMaxH3ReferenceToVideo 的 model（已含基础 LoRA，不含皮肤/质感 LoRA）"}),
                "positive": ("CONDITIONING", {"tooltip": "来自参考节点的 positive"}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件；接一个空/负面 ClipTextEncode"}),
                "latent": ("LATENT", {"tooltip": "来自参考节点的 LATENT"}),
                "lora_name": (loras, {"tooltip": "只在最后 N 步注入的 LoRA（如皮肤质感）"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "该 LoRA 的模型强度"}),
                "lora_clip": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "该 LoRA 的 CLIP 强度；纹理/皮肤 LoRA 一般 0"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "last_steps": ("INT", {"default": 5, "min": 1, "max": 200, "tooltip": "最后多少步改用「合成该 LoRA」的模型采样（皮肤/质感在收尾注入）"}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("LATENT", "分段信息")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"

    def process(self, model, positive, negative, latent, lora_name, lora_strength,
                lora_clip, seed, steps, last_steps, cfg, sampler_name, scheduler):
        import comfy.sample
        import comfy.samplers

        # 后段模型 = 基座 + 该 LoRA；前段 = 基座
        model_lora, _ = _apply_lora(model, None, lora_name, lora_strength, lora_clip)

        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent_image, latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise0 = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        if last_steps <= 0:
            samples = comfy.sample.sample(
                model, noise0, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                denoise=1.0, force_full_denoise=True, noise_mask=latent.get("noise_mask"), seed=seed)
            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples
            return (out, f"全程基座采样：{steps} 步")

        device = model.load_device
        ms = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, steps).to(device)
        split = max(1, min(steps, steps - last_steps))
        sigmas1 = sigmas[:split + 1]
        sigmas2 = sigmas[split:]
        sampler = comfy.samplers.ksampler(sampler_name)
        # 实时预览（AV 嵌套 latent 会取视频流 tensors[0] 解码）
        import latent_preview
        callback = latent_preview.prepare_callback(model, steps)

        # 段1：基座跑 0..split（停在 sigma[split]，未全去噪）
        pass1 = comfy.sample.sample_custom(
            model, noise0, cfg, sampler, sigmas1, positive, negative, latent_image,
            callback=callback, seed=seed)
        # CONST flow: sample 输出 = inverse_noise_scaling(σ, x) = x/(1-σ)，恢复原始 x_t
        sigma_split = float(sigmas1[-1])
        x_t = pass1 * (1.0 - sigma_split)

        # 段2：合成 LoRA 模型，从 x_t 续跑 sigmas2。noise=latent=x_t →
        # noise_scaling(σ,x_t,x_t)=σ*x_t+(1-σ)*x_t=x_t（恒等，不重加噪），真正同轨迹续采。
        pass2 = comfy.sample.sample_custom(
            model_lora, x_t, cfg, sampler, sigmas2, positive, negative, x_t,
            callback=callback, seed=seed)

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = pass2
        info = (f"同轨迹两段：基座 0~{split}步 → 合成LoRA {split}~{steps}步"
                f"（最后{last_steps}步）注入 {lora_name}")
        return (out, info)


class ZoeyMiniMaxH3Ref2VA_LastSteps:
    """自包含版「参考转视频 + 最后 N 步注入 LoRA」两段采样节点。

    内部复用 ZoeyMiniMaxH3ReferenceToVideo 生成参考条件与 AC/AV latent，
    再走「基座整段 → LoRA 精修」：段1 基座去噪→贴合参考/运镜；段2 合成皮肤 LoRA 模型做
    denoise<1 精修→收尾注入质感。一个节点搞定：直接出视频帧。未选 last_lora 时等价于全程基座参考转视频。
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import comfy.samplers
        base = ZoeyMiniMaxH3ReferenceToVideo.INPUT_TYPES()
        loras = [_NONE] + folder_paths.get_filename_list("loras")
        base["required"].update({
            "negative": ("CONDITIONING", {"tooltip": "负向条件；接一个空/负面 ClipTextEncode"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
            "last_steps": ("INT", {"default": 5, "min": 0, "max": 200, "tooltip": "最后多少步改用「合成该LoRA」的模型采样；0=全程基座（等价普通参考转视频）"}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            "last_lora_name": (loras, {"default": _NONE, "tooltip": "只在最后 N 步注入的 LoRA（如皮肤质感）；无=全程基座"}),
            "last_lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            "last_lora_clip": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "该 LoRA 的 CLIP 强度；纹理/皮肤 LoRA 一般 0"}),
        })
        # negative 在本节点是必连（两段采样要用），从基座 optional 移除避免重复
        base["optional"].pop("negative", None)
        return base

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("LATENT", "视频帧", "提示词/分段信息")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"

    def process(self, prompt, resolution, aspect, duration, ref_image_size,
                auto_declaration, director_mode, director_shots, library, mode,
                sage_attention, dialogue_convert, ref_purposes, unet_name, clip_name,
                vae_name, audio_vae_name, lora_list, loras, layout, layout_enabled,
                negative, seed, steps, last_steps, cfg, sampler_name, scheduler,
                last_lora_name, last_lora_strength, last_lora_clip, **refs):
        import nodes
        # 复用参考节点生成 model(vae/audio_vae) + positive(cond) + latent
        ref_node = ZoeyMiniMaxH3ReferenceToVideo()
        model, vae, audio_vae, cond, latent_in, prompt_text, _lv_frames = ref_node.process(
            unet_name, clip_name, vae_name, audio_vae_name, lora_list, loras,
            prompt, mode, resolution, aspect, duration, ref_image_size, auto_declaration,
            director_mode, director_shots, library, sage_attention, dialogue_convert,
            ref_purposes, layout, layout_enabled, **refs)

        # 无 last_lora 或 last_steps==0：纯基座，一次采样到底
        if last_lora_name == _NONE or last_steps <= 0:
            import comfy.sample
            latent_image = comfy.sample.fix_empty_latent_channels(
                model, latent_in["samples"], latent_in.get("downscale_ratio_spacial", None),
                latent_in.get("downscale_ratio_temporal", None))
            batch_inds = latent_in["batch_index"] if "batch_index" in latent_in else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler, cond, negative, latent_image,
                denoise=1.0, noise_mask=latent_in.get("noise_mask"), seed=seed)
            out = latent_in.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples
            info = f"全程基座采样（未注入 last_lora）：{steps} 步"
        else:
            out, info = ZoeyMiniMaxH3LastStepsLora().process(
                model, cond, negative, latent_in, last_lora_name, last_lora_strength,
                last_lora_clip, seed, steps, last_steps, cfg, sampler_name, scheduler)
            info = prompt_text.replace("\n", " ") + " ｜ " + info

        # 视频 latent 解码出视频帧（可直接接 VideoCombine）
        frames = nodes.VAEDecode().decode(out["samples"], vae)[0]
        return (out, frames, info)


class ZoeyMiniMaxH3LongVideo:
    """长视频驱动节点：复用主节点已加载 model/vae/audio_vae，自己加载 clip，逐段提示词、末帧续接、按序拼接。

    接法：主节点出 model/vae/audio_vae/视频帧/音频 给本节点；本节点内部循环调用主节点生成逻辑——
    每轮把上一段末尾帧返回给主节点生成逻辑当参考图，采样出下一段（用户逐段填提示词）。全部完成后
    torch.cat 拼帧 + 音频按序拼接 → IMAGE+AUDIO → 外置 VHS VideoCombine 出 mp4。
    画布/时长/步数由主节点首帧形状自动推导（无需重复填写采样/时长/分辨率）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "first_frames": ("IMAGE", {"tooltip": "主节点「视频帧」输出：第1段 + 末帧续接的种子"}),
            "first_audio": ("AUDIO", {"tooltip": "主节点「音频」输出（可选；无音频则输出纯视频）"}),
            "negative": ("CONDITIONING",),
        }
        for i in range(_IMAGE_SLOTS):
            opt[f"ref_image_{i}"] = ("IMAGE",)
        for i in range(_VIDEO_SLOTS):
            opt[f"ref_video_{i}"] = ("IMAGE",)
            opt[f"ref_video_audio_{i}"] = ("AUDIO",)
        for i in range(_AUDIO_SLOTS):
            opt[f"ref_audio_{i}"] = ("AUDIO",)
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "主节点「model」输出（复用已加载模型，不重复加载）"}),
                "vae": ("VAE", {"tooltip": "主节点「vae」输出"}),
                "audio_vae": ("VAE", {"tooltip": "主节点「audio_vae」输出（无音频可为空）"}),
                "clip": ("CLIP", {"tooltip": "主节点「clip」输出（post-LoRA，直接复用）"}),
                "settings": ("STRING", {"multiline": True,
                                        "tooltip": "主节点「设置」输出：steps/cfg/sampler_name/scheduler/seed，自动同步采样参数"}),
                "segments": ("INT", {"default": 2, "min": 1, "max": 40,
                                     "tooltip": "总段数：第1段用主节点首帧，此节点续接 2..N。"}),
            },
            "optional": {
                **opt,
                # 固定槽位：第2段起每段一个提示词框（最多40段→39个）。前端按 segments 用 toggleWidget 显示/隐藏，
                # 预定义而非运行时增删，保证 widget 列表恒定 → widgets_values 不串位、序列化稳定。
                **{f"prompt_{i}": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": f"第 {i + 2} 段提示词"}) for i in range(39)},
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("视频帧", "音频")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"
    OUTPUT_NODE = True

    def process(self, model, vae, audio_vae, clip, settings="{}", segments=2,
                first_frames=None, first_audio=None, negative=None, **kwargs):
        import torch, json
        segments = max(1, int(segments or 2))
        if first_frames is None:
            raise ValueError("长视频节点需要接入主节点「视频帧」作为第1段。")
        sp = {}
        try:
            sp = json.loads(settings or "{}")
        except Exception:
            sp = {}
        if not isinstance(sp, dict):
            sp = {}
        steps = int(sp.get("steps", 25) or 25)
        cfg = float(sp.get("cfg", 4.0) or 4.0)
        sampler_name = sp.get("sampler_name", "res_multistep") or "res_multistep"
        scheduler = sp.get("scheduler", "simple") or "simple"
        seed = int(sp.get("seed", 0) or 0)
        # 第2段起每段一个提示词框（前端按 segments 显示/隐藏 prompt_0..N）；留空 = 沿用上一段（链式续写）
        plist = [kwargs.get(f"prompt_{i}", "") or "" for i in range(max(0, segments - 1))]
        # 画布 + 每段帧数：直接从主节点首段帧推导，保证各段能拼接
        H, W = first_frames.shape[1], first_frames.shape[2]
        dur = first_frames.shape[0] / 24.0
        # 参考素材：与主节点同源接入（保留全部人物/场景/音频），后门透传给 _segment
        refs = {k: v for k, v in kwargs.items() if k.startswith("ref_") and v is not None}
        ref_node = ZoeyMiniMaxH3ReferenceToVideo()
        seg_frames = [first_frames]
        seg_audios = [first_audio] if first_audio is not None else []
        prev = first_frames[-1:]
        last_prompt = ""
        for i in range(2, segments + 1):
            idx = i - 2
            p = plist[idx] if idx < len(plist) else ""
            if not (p or "").strip():
                p = last_prompt  # 该段提示词留空 → 沿用上一段（链式续写）
            last_prompt = p
            _m, _v, _av, _c, _cond, _lat, _info, fr, au, _set = ref_node._segment(
                model=model, clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=p, mode="参考", resolution="自动", aspect="自动", duration=dur,
                ref_image_size="match", auto_declaration=False, director_mode=False,
                director_shots="[]", library="", tray="[]", sage_attention=False,
                dialogue_convert=True, ref_purposes="{}", layout="{}", layout_enabled=False,
                negative=negative, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
                scheduler=scheduler, last_lora_name=_NONE, last_lora_strength=1.0,
                last_lora_clip=0.0, last_steps=0, width_override=W, height_override=H,
                prev_frame=prev, **refs)
            seg_frames.append(fr)
            if au is not None:
                seg_audios.append(au)
            prev = fr[-1:]
        out = torch.cat(seg_frames, dim=0)
        audio = None
        if len(seg_audios) == segments and all(a is not None for a in seg_audios):
            audio = dict(seg_audios[0])
            audio["waveform"] = torch.cat([a["waveform"] for a in seg_audios], dim=-1)
        return (out, audio)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS["ZoeyStylePrompt"] = ZoeyStylePrompt
NODE_DISPLAY_NAME_MAPPINGS["ZoeyStylePrompt"] = "Zoey - 风格下拉选择 (@)"

if _H3_AVAILABLE:
    NODE_CLASS_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = ZoeyMiniMaxH3ReferenceToVideo
    NODE_DISPLAY_NAME_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = "Zoey - MiniMax H3 参考转视频 (@)"
    NODE_CLASS_MAPPINGS["ZoeyMiniMaxH3LongVideo"] = ZoeyMiniMaxH3LongVideo
    NODE_DISPLAY_NAME_MAPPINGS["ZoeyMiniMaxH3LongVideo"] = "Zoey - MiniMax H3 长视频 (循环·末帧续接)"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
