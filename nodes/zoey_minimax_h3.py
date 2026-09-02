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
                      dialogue_convert=True):
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
        # 跨镜一致性约束：从第二镜起补一句保持角色/场景/服装一致
        if i > 0 and consistent:
            shot_text += "\n保持与上一镜相同的角色、场景、服装与光线。"
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
        loras = [_NONE] + folder_paths.get_filename_list("loras")
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
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "VAE", "VAE", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "vae", "audio_vae", "positive", "LATENT", "提示词(已转换)")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"

    def process(self, unet_name, clip_name, vae_name, audio_vae_name,
                lora_list, loras,
                prompt, mode, resolution, aspect, duration,
                ref_image_size, auto_declaration, director_mode, director_shots,
                library="", sage_attention=False,
                dialogue_convert=True, ref_purposes="{}", **refs):
        # ── 内置模型加载：UNet + CLIP + VAE + 音频 VAE，再叠 LoRA ──
        model = _load_unet(unet_name)
        clip = _load_clip(clip_name)
        vae = _load_vae(vae_name)
        audio_vae = _load_vae(audio_vae_name) if audio_vae_name and audio_vae_name != _NONE else None
        for lora_name, lora_model, lora_clip in _parse_loras(loras):
            model, clip = _apply_lora(model, clip, lora_name, lora_model, lora_clip)

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

        image_slots = sorted(i for i in range(_IMAGE_SLOTS) if refs.get(f"ref_image_{i}") is not None)
        has_video = any(v is not None for v in ref_videos.values())
        has_audio = any(a is not None for a in ref_audios.values()) \
            or any(a is not None for a in ref_video_audios.values())

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

        width, height = _compute_canvas(ref_images, resolution, aspect)

        if eff in ("T2V", "I2V"):
            # I2V：第一张已连接参考图作首帧；≥2 张时最后一张作尾帧（首尾帧）
            first_frame = ref_images[f"ref_image_{image_slots[0]}"] if (eff == "I2V" and image_slots) else None
            last_frame = ref_images[f"ref_image_{image_slots[-1]}"] if (eff == "I2V" and len(image_slots) >= 2) else None
            cond, latent, prompt_text = self._run_text_image(
                clip, vae, prompt, width, height, duration,
                director_mode, director_shots, eff == "I2V",
                first_frame, dialogue_convert, last_frame)
        else:
            cond, latent, prompt_text = self._run_ref2va(
                clip, vae, audio_vae, prompt, ref_images, ref_videos,
                ref_video_audios, ref_audios, counts, width, height,
                duration, ref_image_size, auto_declaration, director_mode,
                director_shots, library, image_slots, dialogue_convert,
                ref_purposes)

        # 集成 SageAttention：在模型克隆上启用，随 CONDITIONING/LATENT 一并返回
        if sage_attention:
            model = _apply_sage_attention(model)
        return (model, vae, audio_vae, cond, latent, prompt_text)

    def _compose_plain_director(self, shots_json, dialogue_convert=True):
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
            if i > 0 and consistent:
                shot_text += "\n保持与上一镜相同的角色、场景、服装与光线。"
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
                        dialogue_convert=True, last_frame=None):
        """T2V / I2V：走官方 MiniMaxH3ImageToVideo，不做 @ 引用展开。"""
        if director_mode:
            prompt_text, total = self._compose_plain_director(director_shots, dialogue_convert)
            length = _frame_count(total)
        else:
            prompt_text = (prompt or "").strip()
            length = _frame_count(duration)
        outputs = MiniMaxH3ImageToVideo.execute(
            clip, vae, prompt_text, width, height, length,
            first_frame=first_frame if use_first_frame else None,
            last_frame=last_frame if use_first_frame else None,
        )
        return (outputs.args[0], outputs.args[1], prompt_text)

    def _run_ref2va(self, clip, vae, audio_vae, prompt, ref_images, ref_videos,
                    ref_video_audios, ref_audios, counts, width, height, duration,
                    ref_image_size, auto_declaration, director_mode, director_shots,
                    library, image_slots, dialogue_convert=True, ref_purposes="{}"):
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
                pic_of, aud_of, dialogue_convert)
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "style_label")
    FUNCTION = "run"
    CATEGORY = "Zoey/Minimax H3"

    def run(self, style, morph="", escape_route="", final="", mood=""):
        p = _STYLE_MAP.get(style, "")
        for key, val in (("[MORPH]", morph), ("[ESCAPE_ROUTE]", escape_route),
                         ("[FINAL]", final), ("[MOOD]", mood)):
            if val:
                p = p.replace(key, val)
        return (p, style)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS["ZoeyStylePrompt"] = ZoeyStylePrompt
NODE_DISPLAY_NAME_MAPPINGS["ZoeyStylePrompt"] = "Zoey - 风格下拉选择 (@)"

if _H3_AVAILABLE:
    NODE_CLASS_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = ZoeyMiniMaxH3ReferenceToVideo
    NODE_DISPLAY_NAME_MAPPINGS["ZoeyMiniMaxH3ReferenceToVideo"] = "Zoey - MiniMax H3 参考转视频 (@)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
