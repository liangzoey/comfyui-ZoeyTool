# -*- coding: utf-8 -*-
"""MiniMax H3 @ 标签解析纯逻辑（无 comfy 依赖，便于单测）。

用户提示词里用 @P1/@V1/@A1 引用参考素材，运行时展开为 H3 原生标签：
    @P1 -> <Picture 1>    第 1 张已连接的参考图
    @V1 -> <Video 1>      第 1 段已连接的参考视频
    @A1 -> <Audio 1>      第 1 段已连接的参考音频
编号按各类型的连接顺序从 1 开始，忽略中间空槽位。

H3 的 Audio 计数顺序与节点内部一致：所有"带音轨的参考视频"先入队（按视频连接顺序），
随后才是独立参考音频。因此 @A 的总数 = 有音轨的视频数 + 独立音频数。
"""

import re

_TAG_RE = re.compile(r"@([PpVvAa])\s*(\d+)")

_TYPE_LIMIT_KEY = {"P": "pictures", "V": "videos", "A": "audios"}
_TYPE_TAG = {"P": "Picture", "V": "Video", "A": "Audio"}
_TYPE_LABEL_ZH = {"P": "参考图", "V": "参考视频", "A": "参考音频"}


def count_refs(image_count, video_count, video_audio_count, audio_count):
    """汇总 H3 各类型参考数量。

    video_audio_count 只统计"视频与音轨同时已连接"的槽位，因为 H3 节点仅在
    视频存在时才把它配套的音轨作为 Audio 项入队。
    """
    paired = min(video_count, video_audio_count)
    return {
        "pictures": image_count,
        "videos": video_count,
        "audios": paired + audio_count,
    }


def expand_at_tags(prompt, counts):
    """把 @P1/@V1/@A1 展开为 <Picture N>/<Video N>/<Audio N>。

    越界引用抛出 ValueError 并提示当前已连接数量。
    """
    def repl(match):
        kind = match.group(1).upper()
        idx = int(match.group(2))
        limit = counts[_TYPE_LIMIT_KEY[kind]]
        if idx < 1:
            raise ValueError(f"@{kind}{idx} 索引从 1 开始")
        if idx > limit:
            raise ValueError(
                f"@{kind}{idx} 超出范围：当前只连接了 {limit} 个{_TYPE_LABEL_ZH[kind]}")
        return f"<{_TYPE_TAG[kind]} {idx}>"

    return _TAG_RE.sub(repl, prompt)


def _join_names(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_declaration(counts):
    """按已连接参考素材生成 H3 的引用声明行，形如：
    Use <Picture 1> and <Picture 2> as reference frames, and <Video 1> as reference motion, and <Audio 1> exactly as it is.
    无任何参考素材时返回空字符串。
    """
    parts = []
    if counts["pictures"]:
        pics = [f"<Picture {i}>" for i in range(1, counts["pictures"] + 1)]
        frame = "frame" if counts["pictures"] == 1 else "frames"
        parts.append(f"{_join_names(pics)} as reference {frame}")
    if counts["videos"]:
        vids = [f"<Video {i}>" for i in range(1, counts["videos"] + 1)]
        parts.append(f"{_join_names(vids)} as reference motion")
    if counts["audios"]:
        auds = [f"<Audio {i}>" for i in range(1, counts["audios"] + 1)]
        parts.append(f"{_join_names(auds)} exactly as it is")
    if not parts:
        return ""
    return "Use " + ", and ".join(parts) + "."
