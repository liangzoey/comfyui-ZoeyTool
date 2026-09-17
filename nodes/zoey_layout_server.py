# -*- coding: utf-8 -*-
"""Zoey 3D 白模导演台布局截图 HTTP 接口。

布局截图（previs 渲染图）存 <ComfyUI>/input/zoey_layout/，供 H3 节点在 process 时
作为「构图参考」图自动注入。预览走标准 /view?type=input&subfolder=zoey_layout。
前端由 Three.js 渲染当前视角 → toDataURL → POST /zoey/layout/save 上传拿 filename。
"""

import os
import re
import time

import folder_paths
from aiohttp import web
from server import PromptServer

_LAYOUT_SUBFOLDER = "zoey_layout"
LAYOUT_DIR = os.path.join(folder_paths.get_input_directory(), _LAYOUT_SUBFOLDER)

_SAFE = re.compile(r"[^0-9A-Za-z_\-]+")


@PromptServer.instance.routes.post("/zoey/layout/save")
async def zoey_layout_save(request):
    """接收一张 PNG 数据，存进 input/zoey_layout/，返回 {filename}。

    请求：multipart 表单，字段 file = 图片数据（filename 可带扩展名）。
    返回：{"filename": "layout_<ts>_<safe>.png"}
    """
    reader = await request.multipart()
    saved = None
    async for part in reader:
        if part.name != "file":
            continue
        orig = (part.filename or "layout.png").replace("\\", "/")
        base = os.path.splitext(os.path.basename(orig))[0]
        ext = os.path.splitext(orig)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        safe = _SAFE.sub("", base)[:40] or "layout"
        name = f"{time.time_ns()}_{safe}{ext}"
        os.makedirs(LAYOUT_DIR, exist_ok=True)
        path = os.path.join(LAYOUT_DIR, name)
        with open(path, "wb") as f:
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
        saved = name
    if not saved:
        return web.json_response({"error": "no file provided"}, status=400)
    return web.json_response({"filename": saved})
