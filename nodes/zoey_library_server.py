# -*- coding: utf-8 -*-
"""Zoey 永久素材库 HTTP 接口。

媒体存 <ComfyUI>/input/zoey_library/，清单 library.json（条目数组顺序即 @L 序号）。
预览用标准 /view?filename=X&type=input&subfolder=zoey_library。
"""

import json
import os
import re
import time

import folder_paths
from aiohttp import web
from server import PromptServer

_LIB_SUBFOLDER = "zoey_library"
LIB_DIR = os.path.join(folder_paths.get_input_directory(), _LIB_SUBFOLDER)
MANIFEST_PATH = os.path.join(LIB_DIR, "library.json")

_SAFE = re.compile(r"[^0-9A-Za-z_\-]+")


def _load_manifest():
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError):
        return []


def _save_manifest(entries):
    os.makedirs(LIB_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)


@PromptServer.instance.routes.get("/zoey/library")
async def zoey_library_get(request):
    return web.json_response({"entries": _load_manifest()})


@PromptServer.instance.routes.post("/zoey/library/upload")
async def zoey_library_upload(request):
    reader = await request.multipart()
    saved = None
    kind = None
    async for part in reader:
        if part.name == "kind":
            raw = await part.read()
            kind = raw.decode("utf-8", "ignore") if raw else None
        elif part.name == "file":
            orig = (part.filename or "upload").replace("\\", "/")
            base = os.path.splitext(os.path.basename(orig))[0]
            ext = os.path.splitext(orig)[1].lower()
            safe = _SAFE.sub("", base)[:40] or "upload"
            name = f"{time.time_ns()}_{safe}{ext}"
            os.makedirs(LIB_DIR, exist_ok=True)
            path = os.path.join(LIB_DIR, name)
            with open(path, "wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            saved = name
    if not saved:
        return web.json_response({"error": "no file provided"}, status=400)
    return web.json_response({"filename": saved, "kind": kind})


@PromptServer.instance.routes.post("/zoey/library/save")
async def zoey_library_save(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad json"}, status=400)
    entries = body.get("entries", [])
    if not isinstance(entries, list):
        return web.json_response({"error": "entries must be a list"}, status=400)
    _save_manifest(entries)
    return web.json_response({"ok": True})


@PromptServer.instance.routes.post("/zoey/library/delete_file")
async def zoey_library_delete_file(request):
    """删除清单里已无引用的孤儿文件，防磁盘膨胀。"""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad json"}, status=400)
    filename = str(body.get("filename", "") or "").strip()
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return web.json_response({"error": "bad filename"}, status=400)
    entries = _load_manifest()
    used = {e.get("file") for e in entries} | {e.get("audio_file") for e in entries}
    if filename in used:
        return web.json_response({"error": "file still in use"}, status=400)
    path = os.path.join(LIB_DIR, filename)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return web.json_response({"ok": True})
