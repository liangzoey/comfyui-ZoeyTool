# -*- coding: utf-8 -*-
"""
Zoey Monitor Node - 系统监控节点
独立 HTTP 服务 + 后台显存自动清理
"""

import gc
import json
import subprocess
import torch
import http.server as _hs
import threading as _th
import json as _json
import time as _time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _cleanup(force=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if force:
            torch.cuda.synchronize()
    # 使用 ComfyUI 模型管理释放显存
    try:
        from comfy import model_management as _mm
        _mm.soft_empty_cache()
        if force:
            _mm.cleanup_models()
    except Exception:
        pass


def _vram():
    """PyTorch 已分配显存（用于自动清理阈值判断，只算本进程 torch 的占用）。"""
    if not torch.cuda.is_available():
        return 0.0, 1.0, 0.0
    p = torch.cuda.get_device_properties(0)
    t = p.total_memory / (1024**3)
    a = torch.cuda.memory_allocated() / (1024**3)
    return a, t, (a / t * 100) if t > 0 else 0.0


def _vram_system():
    """系统级显存占用（含其他进程），用于显示，与 nvidia-smi 一致。"""
    if not torch.cuda.is_available():
        return 0.0, 1.0, 0.0
    free, total = torch.cuda.mem_get_info()
    used = max(0.0, total - free)
    return used / (1024**3), total / (1024**3), (used / total * 100) if total > 0 else 0.0


_PSUTIL_LOCK = _th.Lock()


def _cpu_ram():
    if not HAS_PSUTIL:
        return None
    try:
        # psutil.cpu_percent 使用全局采样，多个线程并发调用会互相污染，必须加锁
        with _PSUTIL_LOCK:
            cpu = psutil.cpu_percent(interval=0.1)
        r = psutil.virtual_memory()
        return cpu, r.used / (1024**3), r.total / (1024**3), r.percent
    except Exception:
        return None


def _gpu_info():
    """返回 (温度°C, 利用率%)，异常时 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=3, stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _collect():
    s = {}
    cr = _cpu_ram()
    if cr:
        s["cpu"] = round(cr[0], 1)
        s["ram_used"] = round(cr[1], 2)
        s["ram_total"] = round(cr[2], 2)
        s["ram"] = round(cr[3], 1)
    vu, vt, vp = _vram_system()
    s["vram_used"] = round(vu, 2)
    s["vram_total"] = round(vt, 2)
    s["vram"] = round(vp, 1)
    gi = _gpu_info()
    if gi:
        s["gpu_temp"] = gi[0]
        s["gpu_util"] = gi[1]
    return s


# ── 全局配置 ──
_config = {"threshold": 85, "aggressive": False}
_config_lock = _th.Lock()


# ====================== 节点 ======================
class ZoeySystemMonitor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold": ("FLOAT", {"default": 85, "min": 10, "max": 99, "step": 1}),
                "aggressive": ("BOOLEAN", {"default": False, "label_on": "激进", "label_off": "标准"}),
            },
            "optional": {
                "passthrough": ("*",),
            },
            "hidden": {
                "_cached": ("STRING", {"default": json.dumps(_collect())}),
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("passthrough", "stats_json")
    FUNCTION = "run"
    CATEGORY = "Zoey Tool/系统工具"
    OUTPUT_NODE = True

    def run(self, threshold, aggressive, passthrough=None, _cached=""):
        with _config_lock:
            _config["threshold"] = threshold
            _config["aggressive"] = aggressive

        stats = _collect()
        stats["threshold"] = threshold
        stats["cleaned"] = False
        stats["bg_cleaned"] = False

        allocated, total, pct = _vram()

        if not torch.cuda.is_available():
            gc.collect()
            stats["cleaned"] = True
            return {"ui": {"monitor": (json.dumps(stats, ensure_ascii=False),)}, "result": (passthrough, json.dumps(stats, ensure_ascii=False))}

        if pct >= threshold:
            before = allocated
            _cleanup(force=aggressive)
            after, _, pct2 = _vram()
            freed = before - after
            stats["cleaned"] = True
            stats["freed_gb"] = round(freed, 2)

        return {"ui": {"monitor": (json.dumps(stats, ensure_ascii=False),)}, "result": (passthrough, json.dumps(stats, ensure_ascii=False))}


NODE_CLASS_MAPPINGS = {"ZoeySystemMonitor": ZoeySystemMonitor}
NODE_DISPLAY_NAME_MAPPINGS = {"ZoeySystemMonitor": "Zoey - 系统监控"}


# ── 独立 HTTP Server + 后台显存监控 ──
_ZOEY_PORT = 18888
_LATEST_STATS = _collect()
_BG_CLEANED = False
_BG_FREED = 0.0


class _ZoeyHandler(_hs.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(_json.dumps(_LATEST_STATS).encode("utf-8"))
        elif self.path == "/cleanup":
            _cleanup(force=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            st = _collect()
            st["cleaned"] = True
            st["freed_gb"] = 0
            self.wfile.write(_json.dumps(st).encode("utf-8"))
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Zoey Monitor")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass


def _monitor_loop():
    global _LATEST_STATS, _BG_CLEANED, _BG_FREED
    while True:
        # 采集系统状态
        stats = _collect()

        # 后台显存清理：读取用户设置的阈值
        with _config_lock:
            thresh = _config["threshold"]
            aggr = _config["aggressive"]

        # 清理判断用 torch 已分配显存（_vram），避免被其他进程占用触发误清理
        if torch.cuda.is_available() and _vram()[2] >= thresh:
            before, _, _ = _vram()
            _cleanup(force=aggr)
            after, _, _ = _vram()
            freed = before - after
            if freed > 0.0001:
                _BG_CLEANED = True
                _BG_FREED = freed
                stats["bg_cleaned"] = True
                stats["bg_freed_gb"] = round(freed, 2)
                # 重新采集
                stats.update(_collect())

            # 如果清理后仍超阈值，再清一次
            if _vram()[2] >= thresh:
                _cleanup(force=True)
                stats.update(_collect())

        stats["bg_cleaned"] = _BG_CLEANED
        if _BG_CLEANED:
            stats["bg_freed_gb"] = round(_BG_FREED, 2)

        _LATEST_STATS = stats
        _time.sleep(1)


def _start_server():
    try:
        server = _hs.HTTPServer(("127.0.0.1", _ZOEY_PORT), _ZoeyHandler)
        _th.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[Zoey] HTTP 服务已启动: http://127.0.0.1:{_ZOEY_PORT}/stats")
        _th.Thread(target=_monitor_loop, daemon=True).start()
    except Exception as e:
        print(f"[Zoey] HTTP 服务启动失败: {e}")

_th.Thread(target=_start_server, daemon=True).start()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
