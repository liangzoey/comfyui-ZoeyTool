# -*- coding: utf-8 -*-
"""Zoey - 音乐生成：把 YuE2 整条管线（外加写词/变声/转写）压成一个节点。

纯后端节点：不暴露中间图，用户只填「描述 + 时长」就能出歌。
内部按顺序做这些事（每一步的显存都不和时间轴重叠）：

    [写词]  llama.cpp + 本地 GGUF      → 一句话变成风格串 + 歌词
    [生成]  YuE2 核心节点（进程内直调）  → ABC 写谱 → 音乐 token → 采样 → 解码
    [变声]  Demucs + Seed-VC          → 换音色后混回伴奏
    [转写]  Qwen3-ASR                 → 出歌词文本

关于「时长」：YuE2 的 max_duration 只是 token 预算上限（×25 帧/秒），
长歌词会被上下文静默压小，模型也会提前收尾。所以本节点用真实长度回环：
把 YuE2GenerateMusic 返回的 seconds 喂给潜空间节点，并在结果面板回报实际时长。
"""

import atexit
import contextlib
import gc
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

import torch
import folder_paths
from comfy.utils import ProgressBar

logger = logging.getLogger("ZoeyTool")

# ====================== 外部引擎路径 ======================
# 这些是机器上的绝对路径，换机器需要改这里。
LLAMA_SERVER = r"C:\Users\liangzoey\llama.cpp\bin\llama-server.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
GGUF_DIR = r"C:\Users\liangzoey\models"

ASR_PYTHON = r"D:\YuEStudio\runtime\asrvenv\Scripts\python.exe"
ASR_SCRIPT = r"D:\YuEStudio\resources\asr.py"
ASR_MODEL_ROOT = r"C:\Users\liangzoey\models\asr"

VC_PYTHON = r"D:\YuEStudio\runtime\vcenv\Scripts\python.exe"
VC_SCRIPT = r"D:\YuEStudio\resources\voice_convert.py"
SEEDVC_DIR = r"D:\YuEStudio\runtime\seedvc"
VC_FFMPEG = r"F:\ComfyUIClassic\ComfyUIClassic\libs\ffmpeg\Library\bin\ffmpeg.exe"

# 每行唱词约 7 秒（对照官方模板 16 行 ≈ 115 秒）
SECONDS_PER_LINE = 7
# YuE2 一帧 = 1/25 秒
FRAMES_PER_SECOND = 25
# YuE2 上下文窗口，超过这个数的 token 预算会被静默截断
YUE2_CONTEXT = 24576

TAKES_MAX = 4
SEED_STRIDE = 7919

# 采样参数固定成 YuE Studio 一直在用的那套。它们对成品几乎没影响，
# 但摆出来会让人以为能调——cfg 尤其危险（YuE2 的 CFG 在文本编码器内部做，这里改大只会炸）。
DEFAULT_STEPS = 32
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "dpm_2"
DEFAULT_SCHEDULER = "sgm_uniform"
DEFAULT_ABC_MODE = "full"
DEFAULT_LORA_STRENGTH = 1.0

# ====================== 风格选择器 ======================
# 和 YuE Studio 的选项表一致（src/renderer/src/lib/styleOptions.ts）。
# ComfyUI 的 COMBO 就是服务端定义的下拉框，不需要任何前端代码。
NONE = "无"
STYLE_LANGUAGES = ["Chinese", "Cantonese", "English", "Chinese, Japanese"]
STYLE_GENRES = ["folk", "contemporary pop", "alternative rock", "synthwave",
                "chinese traditional", "hip hop", "jazz", "r&b"]
STYLE_VOCALS = ["warm female vocal", "ethereal female vocal", "husky male vocal",
                "clear male vocal", "gritty male vocal with raspy edge", "soft child vocal"]
STYLE_INSTRUMENTS = ["fingerstyle acoustic guitar", "piano", "electric guitar", "bass",
                     "live drums", "synths", "strings", "harmonica", "guzheng",
                     "chinese bamboo flute"]
STYLE_PRODUCTIONS = ["warm analog production", "clean modern production",
                     "raw live-room production", "retro 80s production", "lo-fi intimate"]
INSTRUMENT_CHOICES = [NONE] + STYLE_INSTRUMENTS


def _compose_style(language, vocal, genre, bpm, instruments, production):
    """拼风格串，公式和 YuE Studio 的 composeStyle 一致：

        语言 + 人声 + 曲风 + BPM + 乐器 + 制作质感

    BPM 必须带单位（写 78 BPM 而不是 78），语言必须显式写出来
    （漏了会用英文口音唱中文）——这两条是踩过坑的。
    """
    parts = [language, vocal, genre, f"{int(round(bpm))} BPM"]
    picked = [i for i in instruments if i and i != NONE]
    if picked:
        parts.append(", ".join(picked))
    parts.append(production)
    return ", ".join(p for p in parts if p)

# ====================== 子进程治理 ======================
_PROCS = {}
_PROCS_LOCK = threading.Lock()


def _kill_tree(pid):
    """连子孙一起杀：voice_convert.py 会自我复衍生出孙进程。"""
    if not pid:
        return
    try:
        subprocess.run(
            ["taskkill", "/pid", str(pid), "/T", "/F"],
            capture_output=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        pass


@atexit.register
def _reap_orphans():
    with _PROCS_LOCK:
        pids = list(_PROCS.keys())
    for pid in pids:
        _kill_tree(pid)


class ExternalProcessError(RuntimeError):
    pass


@contextlib.contextmanager
def _external(cmd, env=None, cwd=None, label="外部进程", drain_stdout=True):
    """起一个子进程，无论怎么退出都保证连子孙一起收掉。

    只 catch Exception 的调用方不会吞掉 Cancel：ComfyUI 的
    InterruptProcessingException 继承自 BaseException，会一路穿透，而本函数的
    finally 照样执行。
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    with _PROCS_LOCK:
        _PROCS[proc.pid] = proc
    logger.info(f"[Zoey音乐] 启动{label}，pid={proc.pid}")

    stderr_tail = []
    drainers = []

    def _drain(stream, sink):
        try:
            for line in iter(stream.readline, ""):
                sink.append(line.rstrip())
                if len(sink) > 60:
                    del sink[0]
        except Exception:
            pass

    if drain_stdout:
        drainers.append(threading.Thread(target=_drain, args=(proc.stdout, []), daemon=True))
    drainers.append(threading.Thread(target=_drain, args=(proc.stderr, stderr_tail), daemon=True))
    for t in drainers:
        t.start()

    try:
        yield proc, stderr_tail
    finally:
        _kill_tree(proc.pid)
        with _PROCS_LOCK:
            _PROCS.pop(proc.pid, None)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        for s in (proc.stdout, proc.stderr, proc.stdin):
            try:
                if s:
                    s.close()
            except Exception:
                pass


def _tail(lines, n=6):
    return "\n".join(lines[-n:]) if lines else "（无输出）"


# ====================== 显存仲裁 ======================
def _release_vram():
    """把 ComfyUI 持有的模型全部卸下，为外部引擎腾显存。

    不需要回抢：ModelPatcher 对象仍然有效，ComfyUI 会按需重新加载权重。
    """
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as e:
        logger.warning(f"[Zoey音乐] 卸载模型时出错（忽略）：{e}")
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _check_interrupt():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


# ====================== 控件候选项 ======================
def _yue2_ckpt_list():
    try:
        names = folder_paths.get_filename_list("checkpoints")
    except Exception:
        return []
    hit = [n for n in names if "yue2" in os.path.basename(n).lower()]
    return hit or names


def _lora_list():
    try:
        names = folder_paths.get_filename_list("loras")
    except Exception:
        return ["无"]
    # YuE2 的 LoRA 才有效，别的模型用了只会让音质变差
    hit = [n for n in names if "yue2" in n.lower()]
    return ["无"] + (hit or names)


def _audio_encoder_list():
    try:
        names = folder_paths.get_filename_list("audio_encoders")
    except Exception:
        return []
    hit = [n for n in names if "sheetsage" in os.path.basename(n).lower()]
    return hit or names


def _has_valid_gguf_header(path):
    """GGUF 头校验：magic + 版本 + KV 数量。防住下了一半的残文件。"""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except Exception:
        return False
    if len(head) < 24 or head[:4] != b"GGUF":
        return False
    import struct
    version = struct.unpack("<I", head[4:8])[0]
    kv_count = struct.unpack("<Q", head[16:24])[0]
    return version in (2, 3) and 5 <= kv_count <= 100000


def _gguf_list():
    out = []
    if not os.path.isdir(GGUF_DIR):
        return out
    for name in sorted(os.listdir(GGUF_DIR)):
        if not name.lower().endswith(".gguf"):
            continue
        if name.lower().startswith("mmproj"):
            continue  # 视觉投影文件，不是对话模型
        path = os.path.join(GGUF_DIR, name)
        try:
            if os.path.getsize(path) < 1_000_000_000:
                continue
        except OSError:
            continue
        if _has_valid_gguf_header(path):
            out.append(name)
    return out


def _input_audio_files():
    try:
        ind = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            os.listdir(ind), ["audio", "video"]
        )
        return ["无"] + sorted(files)
    except Exception:
        return ["无"]


def _asr_model_list():
    out = []
    if not os.path.isdir(ASR_MODEL_ROOT):
        return out
    for name in sorted(os.listdir(ASR_MODEL_ROOT)):
        if os.path.exists(os.path.join(ASR_MODEL_ROOT, name, "config.json")):
            out.append(name)
    return out


# ====================== HTTP 小工具（llama-server） ======================
def _http_get(url, timeout=2.5):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _http_post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


# ====================== AI 写词 ======================
LYRICS_SYSTEM_PROMPT = """你是 YuE2 音乐生成模型的提示词工程师，同时也是一位中文词作者。
用户给你一句简短的描述，你要产出两部分：一段「风格提示词」和一份「歌词」。

【风格提示词规则】
严格按这个公式，用英文逗号分隔，全部用英文：
语言 + 人声特征 + 曲风 + BPM + 乐器 + 制作质感

格式示意（只是示范字段顺序和写法，必须按用户描述重新填写每一项，不要照抄）：
English, gritty male vocal, alternative rock, 132 BPM, distorted electric guitars, live drums, raw garage production

- BPM 必须带单位，写「78 BPM」而不是「78」；慢歌 60-90，中速 90-120，快歌 120-160
- 语言必须显式写出来（Chinese / Cantonese / English / Chinese, Japanese），漏了会用英文口音唱中文
- 人声特征：warm female vocal / ethereal female vocal / husky male vocal / clear male vocal / gritty male vocal with raspy edge
- 乐器：fingerstyle acoustic guitar / piano / electric guitar / bass / live drums / synths / strings / harmonica / guzheng
- 制作质感：warm analog production / clean modern production / raw live-room production / retro 80s production / lo-fi intimate

【歌词规则】
- 用段落标签组织结构：[Verse] [Chorus] [Bridge] [Outro]
- 只写「要唱出来的字」，不要标点符号，不要写（吉他solo）（渐弱）这类演奏说明
- 每行 7-14 个字，尽量押韵但不硬凑
- 副歌重复同样的词，让它抓耳；主歌换画面、推进叙事
- 用具体意象代替空泛抒情（写「桌上那半瓶汽水」而不是「我很伤心」），具体意象要贴合这首歌的主题，且每首歌都要新写，不要复用本提示词里的字句
- 歌词总量要贴合唱片长度：按每行约 7 秒估算，用户会告诉你目标时长
- 语言与风格提示词里的语言保持一致

只输出 JSON，不要任何解释文字：
{"style": "英文风格提示词", "lyrics": "带段落标签的歌词，用\\n换行"}"""


def _build_lyrics_user_message(description, target_seconds):
    lines = max(8, round(target_seconds / SECONDS_PER_LINE))
    hint = (
        f"\n\n（要求：整首歌约 {int(round(target_seconds))} 秒。"
        f"全部唱词（所有段落加起来，副歌重复的每一遍都算）大约 {lines} 行，上下别差超过 2 行）"
    )
    return f"帮我写一首歌：{description.strip()}{hint}"


def _parse_lyrics_result(raw):
    """模型偶尔会裹 ```json 围栏或加解释，容错解析。"""
    if not raw:
        return None
    text = raw.strip()
    import re
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    style = (obj.get("style") or "").strip()
    lyrics = (obj.get("lyrics") or "").strip()
    if not style or not lyrics:
        return None
    return {"style": style, "lyrics": lyrics}


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _with_llama(model_file, fn, label="写词模型"):
    """起 llama-server → 跑 fn(base_url) → 保证回收。"""
    model_path = os.path.join(GGUF_DIR, model_file)
    if not os.path.exists(model_path):
        raise ValueError(f"找不到写词模型：{model_path}")
    if not os.path.exists(LLAMA_SERVER):
        raise ValueError(f"找不到 llama-server：{LLAMA_SERVER}")

    port = _free_port()
    env = dict(os.environ)
    # 预编译的 llama.cpp 不带 cudart。这个目录不上 PATH 会静默回落 CPU，慢 10 倍且不报错。
    env["PATH"] = CUDA_BIN + ";" + env.get("PATH", "")

    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "-c", "8192",
        "-ngl", "999",
        "-fa", "on",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--reasoning", "off",
        "--parallel", "1",
        "--alias", "yue-lyrics",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]

    with _external(cmd, env=env, label=label) as (proc, err):
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 180
        while time.time() < deadline:
            _check_interrupt()
            if proc.poll() is not None:
                raise ExternalProcessError(
                    f"{label}启动即退出（code {proc.returncode}）：\n{_tail(err)}"
                )
            try:
                status, _ = _http_get(f"{base}/health")
                if status == 200:
                    break
            except Exception:
                pass
            time.sleep(1.5)
        else:
            raise ExternalProcessError(f"{label}启动超时（180 秒）")

        logger.info(f"[Zoey音乐] {label}就绪，端口 {port}")
        return fn(base)


def _llm_write_lyrics(model_file, description, target_seconds):
    def _chat(base):
        body = {
            "model": "yue-lyrics",
            "messages": [
                {"role": "system", "content": LYRICS_SYSTEM_PROMPT},
                {"role": "user", "content": _build_lyrics_user_message(description, target_seconds)},
            ],
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
        }
        url = f"{base}/v1/chat/completions"
        deadline = time.time() + 90
        while True:
            _check_interrupt()
            try:
                status, text = _http_post_json(url, body)
                break
            except urllib.error.HTTPError as e:
                # 健康检查通过不代表能推理，服务端这时还会回 503
                if e.code == 503 and time.time() < deadline:
                    time.sleep(1.5)
                    continue
                raise ExternalProcessError(f"写词服务返回 {e.code}")
            except Exception as e:
                if time.time() < deadline:
                    time.sleep(1.5)
                    continue
                raise ExternalProcessError(f"连接写词服务失败：{e}") from e
        try:
            data = json.loads(text)
        except Exception:
            raise ExternalProcessError("写词服务返回的不是 JSON")
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        parsed = _parse_lyrics_result(content)
        if not parsed:
            raise ExternalProcessError(
                "写词模型没返回合法结果，原始输出：\n" + (content or "（空）")[:500]
            )
        return parsed

    return _with_llama(model_file, _chat, label="写词模型")


# ====================== ASR 转写 ======================
def _asr_transcribe(audio_path, language="zh"):
    models = _asr_model_list()
    if not models:
        raise ValueError(f"没有可用 ASR 模型，请在 {ASR_MODEL_ROOT} 下放一个含 config.json 的模型目录")
    if not os.path.exists(ASR_PYTHON):
        raise ValueError(f"找不到 ASR 环境：{ASR_PYTHON}")
    if not os.path.exists(ASR_SCRIPT):
        raise ValueError(f"找不到 ASR 脚本：{ASR_SCRIPT}")

    model_path = os.path.join(ASR_MODEL_ROOT, models[0])
    cmd = [ASR_PYTHON, ASR_SCRIPT, "--model", model_path]

    # stdout 是协议通道（脚本把真 stdout 复制出来了），所以不能交给 drain 线程
    with _external(cmd, label="语音识别", drain_stdout=False) as (proc, err):
        deadline = time.time() + 300
        while time.time() < deadline:
            _check_interrupt()
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    raise ExternalProcessError(
                        f"语音识别启动失败（code {proc.returncode}）：\n{_tail(err)}"
                    )
                time.sleep(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue  # 库打的杂音
            if obj.get("ready"):
                break
        else:
            raise ExternalProcessError("语音识别启动超时（300 秒）")

        payload = json.dumps({"id": 1, "audio": audio_path, "language": language}) + "\n"
        proc.stdin.write(payload)
        proc.stdin.flush()

        deadline = time.time() + 120
        while time.time() < deadline:
            _check_interrupt()
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    raise ExternalProcessError(
                        f"语音识别中途退出：\n{_tail(err)}"
                    )
                time.sleep(0.2)
                continue
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            if obj.get("id") != 1:
                continue
            if obj.get("error"):
                raise ExternalProcessError(f"语音识别失败：{obj['error']}")
            return (obj.get("text") or "").strip()
        raise ExternalProcessError("语音识别超时（120 秒）")


# ====================== 变声 ======================
def _voice_convert(song_path, reference_path, out_path, semitones=0, steps=30):
    for p, what in ((VC_PYTHON, "变声环境"), (VC_SCRIPT, "变声脚本"), (SEEDVC_DIR, "Seed-VC 仓库")):
        if not os.path.exists(p):
            raise ValueError(f"找不到{what}：{p}")
    if not os.path.exists(reference_path):
        raise ValueError(f"找不到音色参考音频：{reference_path}")

    workdir = os.path.join(os.path.dirname(out_path), "_vc_work")
    os.makedirs(workdir, exist_ok=True)

    cmd = [
        VC_PYTHON, VC_SCRIPT,
        "--song", song_path,
        "--reference", reference_path,
        "--out", out_path,
        "--seedvc", SEEDVC_DIR,
        "--workdir", workdir,
        "--semitones", str(int(semitones)),
        "--steps", str(int(steps)),
    ]
    if os.path.exists(VC_FFMPEG):
        cmd += ["--ffmpeg", VC_FFMPEG]

    env = dict(os.environ)
    # 这一步必须走官方源：hf-mirror 不返回 x-repo-commit 头，新版 huggingface_hub 会拒收
    env["HF_ENDPOINT"] = ""
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUTF8"] = "1"

    with _external(cmd, env=env, label="音色转换") as (proc, err):
        while True:
            _check_interrupt()
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.returncode != 0:
            raise ExternalProcessError(
                f"音色转换失败（code {proc.returncode}）：\n{_tail(err, 12)}"
            )
    if not os.path.exists(out_path):
        raise ExternalProcessError(f"音色转换结束但没有产出文件：{out_path}")
    return out_path


# ====================== YuE2 核心管线 ======================
def _load_audio_file(abs_path):
    from comfy_extras.nodes_audio import load as _load
    wave, sr = _load(abs_path)
    return {"waveform": wave.unsqueeze(0), "sample_rate": sr}


def _fmt_seconds(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


class ZoeyYue2Studio:
    """Zoey 音乐生成 - 一个节点出整首歌（文生 / 翻唱 / 写词 / 变声）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["文生音乐", "翻唱"], {"default": "文生音乐"}),
                "desc": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "一句话描述你想要的歌。开了「AI 写词」时由它生成风格和歌词。",
                }),
                "duration": ("INT", {
                    "default": 120, "min": 30, "max": 300, "step": 10,
                    "display": "slider",
                    "tooltip": "时长上限（秒），按每行歌词约 7 秒折算。注意这是上限不是承诺："
                               "歌词太长会挤占预算，模型也可能提前收尾。真实长度会写在「结果说明」里。",
                }),
                "takes": ("INT", {"default": 1, "min": 1, "max": TAKES_MAX,
                                  "tooltip": "一次出几版。每版种子不同，挑一首满意的。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),

                # ── 风格选择器（后端 COMBO，就是下拉框） ──
                "language": (STYLE_LANGUAGES, {
                    "default": "Chinese",
                    "tooltip": "唱词语言。必须显式选，漏了会用英文口音唱中文。",
                }),
                "vocal": (STYLE_VOCALS, {"default": "husky male vocal"}),
                "genre": (STYLE_GENRES, {"default": "folk"}),
                "bpm": ("INT", {"default": 78, "min": 60, "max": 180, "step": 1,
                                "tooltip": "慢歌 60-90，中速 90-120，快歌 120-160。"}),
                "instrument_1": (INSTRUMENT_CHOICES, {"default": "fingerstyle acoustic guitar"}),
                "instrument_2": (INSTRUMENT_CHOICES, {"default": "harmonica"}),
                "instrument_3": (INSTRUMENT_CHOICES, {"default": NONE}),
                "production": (STYLE_PRODUCTIONS, {"default": "warm analog production"}),

                # ── 手填覆盖项 ──
                "style": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "手填风格提示词。留空则由上面几个选择器自动拼。填了就以这里为准。",
                }),
                "lyrics": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "歌词，用 [Verse] [Chorus] 分段。留空时若开了 AI 写词则由模型写。"}),
                "ai_lyrics": ("BOOLEAN", {"default": False,
                                          "tooltip": "用本地大模型写歌词。只补空着的字段，手填的一律优先。"}),
                "llm_model": (_gguf_list() or ["（没找到 GGUF 模型）"], {}),
                "ref_audio_file": (_input_audio_files(), {
                    "tooltip": "翻唱 / 变声用的参考音频。也可以直接连 AUDIO 输入口，那边优先。",
                }),
                "ckpt_name": (_yue2_ckpt_list() or ["（没找到 YuE2 模型）"], {}),
                "lora_name": (_lora_list(), {}),
                "format": (["flac", "mp3"], {"default": "flac"}),
                "vc_enabled": ("BOOLEAN", {"default": False, "tooltip": "生成后把主唱换成参考音频的音色。"}),
                "vc_semitones": ("INT", {"default": 0, "min": -12, "max": 12}),
                "vc_steps": ("INT", {"default": 30, "min": 10, "max": 100}),
                "asr_enabled": ("BOOLEAN", {"default": False, "tooltip": "把成品转写成文本，方便核对唱对没有。"}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "结果说明")
    FUNCTION = "run"
    CATEGORY = "Zoey Tool/音乐生成"
    OUTPUT_NODE = True

    # ------------------------------------------------------------------
    def run(self, mode, desc, duration, takes, seed,
            language, vocal, genre, bpm,
            instrument_1, instrument_2, instrument_3, production,
            style, lyrics, ai_lyrics, llm_model, ref_audio_file,
            ckpt_name, lora_name, format,
            vc_enabled, vc_semitones, vc_steps, asr_enabled,
            ref_audio=None, unique_id=None, prompt=None, extra_pnginfo=None):

        import nodes as comfy_nodes
        import comfy.sample
        from comfy_extras.nodes_yue2 import (
            YuE2GenerateABC, YuE2GenerateMusic, EmptyYuE2LatentAudio,
        )
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_audio_encoder import (
            AudioEncoderLoader, SheetSage2AudioToABC,
        )
        from comfy_api.latest import UI

        warnings = []
        is_cover = (mode == "翻唱")
        duration = int(duration)
        takes = max(1, min(TAKES_MAX, int(takes)))

        # ---- 前置校验：这时候啥都还没产出，报错是诚实的 ----
        needs_reference = is_cover or vc_enabled
        reference_path = self._resolve_reference(ref_audio, ref_audio_file) if needs_reference else None
        if needs_reference and not reference_path:
            what = "翻唱" if is_cover else "音色转换"
            raise ValueError(f"{what}需要一段参考音频：连上 AUDIO 输入口，或在「参考音频」里选一个")

        desc = (desc or "").strip()
        style_override = (style or "").strip()
        lyrics = (lyrics or "").strip()

        # 风格优先级：手填 > AI 写词 > 下拉选择器拼出来的
        composed = _compose_style(
            language, vocal, genre, bpm,
            [instrument_1, instrument_2, instrument_3], production,
        )
        style = style_override or composed

        # ---- Phase 0：写词（必须排在 YuE2 加载之前，抢显存） ----
        # 手填的和下拉拼出来的都算「已经有风格了」，这时 AI 只补歌词；
        # 都没有才让 AI 连风格一起写。
        want_ai = bool(ai_lyrics) and (not style_override or not lyrics)
        if want_ai and not desc:
            warnings.append("开了 AI 写词但没填「歌曲描述」，已跳过写词")
        elif want_ai:
            _release_vram()
            logger.info(f"[Zoey音乐] 用 {llm_model} 写词中…")
            got = _llm_write_lyrics(llm_model, desc, duration)
            if not style_override:
                style = got["style"]
            if not lyrics:
                lyrics = got["lyrics"]
            _release_vram()

        lines = [l for l in lyrics.splitlines() if l.strip() and not l.strip().startswith("[")]
        est = len(lines) * SECONDS_PER_LINE
        if lyrics and est > duration:
            warnings.append(
                f"歌词约 {len(lines)} 行、估计 {est} 秒，超过设定的 {duration} 秒，结尾可能被截断"
            )

        bar = ProgressBar(takes * (DEFAULT_STEPS + 2))

        # ---- Phase 1：YuE2 ----
        logger.info(f"[Zoey音乐] 加载模型 {ckpt_name}")
        model, clip, vae = comfy_nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)

        if lora_name and lora_name != "无":
            # 必须同时命中 MODEL 和 CLIP：曲风/旋律 token 是 CLIP 那半边产生的，
            # 只接 MODEL 的话听感几乎不变。
            model, clip = comfy_nodes.LoraLoader().load_lora(
                model, clip, lora_name, DEFAULT_LORA_STRENGTH, DEFAULT_LORA_STRENGTH
            )
            logger.info(f"[Zoey音乐] 已应用 LoRA {lora_name}")

        # 翻唱：转谱只做一次，各版共用同一份 ABC
        cover_abc = None
        if is_cover:
            logger.info("[Zoey音乐] 识别参考曲旋律…")
            encoder = AudioEncoderLoader.execute(
                audio_encoder_name=(_audio_encoder_list() or ["sheetsage2_bf16.safetensors"])[0]
            ).result[0]
            if ref_audio is not None:
                ref_for_abc = ref_audio
            else:
                ref_for_abc = _load_audio_file(reference_path)
            cover_abc = SheetSage2AudioToABC.execute(
                audio_encoder=encoder, audio=ref_for_abc, mode="melody"
            ).result[0]
            if isinstance(cover_abc, (list, tuple)):
                cover_abc = cover_abc[0] if cover_abc else ""
            if not cover_abc or not str(cover_abc).strip():
                raise ValueError("识别完成但没有拿到乐谱，换一首旋律清晰一点的歌试试")
            bar.update(1)

        prefix = f"yue/{time.strftime('%Y%m%d_%H%M%S')}/song"
        outputs = []

        for i in range(takes):
            _check_interrupt()
            take_seed = (int(seed) + i * SEED_STRIDE) & 0xFFFFFFFFFFFFFFFF
            try:
                result = self._generate_take(
                    i, comfy_nodes, comfy.sample, UI,
                    YuE2GenerateABC, YuE2GenerateMusic, EmptyYuE2LatentAudio,
                    VAEDecodeAudio, model, clip, vae,
                    style, lyrics, take_seed, duration, DEFAULT_STEPS, DEFAULT_CFG,
                    DEFAULT_SAMPLER, DEFAULT_SCHEDULER, cover_abc, DEFAULT_ABC_MODE,
                    prefix, bar,
                )
            except Exception as e:
                # 单版失败不毁掉已完成的版本：记下来，继续或收尾
                warnings.append(f"第 {i + 1} 版生成失败：{e}")
                logger.exception("[Zoey音乐] 单版生成失败")
                if not outputs:
                    raise
                break
            outputs.append(result)
            logger.info(
                f"[Zoey音乐] 第 {i + 1}/{takes} 版完成，实际 {result['seconds']:.1f} 秒"
            )

        if not outputs:
            raise ValueError("一版都没生成成功，请查看日志")

        # ---- Phase 2：卸载 YuE2，再跑后处理 ----
        if vc_enabled or asr_enabled or format == "mp3":
            _release_vram()

        if vc_enabled:
            for item in outputs:
                try:
                    vc_out = os.path.join(
                        os.path.dirname(item["saved"][0]["abs"]), "vc",
                        os.path.basename(item["saved"][0]["abs"]),
                    )
                    os.makedirs(os.path.dirname(vc_out), exist_ok=True)
                    _voice_convert(
                        item["saved"][0]["abs"], reference_path, vc_out,
                        semitones=vc_semitones, steps=vc_steps,
                    )
                    item["audio"] = _load_audio_file(vc_out)
                    item["vc_file"] = vc_out
                    item["vc_saved"] = self._saved_of(vc_out)
                    logger.info(f"[Zoey音乐] 变声完成：{vc_out}")
                except BaseException as e:
                    if not isinstance(e, Exception):
                        raise
                    warnings.append(f"第 {item['index'] + 1} 版变声失败：{e}")
                    logger.exception("[Zoey音乐] 变声失败")

        if asr_enabled:
            for item in outputs:
                try:
                    target = item.get("vc_file") or item["saved"][0]["abs"]
                    text = _asr_transcribe(target)
                    txt_path = os.path.splitext(target)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    item["asr_text"] = text
                except BaseException as e:
                    if not isinstance(e, Exception):
                        raise
                    warnings.append(f"第 {item['index'] + 1} 版转写失败：{e}")
                    logger.exception("[Zoey音乐] 转写失败")

        if format == "mp3":
            for item in outputs:
                try:
                    res = UI.AudioSaveHelper.get_save_audio_ui(
                        item["audio"], prefix, None, format="mp3", quality="320k"
                    )
                    item["mp3"] = [self._abs_of(r) for r in res.as_dict().get("audio", [])]
                except BaseException as e:
                    if not isinstance(e, Exception):
                        raise
                    warnings.append(f"第 {item['index'] + 1} 版 MP3 转码失败：{e}")

        # ---- 元数据 ----
        meta = {
            "mode": mode,
            "duration_requested": duration,
            "style": style,
            "lyrics": lyrics,
            "abc_mode": DEFAULT_ABC_MODE,
            "takes": [
                {
                    "index": it["index"] + 1,
                    "seed": it["seed"],
                    "seconds": round(it["seconds"], 2),
                    "file": it["saved"][0]["abs"] if it["saved"] else None,
                    "vc_file": it.get("vc_file"),
                    "mp3": it.get("mp3"),
                    "abc": (it.get("abc") or "")[:2000],
                }
                for it in outputs
            ],
            "warnings": warnings,
        }
        try:
            meta_path = os.path.join(os.path.dirname(outputs[0]["saved"][0]["abs"]), "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(os.path.join(os.path.dirname(meta_path), "lyrics.txt"), "w", encoding="utf-8") as f:
                f.write(lyrics)
        except Exception as e:
            logger.warning(f"[Zoey音乐] 写 meta.json 失败：{e}")

        # ---- 组装返回值 ----
        waves = [it["audio"]["waveform"] for it in outputs]
        maxlen = max(w.shape[-1] for w in waves)
        padded = [
            torch.nn.functional.pad(w, (0, maxlen - w.shape[-1])) if w.shape[-1] < maxlen else w
            for w in waves
        ]
        batch = {"waveform": torch.cat(padded, dim=0), "sample_rate": outputs[0]["audio"]["sample_rate"]}

        report_lines = []
        for it in outputs:
            extra = []
            if it.get("vc_file"):
                extra.append("已变声")
            if it.get("mp3"):
                extra.append("已转 MP3")
            suffix = f"（{'，'.join(extra)}）" if extra else ""
            report_lines.append(
                f"第 {it['index'] + 1} 版　实际 {_fmt_seconds(it['seconds'])}（{it['seconds']:.1f} 秒）{suffix}"
            )
            for p in ([it["saved"][0]["abs"]] if it["saved"] else []):
                report_lines.append(f"　　{p}")
            for p in (it.get("mp3") or []):
                report_lines.append(f"　　{p}")

        if warnings:
            report_lines.append("")
            report_lines.append("注意：")
            report_lines.extend(f"　· {w}" for w in warnings)

        report = "\n".join(report_lines)
        payload = {
            "takes": [
                {"index": it["index"] + 1, "seconds": round(it["seconds"], 2),
                 "seed": it["seed"], "file": it["saved"][0]["abs"] if it["saved"] else None}
                for it in outputs
            ],
            "files": [it["saved"][0]["sub"] for it in outputs if it["saved"]],
            "warnings": warnings,
            "report": report,
        }

        # 前端靠 ui.audio 渲染播放器；变声过的就播变声后的那条
        ui_audio = []
        for it in outputs:
            play = it.get("vc_saved") or (it["saved"][0]["sub"] if it["saved"] else None)
            if play:
                ui_audio.append(play)

        return {
            "ui": {
                "audio": ui_audio,
                "zoey_yue": (json.dumps(payload, ensure_ascii=False),),
            },
            "result": (batch, report),
        }

    # ------------------------------------------------------------------
    def _resolve_reference(self, ref_audio, ref_audio_file):
        """优先用接进来的 AUDIO，否则用下拉选的文件。"""
        if ref_audio is not None:
            out = os.path.join(folder_paths.get_output_directory(), "yue", "_ref.wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            wave = ref_audio["waveform"]
            if wave.dim() == 3:
                wave = wave[0]                      # 只取第一首
            if wave.dim() == 2 and wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)   # 多声道并成单声道
            sr = int(ref_audio["sample_rate"])
            data = wave.cpu().numpy()
            if data.ndim == 2:
                data = data.T                       # (C, L) → (L, C)，soundfile 要按样本在前
            import soundfile as sf
            sf.write(out, data, sr)
            return out
        if ref_audio_file and ref_audio_file != "无":
            p = os.path.join(folder_paths.get_input_directory(), ref_audio_file)
            if not os.path.exists(p):
                raise ValueError(f"找不到参考音频：{p}")
            return p
        return None

    def _abs_of(self, saved_result):
        return os.path.join(
            folder_paths.get_output_directory(),
            saved_result.get("subfolder", ""),
            saved_result["filename"],
        )

    def _saved_of(self, abs_path):
        """把 output 目录下的绝对路径包成 ComfyUI 前端能认的 {filename, subfolder, type}。"""
        out_dir = folder_paths.get_output_directory()
        rel = os.path.relpath(os.path.dirname(abs_path), out_dir)
        sub = "" if rel == "." else rel.replace("\\", "/")
        return {"filename": os.path.basename(abs_path), "subfolder": sub, "type": "output"}

    # ------------------------------------------------------------------
    def _generate_take(self, index, comfy_nodes, comfy_sample, UI,
                       YuE2GenerateABC, YuE2GenerateMusic, EmptyYuE2LatentAudio,
                       VAEDecodeAudio, model, clip, vae,
                       style, lyrics, seed, duration, steps, cfg,
                       sampler_name, scheduler, cover_abc, abc_mode,
                       prefix, bar):
        """跑一版：写谱 → 生成 → 采样 → 解码 → 立刻落盘。"""
        import comfy.model_management as mm

        if cover_abc is not None:
            abc = cover_abc
            music_mode = "melody"
        else:
            out = YuE2GenerateABC.execute(
                clip=clip, style=style, lyrics=lyrics, seed=seed, mode=abc_mode,
                max_abc_tokens=8192, temperature=0.7, top_p=0.9, top_k=30,
                repetition_penalty=1.005, penalty_window=100,
            )
            abc = out.result[0] if out.result else ""
            music_mode = abc_mode
            bar.update(1)

        if not abc or not str(abc).strip():
            raise ValueError("写谱结果是空的，换个风格或歌词再试")

        mm.throw_exception_if_processing_interrupted()

        gen = YuE2GenerateMusic.execute(
            clip=clip, style=style, lyrics=lyrics, seed=seed, mode=music_mode,
            max_duration=float(duration), temperature=1.0, top_p=0.95, top_k=100,
            repetition_penalty=1.2, abc=str(abc),
        )
        if not gen.result:
            raise RuntimeError("YuE2GenerateMusic 没有返回结果")
        positive, seconds = gen.result[0], gen.result[1]
        bar.update(1)

        negative = comfy_nodes.ConditioningZeroOut().zero_out(positive)[0]

        # 潜空间长度必须用返回的真实秒数。和 duration 不一致时 KSampler 会直接报错。
        latent = EmptyYuE2LatentAudio.execute(seconds=seconds, batch_size=1).result[0]
        latent_image = comfy_sample.fix_empty_latent_channels(
            model, latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        noise = comfy_sample.prepare_noise(latent_image, seed, None)

        def _cb(step, x0, x, total_steps, **kwargs):
            bar.update(1)

        samples = comfy_sample.sample(
            model=model, noise=noise, steps=int(steps), cfg=float(cfg),
            sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative, latent_image=latent_image,
            denoise=1.0, force_full_denoise=True, seed=seed,
            callback=_cb,
        )

        audio = VAEDecodeAudio.execute(vae, {"samples": samples}).result[0]

        res = UI.AudioSaveHelper.get_save_audio_ui(
            audio, prefix, None, format="flac", quality="128k"
        )
        saved_raw = res.as_dict().get("audio", [])
        saved = [{"abs": self._abs_of(r), "sub": r} for r in saved_raw]

        return {
            "index": index,
            "seed": seed,
            "seconds": float(seconds),
            "abc": str(abc),
            "audio": audio,
            "saved": saved,
        }


NODE_CLASS_MAPPINGS = {"ZoeyYue2Studio": ZoeyYue2Studio}
NODE_DISPLAY_NAME_MAPPINGS = {"ZoeyYue2Studio": "Zoey - 音乐生成"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
