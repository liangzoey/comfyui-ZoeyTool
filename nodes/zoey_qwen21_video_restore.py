# -*- coding: utf-8 -*-
"""
Zoey Qwen2.1 视频超清修复（一体化）
节点内部加载 UNET / LoRA / CLIP / VAE（含 Qwen2.1 KV cache），逐帧跑编辑式超清重建。
可选「远景小脸优化」：检测小脸 → 截取放大 → 参考图引导重建 → 缩回原位贴回。

为什么必须逐帧：comfy_extras 的 TextEncodeQwenImage21 只取 batch 的第 0 帧（image[:1]），
整批喂进去只有第一帧有效；参考条件逐帧不同，数据流层没法批量，所以循环放在节点内部，
模型只加载一次、常驻复用。
"""

from collections import OrderedDict
import os

import torch
import comfy.utils
import comfy.model_management
import folder_paths

from comfy.samplers import KSAMPLER_NAMES, SCHEDULER_NAMES
from comfy_extras.nodes_qwen import TextEncodeQwenImage21, QwenImage21Cache


# 同配置的模型栈复用，避免每次执行重读几 GB 权重；只留最近几个，免得换参数时把显存堆爆
_LOAD_CACHE = OrderedDict()
_LOAD_CACHE_MAX = 2

# 检测模型缓存，键为 (folder, name)
_YOLO_CACHE = {}


def _ensure_ultralytics_folder():
    """ultralytics 目录不是核心默认项，靠 Impact Pack 注册；这里兜底，免得它没开时节点注册直接失败。"""
    try:
        fps = folder_paths.folder_names_and_paths
        path = os.path.join(folder_paths.models_dir, "ultralytics")
        if "ultralytics" not in fps:
            fps["ultralytics"] = ([path], {".pt", ".onnx", ".pth"})
            return
        paths, exts = fps["ultralytics"]
        if path not in paths:
            paths.append(path)
        if not exts:
            fps["ultralytics"] = (paths, {".pt", ".onnx", ".pth"})
    except Exception:
        pass


_ensure_ultralytics_folder()


def _to_pil(img):
    from PIL import Image
    arr = (img[0].detach().float().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _detect_faces(model_name, img, conf):
    """img: (1,H,W,3) float 0..1。返回整数像素框 [(x1,y1,x2,y2), ...]。"""
    key = ("ultralytics", model_name)
    yolo = _YOLO_CACHE.get(key)
    if yolo is None:
        from ultralytics import YOLO
        yolo = YOLO(folder_paths.get_full_path_or_raise("ultralytics", model_name))
        _YOLO_CACHE[key] = yolo

    res = yolo.predict(_to_pil(img), conf=conf, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    return [(int(round(float(a))), int(round(float(b))),
             int(round(float(c))), int(round(float(d)))) for a, b, c, d in xyxy]


def _get_stack(unet_name, weight_dtype, lora_name, strength_model,
               cache_device, cache_dtype, clip_name, vae_name):
    key = (unet_name, weight_dtype, lora_name, round(float(strength_model), 4),
           cache_device, cache_dtype, clip_name, vae_name)
    cached = _LOAD_CACHE.get(key)
    if cached is not None:
        _LOAD_CACHE.move_to_end(key)
        return cached

    import nodes as comfy_nodes  # ComfyUI 根目录 nodes.py 的 loader

    model = comfy_nodes.UNETLoader().load_unet(unet_name, weight_dtype)[0]
    if lora_name and lora_name != "None":
        model = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
            model, lora_name, strength_model)[0]
    model = QwenImage21Cache.execute(model, cache_device, cache_dtype)[0]
    clip = comfy_nodes.CLIPLoader().load_clip(clip_name, "qwen_image", "default")[0]
    vae = comfy_nodes.VAELoader().load_vae(vae_name)[0]

    _LOAD_CACHE[key] = (model, clip, vae)
    while len(_LOAD_CACHE) > _LOAD_CACHE_MAX:
        _LOAD_CACHE.popitem(last=False)
    return model, clip, vae


def _feather_patch(ch, cw, f, device, dtype):
    if f <= 0:
        return None
    f = min(f, cw // 2, ch // 2)
    if f <= 0:
        return None
    ramp = torch.linspace(0.0, 1.0, f + 2, device=device, dtype=dtype)[1:-1]
    mask = torch.ones(ch, cw, device=device, dtype=dtype)
    mask[:f, :] *= ramp[:, None]
    mask[-f:, :] *= ramp.flip(0)[:, None]
    mask[:, :f] *= ramp[None, :]
    mask[:, -f:] *= ramp.flip(0)[None, :]
    return mask.view(1, ch, cw, 1)


class ZoeyQwen21VideoRestore:
    """一体化逐帧 Qwen-Image-2.1 超清修复（可选小脸优化）。

    结构由参考图锁定，逐帧之间无几何漂移；denoise=1 为原全重绘，<1 走 img2img 更保结构。
    scale 控制输出分辨率 = 输入帧尺寸 × 倍数，扩散直接在大尺寸上采样（不是放大低清结果）。
    小脸优化：crop（放大后）当 image_1 保姿态，上传参考图当 image_2 出细节，贴回坐标全整数、零偏移。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                                 {"advanced": True}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "cache_device": (["auto", "gpu", "cpu", "off"], {"default": "auto", "advanced": True}),
                "cache_dtype": (["default", "int8", "int4"], {"default": "default", "advanced": True}),

                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "高清HDR"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32,
                                       "tooltip": "参考图编码分辨率，保持长宽比缩放到约 resolution×resolution；0 保持原尺寸（32 倍数）"}),
                "seed": ("INT", {"default": 999, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (KSAMPLER_NAMES,),
                "scheduler": (SCHEDULER_NAMES, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1,
                                    "tooltip": "输出分辨率 = 输入帧尺寸 × 倍数。采样直接在大尺寸上跑，显存和耗时会按倍数的平方涨。"}),

                "face_enhance": ("BOOLEAN", {"default": False, "label_on": "开", "label_off": "关"}),
                "face_model": (folder_paths.get_filename_list("ultralytics"),),
                "face_max_size": ("INT", {"default": 128, "min": 8, "max": 2048, "step": 8,
                                          "tooltip": "检测框长边小于此值才算小脸，才会送去重建"}),
                "face_count": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1,
                                       "tooltip": "每帧最多处理几张脸。命中多张小脸时按面积从大到小排序，只取前 N 张（优先离镜头最近、最大那张）。0 = 不限，全部处理"}),
                "face_padding": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True}),
                "face_work_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32, "advanced": True,
                                           "tooltip": "crop 放大后的长边尺寸，之后原样缩回"}),
                "face_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                                          "tooltip": "太高小脸会被模型靠先验重编、背景被编成白底；0.5–0.7 更贴原图"}),
                "face_prompt": ("STRING", {"multiline": True, "advanced": True,
                                           "default": "保持人物面部特征、五官、表情、角度、姿态、发型与原图完全一致，保持周围背景、光线、肤色不变，仅提升面部清晰度与皮肤、毛发细节",
                                           "tooltip": "小脸重建的提示词；留空则沿用上面的 prompt"}),
                "face_conf": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05, "advanced": True}),
                "face_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1, "advanced": True,
                                         "tooltip": "贴回时的羽化边像素，0 为硬贴"}),
            },
            "optional": {
                "face_ref": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "restore"
    CATEGORY = "Zoey Tool/视频处理"
    DESCRIPTION = ("逐帧 Qwen-Image-2.1 超清修复，可选小脸优化。\n"
                   "小脸：「检测框长边 < face_max_size」的才算；命中多张时按面积从大到小排序，"
                   "face_count 只取前 N 张（即优先最近、最大的脸），0 为不限。")

    def _enhance_faces(self, img, ctx):
        """就地把小脸重建后贴回 img（坐标全整数，零偏移）。"""
        comfy_nodes = ctx["comfy_nodes"]
        boxes = _detect_faces(ctx["face_model"], img, ctx["face_conf"])
        H, W = img.shape[1], img.shape[2]
        ref = ctx["face_ref"]

        # 只留小脸，按面积从大到小排，再按人数上限截断
        small = []
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or max(bw, bh) >= ctx["face_max_size"]:
                continue
            small.append((bw * bh, x1, y1, x2, y2))
        small.sort(key=lambda t: t[0], reverse=True)
        if ctx["face_count"] > 0:
            small = small[:ctx["face_count"]]

        for _, x1, y1, x2, y2 in small:
            bw, bh = x2 - x1, y2 - y1
            pad = ctx["face_padding"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            nx1 = max(0, int(round(cx - bw * (1.0 + 2.0 * pad) / 2.0)))
            ny1 = max(0, int(round(cy - bh * (1.0 + 2.0 * pad) / 2.0)))
            nx2 = min(W, int(round(cx + bw * (1.0 + 2.0 * pad) / 2.0)))
            ny2 = min(H, int(round(cy + bh * (1.0 + 2.0 * pad) / 2.0)))
            cw, ch = nx2 - nx1, ny2 - ny1
            if cw < 8 or ch < 8:
                continue

            crop = img[:, ny1:ny2, nx1:nx2, :]

            k = ctx["face_work_size"] / max(cw, ch)
            uw = max(16, int(round(cw * k)))
            uh = max(16, int(round(ch * k)))
            up = comfy.utils.common_upscale(
                crop.movedim(-1, 1), uw, uh, "lanczos", "disabled").movedim(1, -1)

            refs = {"image_1": up}
            if ref is not None:
                refs["image_2"] = ref[:1]

            encoded = TextEncodeQwenImage21.execute(
                clip=ctx["clip"], prompt=ctx["face_prompt"],
                negative_prompt=ctx["negative_prompt"], vae=ctx["vae"],
                resolution=ctx["resolution"], images=refs,
            )
            (result,) = comfy_nodes.common_ksampler(
                ctx["model"], ctx["seed"], ctx["steps"], ctx["cfg"],
                ctx["sampler_name"], ctx["scheduler"], encoded[0], encoded[1],
                {"samples": ctx["vae"].encode(up)}, denoise=ctx["face_denoise"],
            )
            dec = ctx["vae"].decode(result["samples"])
            if dec.shape[1:3] != (ch, cw):
                dec = comfy.utils.common_upscale(
                    dec.movedim(-1, 1), cw, ch, "lanczos", "disabled").movedim(1, -1)

            mask = _feather_patch(ch, cw, ctx["face_feather"], dec.device, dec.dtype)
            if mask is None:
                img[:, ny1:ny2, nx1:nx2, :] = dec
            else:
                img[:, ny1:ny2, nx1:nx2, :] = dec * mask + crop * (1.0 - mask)

            del encoded, result, dec

    def restore(self, unet_name, weight_dtype, lora_name, strength_model, clip_name, vae_name,
                cache_device, cache_dtype, images, prompt, negative_prompt, resolution,
                seed, steps, cfg, sampler_name, scheduler, denoise, scale,
                face_enhance, face_model, face_max_size, face_count, face_padding, face_work_size,
                face_denoise, face_prompt, face_conf, face_feather, face_ref=None):
        import nodes as comfy_nodes  # common_ksampler

        model, clip, vae = _get_stack(unet_name, weight_dtype, lora_name, strength_model,
                                      cache_device, cache_dtype, clip_name, vae_name)

        total = images.shape[0]
        pbar = comfy.utils.ProgressBar(total)
        out_frames = []

        for i in range(total):
            comfy.model_management.throw_exception_if_processing_interrupted()

            frame = images[i:i + 1]
            tgt_h, tgt_w = frame.shape[1], frame.shape[2]

            encoded = TextEncodeQwenImage21.execute(
                clip=clip, prompt=prompt, negative_prompt=negative_prompt,
                vae=vae, resolution=resolution, images={"image_1": frame},
            )
            positive, negative = encoded[0], encoded[1]

            # 放大倍数作用在采样分辨率上：先把帧插值放大，再在大尺寸上编码/采样/解码
            target = frame
            if scale != 1.0:
                tgt_w = max(16, int(round(tgt_w * scale)))
                tgt_h = max(16, int(round(tgt_h * scale)))
                target = comfy.utils.common_upscale(
                    frame.movedim(-1, 1), tgt_w, tgt_h, "lanczos", "disabled"
                ).movedim(1, -1)

            # vae.encode 返回张量，common_ksampler 要 {"samples": ...}
            latent = {"samples": vae.encode(target)}

            (result,) = comfy_nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, denoise=denoise,
            )

            decoded = vae.decode(result["samples"])
            # VAE 编码会按压缩比裁像素，解码回来可能和目标尺寸差几像素，贴回目标尺寸
            if decoded.shape[1:3] != (tgt_h, tgt_w):
                decoded = comfy.utils.common_upscale(
                    decoded.movedim(-1, 1), tgt_w, tgt_h, "lanczos", "disabled"
                ).movedim(1, -1)

            if face_enhance:
                self._enhance_faces(decoded, {
                    "comfy_nodes": comfy_nodes, "model": model, "clip": clip, "vae": vae,
                    "face_model": face_model, "face_ref": face_ref, "face_conf": face_conf,
                    "face_max_size": face_max_size, "face_count": face_count,
                    "face_padding": face_padding,
                    "face_work_size": face_work_size, "face_denoise": face_denoise,
                    "face_prompt": face_prompt or prompt, "face_feather": face_feather,
                    "negative_prompt": negative_prompt, "resolution": resolution,
                    "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": sampler_name, "scheduler": scheduler,
                })

            out_frames.append(decoded)

            del positive, negative, latent, result, target
            pbar.update(1)

        return (torch.cat(out_frames, dim=0),)


NODE_CLASS_MAPPINGS = {
    "ZoeyQwen21VideoRestore": ZoeyQwen21VideoRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyQwen21VideoRestore": "Zoey Qwen2.1 视频超清修复",
}
