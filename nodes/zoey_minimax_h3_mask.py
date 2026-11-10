# -*- coding: utf-8 -*-
"""Zoey MiniMax H3 遮罩角色替换（SAM3.1 自动遮罩）。

源视频帧 + 替换角色参考图 + 提示词 → 节点内部完成：
  1. SAM3 文本提示分割并逐帧跟踪得到目标遮罩（也可外接手绘遮罩覆盖）
  2. 源视频 VAE 编码，遮罩内加噪重绘、遮罩外保持原样；音频默认原样保留
  3. 走 MiniMax H3 ref2va 条件 + 采样，直接输出替换后的视频帧与音频

等价于「SEC4B+H3角色替换」工作流，但把 SeC 分割换成 ComfyUI 原生 SAM3。
"""

import math

import torch
import torch.nn.functional as F

from .zoey_minimax_h3 import (
    _RESOLUTION_TABLE,
    _apply_lora,
    _load_clip,
    _load_unet,
    _load_vae,
    _NONE,
)
from .zoey_minimax_h3_tags import count_refs, expand_at_tags

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    _H3_AVAILABLE = True
except Exception:
    _H3_AVAILABLE = False

_IMAGE_SLOTS = 4

_MASK_MODES = ["SAM3 自动", "手动遮罩"]

# 实测：H3 ref2va 在遮罩重绘里几乎只认文字，参考图只做辅助。
# 提示词必须把参考人物的长相/发型/服装写清楚，否则模型会自己编一个人。
REPLACE_DEFAULT_PROMPT = (
    "<Subject 1> is the person shown in <Picture 1>: [describe here - age, gender, face shape, "
    "hairstyle, outfit and accessories, e.g. a young woman with long straight black hair, wearing a "
    "black cropped hoodie and gray sweatpants].\n"
    "Replace the tracked person in the source video with <Subject 1>: same position, body pose and "
    "motion as the original person, standing inside the exact scene of the source footage with "
    "identical camera angle, scale, perspective, lighting and color palette.\n"
    "<Subject 1> keeps exactly the face, hairstyle, body proportions, outfit and accessories "
    "described above and shown in <Picture 1>.\n"
    "Everything outside the replaced subject (background, other people, camera motion, timing) "
    "stays exactly as in the source video."
)


def _sam3_files():
    import folder_paths
    files = folder_paths.get_filename_list("checkpoints")
    hits = [f for f in files if "sam3" in f.lower()]
    return hits or files or [_NONE]


def _pick(names, keywords, fallback):
    for k in keywords:
        for n in names:
            if k.lower() in n.lower():
                return n
    return names[0] if names else fallback


def _align_frames(n):
    """H3 视频帧数必须落在 17k+5 网格上。"""
    if n < 5:
        raise ValueError("遮罩替换需要源视频帧（至少 5 帧）；单张图请用 I2V 模式")
    return 5 + ((n - 5) // 17) * 17


def _canvas_from_source(width, height, resolution):
    """按源视频比例算画布；resolution=自动 时原始尺寸封顶到 1344*768 面积。"""
    if resolution and resolution != "自动" and resolution in _RESOLUTION_TABLE:
        tw, th = _RESOLUTION_TABLE[resolution]
        scale = math.sqrt((tw * th) / float(width * height))
    else:
        scale = min(1.0, math.sqrt((1344.0 * 768.0) / float(width * height)))
    cw = max(32, int(round(width * scale / 32.0)) * 32)
    ch = max(32, int(round(height * scale / 32.0)) * 32)
    return cw, ch


def _resize_frames(frames, width, height):
    import comfy.utils
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _fit_mask(mask, n, height, width):
    import comfy.utils
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    m = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).float()
    if (m.shape[-2], m.shape[-1]) != (height, width):
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
    m = comfy.utils.repeat_to_batch_size(m, n, dim=0)
    return m[:, 0].clamp(0.0, 1.0)


def _grow_mask(mask, expand):
    """分离式方形膨胀（GPU），比逐像素 scipy 膨胀快很多。"""
    if expand <= 0:
        return mask
    k = int(expand) * 2 + 1
    m = mask.unsqueeze(1)
    m = F.max_pool2d(m, kernel_size=(1, k), stride=1, padding=(0, expand))
    m = F.max_pool2d(m, kernel_size=(k, 1), stride=1, padding=(expand, 0))
    return m.squeeze(1)


def _feather_mask(mask, feather):
    if feather <= 0:
        return mask
    from scipy.ndimage import gaussian_filter
    out = []
    for m in mask:
        out.append(torch.from_numpy(
            gaussian_filter(m.cpu().numpy(), sigma=float(feather) / 2.0).astype("float32")))
    return torch.stack(out, dim=0).to(mask.device)


def _sam3_track_masks(frames, sam3_name, seg_prompt, threshold, max_objects,
                      object_indices, detect_interval):
    """用 ComfyUI 原生 SAM3 对帧序列做文本提示分割 + 视频跟踪，返回 [N,H,W] 遮罩。"""
    import folder_paths
    import comfy.sd
    import comfy.model_management
    import nodes as core_nodes
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack, SAM3_TrackToMask

    if not (seg_prompt or "").strip():
        raise ValueError("SAM3 自动遮罩需要填写分割提示词（英文名词，如 person / car / robot）")
    if sam3_name in (None, "", _NONE):
        raise ValueError("请选择 SAM3 模型（models/checkpoints 下的 sam3 权重）")

    path = folder_paths.get_full_path_or_raise("checkpoints", sam3_name)
    sam_model, sam_clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    if sam_clip is None:
        raise ValueError("SAM3 权重里没有文本编码器，无法用文本提示分割")

    try:
        cond = core_nodes.CLIPTextEncode().encode(sam_clip, seg_prompt.strip())[0]
        track = SAM3_VideoTrack.execute(
            images=frames,
            model=sam_model,
            initial_mask=None,
            conditioning=cond,
            detection_threshold=float(threshold),
            max_objects=max(0, int(max_objects)),
            detect_interval=max(1, int(detect_interval)),
        ).args[0]
        masks = SAM3_TrackToMask.execute(track, object_indices or "").args[0]
    finally:
        del sam_model, sam_clip
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    if masks is None or masks.numel() == 0 or float(masks.max()) <= 0:
        raise ValueError(
            "SAM3 没分割到目标：把分割提示词换成更具体的英文名词（如 person / woman / red car），"
            "或降低 detect_threshold / 换 max_objects")
    return masks.clamp(0.0, 1.0)


class ZoeyMiniMaxH3MaskReplace:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import folder_paths
        from .zoey_minimax_h3 import _vae_names
        unets = folder_paths.get_filename_list("diffusion_models")
        clips = folder_paths.get_filename_list("text_encoders")
        vaes = _vae_names()
        video_vaes = [v for v in vaes if "audio" not in v.lower()] or vaes
        audio_vaes = [v for v in vaes if "audio" in v.lower()]
        loras = [_NONE] + folder_paths.get_filename_list("loras")
        sam3 = _sam3_files()
        optional = {}
        for i in range(_IMAGE_SLOTS):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": "替换后的角色参考图；提示词里用 @P1 引用"})
        optional["audio"] = ("AUDIO", {"tooltip": "源视频音轨（默认原样保留）"})
        optional["mask"] = ("MASK", {"tooltip": "手绘/外部遮罩；连了就优先用它，忽略 SAM3"})
        optional["negative"] = ("CONDITIONING", {"tooltip": "负向条件；不接则用空提示词"})
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "源视频帧（VHS_LoadVideo / 任意 IMAGE 批次）"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": REPLACE_DEFAULT_PROMPT,
                    "placeholder": "描述遮罩里换成什么、站在源视频的哪个场景；@P1/@P2 引用参考图",
                }),
                "resolution": (["自动"] + list(_RESOLUTION_TABLE.keys()), {"default": "自动"}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                                      "tooltip": "match=按生成面积缩放参考图（快）；max=2048 短边（人物更保真但更慢）"}),
                "mask_mode": (_MASK_MODES, {"default": "SAM3 自动"}),
                "sam3_model": (sam3, {"default": _pick(sam3, ["sam3.1", "sam3"], _NONE),
                                      "tooltip": "SAM3 权重（建议 sam3.1 multiplex fp16）"}),
                "seg_prompt": ("STRING", {"default": "person",
                                          "placeholder": "要分割的英文名词，如 person / car / robot"}),
                "detect_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_objects": ("INT", {"default": 1, "min": 0, "max": 64,
                                        "tooltip": "最多跟踪几个目标；0=CUDA 内部上限"}),
                "object_indices": ("STRING", {"default": "",
                                              "placeholder": "取哪些跟踪目标，逗号分隔如 0,2；空=全部"}),
                "detect_interval": ("INT", {"default": 1, "min": 0, "max": 8,
                                            "tooltip": "每隔 N 帧重新检测一次（1=每帧；0 按 1 处理）"}),
                "mask_grow": ("INT", {"default": 16, "min": 0, "max": 400,
                                      "tooltip": "遮罩外扩像素；外扩越大越能盖住头发/边缘，但会吃掉更多背景"}),
                "mask_feather": ("INT", {"default": 6, "min": 0, "max": 128,
                                         "tooltip": "遮罩羽化（高斯 sigma*2），减少接缝"}),
                "invert_mask": ("BOOLEAN", {"default": False, "label_on": "反选", "label_off": "不反选"}),
                "preserve_audio": ("BOOLEAN", {"default": True, "label_on": "保留原音", "label_off": "重新生成"}),
                "lock_background": ("BOOLEAN", {"default": False, "label_on": "追加约束", "label_off": "关闭",
                                                "tooltip": "提示词里再追加一段「角色融入源视频场景、遮罩外保持不变」；默认提示词已包含，短提示词时打开"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.01,
                                      "tooltip": "遮罩内重绘强度；1=完全重绘，调低更贴近原画面"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "unet_name": (unets, {"default": _pick(unets, ["singularity", "h3", "minimax"], "")}),
                "clip_name": (clips, {"default": _pick(clips, ["qwen3vl_32b_minimax_h3", "minimax_h3", "qwen3vl"], "")}),
                "vae_name": (vaes, {"default": _pick(video_vaes, ["minimax", "h3"], "")}),
                "audio_vae_name": ([_NONE] + vaes, {
                    "default": _pick(audio_vaes, ["h3", "minimax"], _NONE),
                    "tooltip": "H3 音频 VAE；保留源音频或生成音频都需要"}),
                "lora_name": (loras, {"default": _NONE}),
                "lora_strength": ("FLOAT", {"default": 0.75, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_clip": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "MASK", "STRING")
    RETURN_NAMES = ("视频帧", "音频", "遮罩", "信息")
    FUNCTION = "process"
    CATEGORY = "Zoey Tool/MiniMax H3"
    OUTPUT_NODE = True

    def process(self, video, prompt, resolution, ref_image_size, mask_mode, sam3_model,
                seg_prompt, detect_threshold, max_objects, object_indices, detect_interval,
                mask_grow, mask_feather, invert_mask, preserve_audio, lock_background,
                denoise, seed, steps, cfg, sampler_name, scheduler,
                unet_name, clip_name, vae_name, audio_vae_name,
                lora_name, lora_strength, lora_clip,
                audio=None, mask=None, negative=None, **refs):
        if not _H3_AVAILABLE:
            raise RuntimeError("当前 ComfyUI 没有 MiniMax H3 核心节点（需要较新版本 comfy_extras）")
        if video is None:
            raise ValueError("请连接源视频帧")

        import comfy.model_management
        import comfy.sample as comfy_sample
        import latent_preview
        import nodes as core_nodes

        # ── 源视频：对齐 17k+5 帧网格、缩到画布尺寸 ──
        n_src = int(video.shape[0])
        n = _align_frames(n_src)
        if n < n_src:
            video = video[:n]
        cw, ch = _canvas_from_source(int(video.shape[2]), int(video.shape[1]), resolution)
        frames = _resize_frames(video, cw, ch)

        # ── 遮罩：手绘优先，否则 SAM3 分割 + 跟踪 ──
        if mask is not None:
            m = _fit_mask(mask, n, ch, cw).to(frames.device)
            mask_src = "手绘遮罩"
        elif mask_mode == "手动遮罩":
            raise ValueError("遮罩模式选了「手动遮罩」但没有连接 mask 输入")
        else:
            sam_masks = _sam3_track_masks(
                frames, sam3_model, seg_prompt, detect_threshold, max_objects,
                object_indices, detect_interval)
            m = _fit_mask(sam_masks, n, ch, cw).to(frames.device)
            mask_src = f"SAM3:{seg_prompt.strip()}"
        if invert_mask:
            m = 1.0 - m
        m = _grow_mask(m, int(mask_grow))
        m = _feather_mask(m, int(mask_feather)).clamp(0.0, 1.0)
        mask_percent = float(m.mean()) * 100.0

        # ── 加载 H3 模型 + LoRA ──
        if not unet_name:
            import folder_paths
            unet_name = folder_paths.get_filename_list("diffusion_models")[0]
        if not clip_name:
            import folder_paths
            clip_name = folder_paths.get_filename_list("text_encoders")[0]
        if not vae_name:
            vae_name = _first_vae()
        if not audio_vae_name:
            audio_vae_name = _first_audio_vae()
        if audio_vae_name in ("", _NONE):
            audio_vae = None
        else:
            audio_vae = _load_vae(audio_vae_name)

        model = _load_unet(unet_name)
        clip = _load_clip(clip_name)
        vae = _load_vae(vae_name)
        if lora_name and lora_name != _NONE:
            model, clip = _apply_lora(model, clip, lora_name, float(lora_strength), float(lora_clip))

        # ── 条件：替换角色参考图 + 提示词 ──
        ref_images = {}
        for i in range(_IMAGE_SLOTS):
            img = refs.get(f"ref_image_{i}")
            if img is not None:
                ref_images[f"ref_image_{i}"] = img
        counts = count_refs(len(ref_images), 0, 0, 0)
        try:
            prompt_text = expand_at_tags(prompt or "", counts)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if lock_background:
            prompt_text = ("Place the subject inside the source video's scene: match its camera angle, scale, "
                           "perspective, lighting and color palette. Everything outside the replaced subject "
                           "(background, other people, camera motion, timing and framing) stays exactly as in "
                           "the source video.\n" + prompt_text)

        outputs = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt_text, width=cw, height=ch, length=n,
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=ref_images or None,
            ref_videos=None, ref_video_audios=None, ref_audios=None,
        )
        cond = outputs.args[0]

        # ── 组装 AV latent：源视频编码 + 遮罩，音频默认保留 ──
        from comfy_extras.nodes_minimax_h3 import temporal_shape
        video_latent = vae.encode(frames)
        audio_t = temporal_shape(n)[2]

        if audio is not None and preserve_audio:
            if audio_vae is None:
                raise ValueError("要保留源音频，请选择 H3 音频 VAE")
            dur = n / 24.0
            wf = audio["waveform"]
            want = int(round(dur * audio["sample_rate"]))
            if wf.shape[-1] > want:
                audio = dict(audio)
                audio["waveform"] = wf[..., :want]
            from comfy_extras.nodes_audio import VAEEncodeAudio
            audio_z = VAEEncodeAudio.execute(audio_vae, audio).args[0]["samples"]
            audio_latent = {"samples": audio_z, "noise_mask": torch.zeros_like(audio_z)}
        else:
            audio_z = torch.zeros([1, 32, 2, audio_t],
                                  device=comfy.model_management.intermediate_device())
            audio_latent = {"samples": audio_z, "noise_mask": torch.ones_like(audio_z)}

        video_latent = {"samples": video_latent, "noise_mask": m}
        from comfy_extras.nodes_lt import LTXVConcatAVLatent
        latent = LTXVConcatAVLatent.execute(video_latent, audio_latent).args[0]

        # ── 采样：遮罩内重绘，遮罩外与音频按 noise_mask 保持 ──
        latent_image = latent["samples"]
        noise = comfy_sample.prepare_noise(latent_image, int(seed), None)
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        callback = latent_preview.prepare_callback(model, int(steps))
        samples = comfy_sample.sample(
            model, noise, int(steps), float(cfg), sampler_name, scheduler,
            cond, negative, latent_image,
            denoise=float(denoise), force_full_denoise=float(denoise) >= 1.0,
            noise_mask=latent.get("noise_mask"), callback=callback, seed=int(seed))

        out_frames = core_nodes.VAEDecode().decode(vae, {"samples": samples})[0]
        out_audio = None
        if audio_vae is not None:
            from comfy_extras.nodes_audio import VAEDecodeAudio
            out_audio = VAEDecodeAudio().decode(audio_vae, {"samples": samples})[0]

        info = (f"{mask_src} ｜ {n} 帧 {cw}×{ch} ｜ 遮罩占比 {mask_percent:.1f}% ｜ "
                f"{int(steps)} 步 cfg {float(cfg):g} {sampler_name}/{scheduler} ｜ denoise {float(denoise):g}")
        return (out_frames, out_audio, m.to(comfy.model_management.intermediate_device()), info)


def _first_vae():
    import folder_paths
    from .zoey_minimax_h3 import _vae_names
    names = [n for n in _vae_names() if "minimax" in n.lower() and "audio" not in n.lower()]
    return names[0] if names else _vae_names()[0]


def _first_audio_vae():
    from .zoey_minimax_h3 import _vae_names
    names = [n for n in _vae_names() if "minimax" in n.lower() and "audio" in n.lower()]
    return names[0] if names else _NONE


NODE_CLASS_MAPPINGS = {"ZoeyMiniMaxH3MaskReplace": ZoeyMiniMaxH3MaskReplace}
NODE_DISPLAY_NAME_MAPPINGS = {"ZoeyMiniMaxH3MaskReplace": "Zoey - MiniMax H3 遮罩角色替换 (SAM3)"}
