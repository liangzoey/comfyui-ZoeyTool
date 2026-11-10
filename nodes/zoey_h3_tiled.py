# -*- coding: utf-8 -*-
"""MiniMax H3 分块采样：空间 tile + 重叠加权融合（MultiDiffusion 式）。

把目标 video latent 切成带重叠的 tile，每个采样步对每块独立跑一次 DiT
（text/refs/audio 全部作为完整条件），再把各块预测的 velocity 按重叠窗
融回完整 latent。位置编码从全局 PackedLayout 派生——每块用全局归一化网格
的对应子集，因此 RoPE 坐标连续，拼接不错位。

代价：每步跑 N 块，耗时约 ×N；峰值激活降到单块序列的量级。
限制：参考模式（ref2va）的参考图/首尾帧每块都要全量带，序列降幅受参考
体量限制；T2V 收益最大。音频每块都完整算一遍，融合时取均值。
"""
import torch
import torch.nn.functional as F


class _TileLayout:
    """PackedLayout 的最小替身：只喂 _forward 需要的字段。"""
    __slots__ = ("seq_len", "position_ids", "img_pos", "img_update",
                 "audio_pos", "audio_update", "signature", "segments")


def derive_tile_layout(full, h0, h1, w0, w1):
    """从全局 layout 派生一块 tile 的 layout。

    video 段取全局 frame 网格在 [h0:h1, w0:w1] 的子集（坐标仍是全局的），
    前序 text/cond/ref_img/audio 段原样保留。
    """
    import comfy.ldm.minimax.model as mm

    text_len, latent_t, H, W, audio_t = full.signature
    frame, _ = mm._frame_grid(H, W)
    frame = frame.reshape(H // 2, W // 2, 2)
    tile_frame = frame[h0 // 2:h1 // 2, w0 // 2:w1 // 2].reshape(-1, 2)

    vstart = next(a for a, b, k in full.segments if k == "video")
    cursor = float(full.position_ids[vstart, 0])  # 目标视频起始时刻（全局）
    vpos = mm._video_grid(latent_t, tile_frame, cursor)

    seqs, off = [], 0
    for a, b, k in full.segments:
        if k == "video":
            continue
        n = b - a
        seqs.append((off, off + n, k))
        off += n
    nv = latent_t * tile_frame.shape[0]
    seqs.append((off, off + nv, "video"))

    pre_img = full.img_pos < vstart
    n_pre_img = int(pre_img.sum())

    lay = _TileLayout()
    lay.seq_len = vstart + nv
    lay.position_ids = torch.cat([full.position_ids[:vstart], vpos], dim=0)
    lay.img_pos = torch.cat([
        full.img_pos[pre_img],
        torch.arange(vstart, vstart + nv, dtype=full.img_pos.dtype),
    ])
    lay.img_update = torch.cat([
        full.img_update[:n_pre_img],
        torch.ones(nv, dtype=full.img_update.dtype),
    ])
    lay.audio_pos = full.audio_pos
    lay.audio_update = full.audio_update
    lay.signature = (text_len, latent_t, h1 - h0, w1 - w0, audio_t)
    lay.segments = seqs
    return lay


def _axis_split(n, k, overlap, patch=2):
    """把长度 n 均匀分成 k 块，相邻约重叠 overlap，全部对齐到 patch。"""
    if k <= 1:
        return [(0, n)]
    overlap = max(patch, (overlap // patch) * patch)
    tile = (n + (k - 1) * overlap + k - 1) // k
    tile = max(patch, (tile // patch) * patch)
    tile = min(n, tile)
    starts = []
    for i in range(k):
        s = round(i * (n - tile) / (k - 1))
        s = (s // patch) * patch
        s = max(0, min(n - tile, s))
        starts.append(s)
    out = []
    for s in dict.fromkeys(starts):
        r = (s, s + tile)
        if not out or r != out[-1]:
            out.append(r)
    if out[-1][1] < n:
        out[-1] = (n - tile, n)
    return out


def _ramp(n, overlap, device, dtype):
    """1D 线性窗：两端在重叠区线性升降，中段为 1。"""
    w = torch.ones(n, device=device, dtype=dtype)
    if overlap > 0 and overlap * 2 < n:
        r = torch.linspace(0.0, 1.0, overlap + 2, device=device, dtype=dtype)[1:-1]
        w[:overlap] = r
        w[-overlap:] = r.flip(0)
    return w


def _plan_tiles(H, W, nx, ny, overlap):
    hs = _axis_split(H, ny, overlap)   # 竖切行数
    ws = _axis_split(W, nx, overlap)   # 横切列数
    return [(h0, h1, w0, w1) for h0, h1 in hs for w0, w1 in ws]


def make_wrapper(nx, ny, overlap):
    """构造 DIFFUSION_MODEL wrapper：每步内切 nx×ny 块、融合 velocity。"""

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, dict) or payload.get("layout") is None:
            return executor(x, timestep, context, transformer_options, **kwargs)
        if kwargs.get("denoise_mask") is not None or kwargs.get("audio_denoise_mask") is not None:
            # 局部重绘类 mask 暂不支持分块，退回整图
            return executor(x, timestep, context, transformer_options, **kwargs)
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            return executor(x, timestep, context, transformer_options, **kwargs)

        full = payload["layout"]
        video, audio = x[0], x[1]
        if video.shape[0] != 1:
            return executor(x, timestep, context, transformer_options, **kwargs)

        Hl, Wl = full.signature[2], full.signature[3]
        H0, W0 = video.shape[3], video.shape[4]
        video_p = video
        if H0 != Hl or W0 != Wl:
            video_p = F.pad(video, (0, Wl - W0, 0, Hl - H0))

        tiles = _plan_tiles(Hl, Wl, nx, ny, overlap)
        if len(tiles) <= 1:
            return executor(x, timestep, context, transformer_options, **kwargs)

        dtype, device = video_p.dtype, video_p.device
        acc_v = torch.zeros_like(video_p)
        acc_w = torch.zeros((Hl, Wl), device=device, dtype=torch.float32)
        acc_a = None

        for h0, h1, w0, w1 in tiles:
            lay = derive_tile_layout(full, h0, h1, w0, w1)
            tp = dict(payload)
            tp["layout"] = lay
            tk = dict(kwargs)
            tk["minimax_payload"] = tp
            tv = video_p[..., h0:h1, w0:w1].contiguous()
            out = executor([tv, audio], timestep, context, transformer_options, **tk)
            win = _ramp(h1 - h0, overlap, device, torch.float32)[:, None] * _ramp(w1 - w0, overlap, device, torch.float32)[None, :]
            acc_v[..., h0:h1, w0:w1] += out[0].float() * win[None, None, None]
            acc_w[h0:h1, w0:w1] += win
            acc_a = out[1].float() if acc_a is None else acc_a + out[1].float()

        acc_w = acc_w.clamp_min(1e-6)[None, None, None]
        video_vel = (acc_v / acc_w).to(dtype)
        if H0 != Hl or W0 != Wl:
            video_vel = video_vel[:, :, :, :H0, :W0]
        audio_vel = (acc_a / len(tiles)).to(audio.dtype)
        return [video_vel, audio_vel]

    return wrapper


class ZoeyMiniMaxH3TiledSampler:
    """在给定 AV latent 上做分块采样，输出采样后的 AV latent。"""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                    "tooltip": "横向分几块"}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                    "tooltip": "纵向分几块"}),
                "tile_overlap": ("INT", {"default": 8, "min": 2, "max": 128, "step": 2,
                                         "tooltip": "相邻块重叠（latent 单位），越大接缝越顺、耗时越多"}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "run"
    CATEGORY = "ZoeyTool/H3"

    def run(self, model, positive, negative, latent, seed, steps, cfg,
            sampler_name, scheduler, tiles_x, tiles_y, tile_overlap, enabled):
        import comfy.sample
        import comfy.patcher_extension as pe

        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        m = model
        info = "tiled=off"
        if enabled:
            m = model.clone()
            pe.add_wrapper_with_key(
                pe.WrappersMP.DIFFUSION_MODEL, "zoey_h3_tiled",
                make_wrapper(int(tiles_x), int(tiles_y), int(tile_overlap)),
                m.model_options, is_model_options=True)
            info = f"tiled {tiles_x}x{tiles_y} overlap={tile_overlap}"

        samples = comfy.sample.sample(
            m, noise, steps, cfg, sampler_name, scheduler, positive, negative,
            latent_image, denoise=1.0, force_full_denoise=True,
            noise_mask=latent.get("noise_mask", None), seed=seed)
        return ({"samples": samples}, info)


NODE_CLASS_MAPPINGS = {
    "ZoeyMiniMaxH3TiledSampler": ZoeyMiniMaxH3TiledSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMiniMaxH3TiledSampler": "Zoey MiniMax H3 分块采样",
}
