# -*- coding: utf-8 -*-
"""启动时给 comfy.model_base.MiniMaxH3.extra_conds 注入「参考/关键帧 latent 对齐」。

作用：即使 ComfyUI 更新把 model_base.py 里对条件路径的 2 倍对齐冲掉
（更新会覆盖 comfy 核心），ZoeyTool 每次启动都会重新包一层，保证不变量仍在。

幂等：若核心已对齐（新版），重复对齐是 no-op；若被覆盖（被还原），则补上。
"""
import torch

def _align(payload, ps):
    def even(z):
        pad = ()
        for i in range(z.ndim - 2):
            pad = (0, (ps[i] - z.shape[i + 2] % ps[i]) % ps[i]) + pad
        return torch.nn.functional.pad(z, pad, mode="circular")

    kfs = payload.get("keyframes")
    if kfs is not None:
        nk = []
        for kf in kfs:
            kf = dict(kf)
            l = kf.get("latent")
            if l is not None:
                l2 = even(l)
                kf["latent"] = l2
                kf["latent_h"] = l2.shape[3]
                kf["latent_w"] = l2.shape[4]
            nk.append(kf)
        payload["keyframes"] = nk
        payload["cond_video_latents"] = [kf["latent"] for kf in nk if kf.get("latent") is not None]
        payload["cond_audio_latents"] = [kf["audio_latent"] for kf in nk if kf.get("audio_latent") is not None]

    refs = payload.get("refs")
    if refs is not None:
        nr = []
        for r in refs:
            r = dict(r)
            l = r.get("latent")
            if l is not None:
                l2 = even(l)
                r["latent"] = l2
                if "latent_h" in r:
                    r["latent_h"] = l2.shape[3]
                if "latent_w" in r:
                    r["latent_w"] = l2.shape[4]
            nr.append(r)
        payload["refs"] = nr
        payload["cond_video_latents"] = payload.get("cond_video_latents", []) + [r["latent"] for r in nr if r.get("latent") is not None]
        payload["cond_audio_latents"] = payload.get("cond_audio_latents", []) + [r["audio_latent"] for r in nr if r.get("audio_latent") is not None]


def apply():
    """给 MiniMaxH3.extra_conds 包一层对齐；返回是否生效。"""
    try:
        import comfy.model_base as mb
    except Exception:
        return False

    cls = getattr(mb, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        return False
    if getattr(cls.extra_conds, "_zoey_aligned", False):
        return True

    orig = cls.extra_conds

    def wrapped(self, **kwargs):
        out = orig(self, **kwargs)
        try:
            mp = out.get("minimax_payload")
            if mp is not None and hasattr(mp, "cond"):
                _align(mp.cond, self.diffusion_model.patch_size)
        except Exception:
            pass
        return out

    wrapped._zoey_aligned = True
    cls.extra_conds = wrapped
    return True
