import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os


def _find_font(preferred=""):
    candidates = [
        preferred,
        # ── Windows 预装（免费商用） ──
        "C:/Windows/Fonts/msyh.ttc",          # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",        # 微软雅黑粗体
        "C:/Windows/Fonts/msyhl.ttc",         # 微软雅黑细体
        "C:/Windows/Fonts/simhei.ttf",        # 黑体
        "C:/Windows/Fonts/simsun.ttc",        # 宋体
        "C:/Windows/Fonts/simkai.ttf",        # 楷体
        "C:/Windows/Fonts/fangsong.ttf",      # 仿宋
        "C:/Windows/Fonts/arial.ttf",         # Arial
        "C:/Windows/Fonts/arialbd.ttf",       # Arial Bold
        "C:/Windows/Fonts/times.ttf",         # Times New Roman
        # ── 思源系列（Adobe + Google，SIL OFL 免费商用） ──
        "C:/Windows/Fonts/SourceHanSansSC-Regular.otf",
        "C:/Windows/Fonts/SourceHanSansSC-Bold.otf",
        "C:/Windows/Fonts/SourceHanSansSC-Medium.otf",
        "C:/Windows/Fonts/SourceHanSansSC-Light.otf",
        "C:/Windows/Fonts/SourceHanSerifSC-Regular.otf",
        "C:/Windows/Fonts/SourceHanSerifSC-Bold.otf",
        # ── Noto Sans/Serif SC（Google，SIL OFL 免费商用） ──
        "C:/Windows/Fonts/NotoSansSC-Regular.otf",
        "C:/Windows/Fonts/NotoSansSC-Bold.otf",
        "C:/Windows/Fonts/NotoSerifSC-Regular.otf",
        "C:/Windows/Fonts/NotoSerifSC-Bold.otf",
        # ── 阿里巴巴普惠体（免费商用） ──
        "C:/Windows/Fonts/AlibabaPuHuiTi-Regular.ttf",
        "C:/Windows/Fonts/AlibabaPuHuiTi-Bold.ttf",
        "C:/Windows/Fonts/AlibabaPuHuiTi-Medium.ttf",
        "C:/Windows/Fonts/AlibabaPuHuiTi-Light.ttf",
        # ── 得意黑 / Smiley Sans（免费商用） ──
        "C:/Windows/Fonts/SmileySans-Oblique.ttf",
        "C:/Windows/Fonts/SmileySans-Oblique.otf",
        # ── 霞鹜文楷（免费商用） ──
        "C:/Windows/Fonts/LXGWWenKai-Regular.ttf",
        "C:/Windows/Fonts/LXGWWenKai-Bold.ttf",
        "C:/Windows/Fonts/LXGWWenKai-Light.ttf",
        # ── 站酷系列（免费商用） ──
        "C:/Windows/Fonts/ZCOOL_QingKeHuangYou.ttf",
        "C:/Windows/Fonts/ZCOOL_Kuaile.ttf",
        "C:/Windows/Fonts/ZCOOL_XiaoWei.ttf",
        # ── 庞门正道标题体（免费商用） ──
        "C:/Windows/Fonts/PangMenZhengDaoBiaoTi.ttf",
        "C:/Windows/Fonts/PangMenZhengDaoBiaoTi.otf",
        # ── HarmonyOS Sans（华为，免费商用） ──
        "C:/Windows/Fonts/HarmonyOS_Sans_SC_Regular.ttf",
        "C:/Windows/Fonts/HarmonyOS_Sans_SC_Bold.ttf",
        # ── MiSans（小米，免费商用） ──
        "C:/Windows/Fonts/MiSans-Regular.ttf",
        "C:/Windows/Fonts/MiSans-Bold.ttf",
        # ── OPPO Sans（免费商用） ──
        "C:/Windows/Fonts/OPPOSans-Regular.ttf",
        "C:/Windows/Fonts/OPPOSans-Bold.ttf",
        # ── 英文免费商用字体 ──
        "C:/Windows/Fonts/Roboto-Regular.ttf",
        "C:/Windows/Fonts/Roboto-Bold.ttf",
        "C:/Windows/Fonts/OpenSans-Regular.ttf",
        "C:/Windows/Fonts/OpenSans-Bold.ttf",
        "C:/Windows/Fonts/Montserrat-Regular.ttf",
        "C:/Windows/Fonts/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/Poppins-Regular.ttf",
        "C:/Windows/Fonts/Poppins-Bold.ttf",
        "C:/Windows/Fonts/Lato-Regular.ttf",
        "C:/Windows/Fonts/Lato-Bold.ttf",
        "C:/Windows/Fonts/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/NotoSans-Bold.ttf",
        # ── macOS 预装 ──
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/NotoSansCJK-Regular.ttc",
        # ── Linux 常见路径 ──
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",         # 文泉驿正黑
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",       # 文泉驿微米黑
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


class ZoeyTextOverlay:
    """文本叠加 - 在图像上添加可拖拽定位的文字"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "text_config": ("STRING", {"multiline": False, "default": "{}", "dynamicPrompts": False}),
            },
            "optional": {
                "font_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "render"
    CATEGORY = "Zoey Tool/图像处理"
    OUTPUT_NODE = True

    def render(self, image, text, text_config="{}", font_path=""):
        try:
            cfg = json.loads(text_config)
        except Exception:
            cfg = {}

        B, H, W, C = image.shape
        results = []

        fp = _find_font(font_path)
        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(img_np).convert("RGBA")

            overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            size = int(cfg.get("size", 48))
            ox = float(cfg.get("x", 0.5))
            oy = float(cfg.get("y", 0.5))
            rot = float(cfg.get("r", 0))
            opacity = max(0, min(1, float(cfg.get("o", 1))))
            align = cfg.get("align", "center")
            hex_color = cfg.get("color", "#ffffff")

            try:
                hex_c = hex_color.lstrip("#")
                if len(hex_c) == 3:
                    hex_c = "".join(c * 2 for c in hex_c)
                fr = int(hex_c[0:2], 16)
                fg = int(hex_c[2:4], 16)
                fb = int(hex_c[4:6], 16)
            except Exception:
                fr, fg, fb = 255, 255, 255

            fa = int(opacity * 255)
            fill_color = (fr, fg, fb, fa)

            try:
                font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            px = int(ox * W - tw / 2)
            py = int(oy * H - th / 2)

            text_layer = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
            td = ImageDraw.Draw(text_layer)
            td.text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=fill_color)

            if rot != 0:
                text_layer = text_layer.rotate(rot, expand=True, center=(text_layer.width // 2, text_layer.height // 2),
                                                fillcolor=(0, 0, 0, 0))
                # 旋转后重新居中，确保文字中心保持在 (ox*W, oy*H)
                paste_x = int(ox * W - text_layer.width / 2)
                paste_y = int(oy * H - text_layer.height / 2)
            else:
                paste_x = px - 4
                paste_y = py - 4

            overlay.paste(text_layer, (paste_x, paste_y), text_layer)
            result = Image.alpha_composite(pil, overlay).convert("RGB")
            result_t = torch.from_numpy(np.array(result).astype(np.float32) / 255.0)
            results.append(result_t)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ZoeyTextOverlay": ZoeyTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyTextOverlay": "Zoey - 文本叠加",
}
