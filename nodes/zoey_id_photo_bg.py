# -*- coding: utf-8 -*-
"""
Zoey 证件照底色 - 四选一下拉，直接输出底色文字
"""


class ZoeyIDPhotoBackground:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": (["红底", "白底", "蓝底", "绿底"], {"default": "白底"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("background",)
    FUNCTION = "pick"
    CATEGORY = "Zoey Tool/图像工具"

    def pick(self, background):
        return (background,)


NODE_CLASS_MAPPINGS = {
    "ZoeyIDPhotoBackground": ZoeyIDPhotoBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyIDPhotoBackground": "Zoey 证件照底色",
}
