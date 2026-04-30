from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import os

# ⚠️ 核心：ComfyUI 强制要求在此声明 web 目录，否则前端 JS 会 404
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# 必须将 WEB_DIRECTORY 加入 __all__，ComfyUI 启动时会自动读取
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
