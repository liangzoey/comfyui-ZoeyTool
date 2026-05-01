# 原有节点导入
from .batch_image_cropper import NODE_CLASS_MAPPINGS as CROPPER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CROPPER_DISPLAY
from .zoey_tool import NODE_CLASS_MAPPINGS as ZOEY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ZOEY_DISPLAY
from .multifunctional_image_editor import NODE_CLASS_MAPPINGS as EDITOR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as EDITOR_DISPLAY
from .image_edit_prompt_generator import NODE_CLASS_MAPPINGS as PROMPT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY

# 新增：导入 mask_draw_rectangle
# ⚠️ 注意：这里导入的是该文件中定义的变量名，必须与 .py 文件内一致
from .mask_draw_rectangle import NODE_CLASS_MAPPINGS as MASK_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MASK_DISPLAY

# 新增：VR 360° 预览节点导入
try:
    from .vr_360_preview import NODE_CLASS_MAPPINGS as VR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VR_DISPLAY
except ImportError:
    VR_MAPPINGS = {}
    VR_DISPLAY = {}

# 合并所有节点映射
NODE_CLASS_MAPPINGS = { 
    **CROPPER_MAPPINGS, 
    **ZOEY_MAPPINGS, 
    **EDITOR_MAPPINGS, 
    **PROMPT_MAPPINGS, 
    **MASK_MAPPINGS,   # ← 注入 mask_draw_rectangle 节点
    **VR_MAPPINGS 
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    **CROPPER_DISPLAY, 
    **ZOEY_DISPLAY, 
    **EDITOR_DISPLAY, 
    **PROMPT_DISPLAY, 
    **MASK_DISPLAY,    # ← 注入 mask_draw_rectangle 显示名
    **VR_DISPLAY 
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
