class WanPromptGenerator:
    """Wan2.2提示词生成器 - 根据用户选择自动生成影视级视频提示词"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "主体": ("STRING", {"default": "一位年轻女孩", "multiline": False}),
                "场景": ("STRING", {"default": "阳光下的田野", "multiline": False}),
                "动作": ("STRING", {"default": "轻轻抚弄野花", "multiline": False}),
                "光源类型": (["无", "日光", "人工光", "月光", "实用光", "火光", "荧光", "阴天光", "混合光", "晴天光"],),
                "光线类型": (["无", "柔光", "硬光", "顶光", "侧光", "背光", "底光", "边缘光", "剪影", "低对比度", "高对比度"],),
                "时间段": (["无", "白天", "夜晚", "黄昏", "日落", "黎明", "日出"],),
                "景别": (["无", "特写", "近景", "中景", "中近景", "中全景", "全景", "广角"],),
                "构图方式": (["无", "中心构图", "平衡构图", "右侧重构图", "左侧重构图", "对称构图", "短边构图"],),
                "镜头焦段": (["极", "中焦距", "广角", "长焦", "望远", "超广角-鱼眼"],),
                "机位角度": (["无", "过肩角度", "高角度", "低角度", "倾斜角度", "航拍"],),
                "镜头类型": (["无", "干净的单人镜头", "双人镜头", "三人镜头", "群像镜头", "定场镜头"],),
                "色调风格": (["无", "暖色调", "冷色调", "高饱和度", "低饱和度"],),
                "画面风格": (["无", "毛毡风格", "3D卡通", "像素风格", "木偶动画", "3D游戏", "黏土风格",
                                "二次元", "水彩画", "黑白动画", "油画风格", "移轴摄影", "延时拍摄"],),
                "面部表情": (["无", "愤怒", "恐惧", "高兴", "悲伤", "惊讶"],),
                "主体动作": (["无", "跑步", "滑滑板", "踢足球", "网球", "乒乓球", "滑雪",
                                  "篮球", "橄榄球", "顶碗舞", "侧手翻"],),
                "镜头运动": (["无", "镜头推进", "镜头拉远", "镜头向右移动", "镜头向左移动",
                                   "手持镜头", "复合运镜", "跟随镜头", "环绕运镜"],),
            },
            "optional": {
                "主体细节": ("STRING", {"multiline": True, "default": "身着少数民族服饰的黑发苗族少女"}),
                "场景细节": ("STRING", {"multiline": True, "default": "高草丛生的田野，远处有模糊的树木轮廓"}),
                "动作细节": ("STRING", {"multiline": True, "default": "双腿交叉坐下，双手轻轻抚弄身旁的野花"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Zoey Tool/提示词"

    def generate_prompt(self, **kwargs):
        prompt_parts = []

        if kwargs.get("主体细节", ""):
            prompt_parts.append(f"{kwargs['主体']}, {kwargs['主体细节']}")
        else:
            prompt_parts.append(kwargs["主体"])

        if kwargs.get("场景细节", ""):
            prompt_parts.append(f"{kwargs['场景']}, {kwargs['场景细节']}")
        else:
            prompt_parts.append(kwargs["场景"])

        if kwargs.get("动作细节", ""):
            prompt_parts.append(f"{kwargs['动作']}, {kwargs['动作细节']}")
        else:
            prompt_parts.append(kwargs["动作"])

        aesthetics = []
        if kwargs["光源类型"] != "无":
            aesthetics.append(kwargs["光源类型"])
        if kwargs["光线类型"] != "无":
            aesthetics.append(kwargs["光线类型"])
        if kwargs["时间段"] != "无":
            aesthetics.append(kwargs["时间段"])
        if kwargs["景别"] != "无":
            aesthetics.append(kwargs["景别"])
        if kwargs["构图方式"] != "无":
            aesthetics.append(kwargs["构图方式"])
        if kwargs["镜头焦段"] != "无":
            aesthetics.append(kwargs["镜头焦段"])
        if kwargs["机位角度"] != "无":
            aesthetics.append(kwargs["机位角度"])
        if kwargs["镜头类型"] != "无":
            aesthetics.append(kwargs["镜头类型"])
        if kwargs["色调风格"] != "无":
            aesthetics.append(kwargs["色调风格"])
        if aesthetics:
            prompt_parts.append(", ".join(aesthetics))

        if kwargs["画面风格"] != "无":
            prompt_parts.append(kwargs["画面风格"])
        if kwargs["面部表情"] != "无":
            prompt_parts.append(kwargs["面部表情"])
        if kwargs["主体动作"] != "无":
            prompt_parts.append(kwargs["主体动作"])
        if kwargs["镜头运动"] != "无":
            prompt_parts.append(kwargs["镜头运动"])

        full_prompt = ", ".join(prompt_parts)
        return (full_prompt,)


NODE_CLASS_MAPPINGS = {
    "WanPromptGenerator": WanPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanPromptGenerator": "Zoey - Wan2.2提示词生成器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
