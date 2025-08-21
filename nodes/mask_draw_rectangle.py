import torch
import numpy as np
from PIL import Image, ImageDraw
import colorsys
import logging

# 获取ZoeyTool的日志记录器
logger = logging.getLogger("ZoeyTool")

class ZoeyMaskDrawBox:
    """
    Zoey工具集 - 遮罩区域边界框绘制节点
    在输入图像上根据遮罩区域绘制矩形框，支持自定义颜色、透明度和填充选项
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "线宽": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "填充": (["否", "是"], {"default": "否"}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "颜色预设": ([
                    "红色", "橙色", "黄色", "绿色", 
                    "青色", "蓝色", "紫色", "粉色",
                    "白色", "黑色", "灰色"
                ], {"default": "红色"}),
                "亮度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "饱和度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "边距百分比": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "绘制方框"
    CATEGORY = "Zoey工具集/图像编辑"
    
    def 绘制方框(self, 图像, 遮罩, 线宽, 填充="否", 不透明度=1.0, 
               颜色预设="红色", 亮度=0.5, 饱和度=1.0, 边距百分比=5.0):
        logger.debug(f"开始绘制边界框: 批次大小={图像.shape[0]}, 预设颜色={颜色预设}")
        
        # 根据预设名称和亮度/饱和度参数生成颜色
        rgba = self.获取颜色(颜色预设, 亮度, 饱和度, 不透明度)
        
        # 将PyTorch张量转换为PIL图像
        batch_size, height, width, _ = 图像.shape
        结果图像 = []
        
        for i in range(batch_size):
            # 转换当前批次图像为PIL格式
            图像张量 = 图像[i] * 255.0
            图像数组 = np.clip(图像张量.cpu().numpy().astype(np.uint8), 0, 255)
            img = Image.fromarray(图像数组).convert('RGBA')
            
            # 处理遮罩（单张或批量）
            if 遮罩.dim() == 2:
                当前遮罩 = 遮罩
            else:
                当前遮罩 = 遮罩[i] if i < 遮罩.shape[0] else 遮罩[0]
                
            # 获取遮罩区域的外接矩形（带边距）
            边界框 = self.获取遮罩外接矩形(当前遮罩, 边距百分比)
            
            if 边界框:  # 如果找到有效遮罩区域
                # 创建透明图层用于绘制
                叠加层 = Image.new('RGBA', img.size, (0, 0, 0, 0))
                绘制器 = ImageDraw.Draw(叠加层)
                
                if 填充 == "是":
                    # 绘制填充矩形
                    绘制器.rectangle(边界框, fill=rgba, outline=rgba[:3], width=线宽)
                else:
                    # 绘制空心矩形
                    绘制器.rectangle(边界框, outline=rgba, width=线宽)
                
                # 合并图层
                img = Image.alpha_composite(img, 叠加层)
                logger.debug(f"在图像{i}上绘制边界框: {边界框}")
            else:
                logger.warning(f"图像{i}未检测到有效遮罩区域")
            
            # 将PIL图像转回张量（移除非必要alpha通道）
            img = img.convert('RGB')
            图像张量 = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
            结果图像.append(图像张量)
        
        return (torch.stack(结果图像),)
    
    def 获取颜色(self, 颜色预设, 亮度, 饱和度, 不透明度):
        """根据预设名称和参数生成RGBA颜色"""
        # 基础颜色映射 
        颜色映射 = {
            "红色": (0.0, 1.0, 1.0),
            "橙色": (0.08, 1.0, 1.0),
            "黄色": (0.16, 1.0, 1.0),
            "绿色": (0.33, 1.0, 1.0),
            "青色": (0.5, 1.0, 1.0),
            "蓝色": (0.66, 1.0, 1.0),
            "紫色": (0.83, 1.0, 1.0),
            "粉色": (0.92, 1.0, 1.0),
            "白色": (0.0, 0.0, 1.0),
            "黑色": (0.0, 0.0, 0.0),
            "灰色": (0.0, 0.0, 0.5),
        }
        
        # 获取基础色相
        h, s, v = 颜色映射.get(颜色预设, (0.0, 1.0, 1.0))
        
        # 应用用户调整
        s = max(0.0, min(1.0, s * 饱和度))
        v = max(0.0, min(1.0, v * 亮度))
        
        # 转换为RGB 
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        a = int(不透明度 * 255)
        
        return (r, g, b, a)
    
    def 获取遮罩外接矩形(self, 遮罩张量, 边距百分比=5.0):
        """获取遮罩区域的最小外接矩形（带百分比边距）"""
        try:
            # 将PyTorch张量转换为NumPy数组
            遮罩数组 = 遮罩张量.cpu().numpy().squeeze() * 255
            遮罩数组 = 遮罩数组.astype(np.uint8)
            
            # 找到所有非零像素的坐标
            行方向 = np.any(遮罩数组 > 0, axis=1)
            列方向 = np.any(遮罩数组 > 0, axis=0)
            
            if not np.any(行方向) or not np.any(列方向):
                logger.debug("未检测到有效遮罩区域")
                return None  # 没有有效遮罩区域
            
            # 获取最小和最大坐标
            行最小值, 行最大值 = np.where(行方向)[0][[0, -1]]
            列最小值, 列最大值 = np.where(列方向)[0][[0, -1]]
            
            # 计算边距值（基于边界框尺寸的百分比）
            高度, 宽度 = 遮罩数组.shape
            水平边距 = int((列最大值 - 列最小值) * 边距百分比 / 100)
            垂直边距 = int((行最大值 - 行最小值) * 边距百分比 / 100)
            
            # 应用边距并确保在图像范围内
            列最小值 = max(0, 列最小值 - 水平边距)
            行最小值 = max(0, 行最小值 - 垂直边距)
            列最大值 = min(宽度 - 1, 列最大值 + 水平边距)
            行最大值 = min(高度 - 1, 行最大值 + 垂直边距)
            
            logger.debug(f"遮罩边界框计算: x=[{列最小值}-{列最大值}], y=[{行最小值}-{行最大值}]")
            return (列最小值, 行最小值, 列最大值, 行最大值)
        
        except Exception as e:
            logger.error(f"计算边界框时出错: {str(e)}")
            return None

# 节点注册 
NODE_CLASS_MAPPINGS = {
    "ZoeyMaskDrawBox": ZoeyMaskDrawBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoeyMaskDrawBox": "🎨 Zoey - 遮罩边界框绘制"
}