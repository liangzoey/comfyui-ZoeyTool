import torch
import numpy as np
import cv2
import math
from PIL import Image
import torchvision.transforms as transforms

class 多功能图像编辑器:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "操作类型": ([
                    "水平翻转", 
                    "垂直翻转", 
                    "旋转180度", 
                    "顺时针旋转90度", 
                    "逆时针旋转90度",
                    "图像分割",
                    "边缘检测",
                    "高斯模糊",
                    "锐化",
                    "阈值二值化",
                    "颜色反转",
                    "灰度化",
                    "透视变换",
                    "图像融合",
                    "风格化"
                ], {"default": "水平翻转"}),
                "横向分割数": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "纵向分割数": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "模糊半径": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "锐化强度": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "阈值": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "融合图像": ("IMAGE",),
                "融合权重": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "执行操作"
    CATEGORY = "图像处理"
    OUTPUT_IS_LIST = (True,)

    def 执行操作(self, 图像, 操作类型, 横向分割数, 纵向分割数, 模糊半径=5.0, 锐化强度=1.5, 
              阈值=128, 融合图像=None, 融合权重=0.5):
        
        # 将PyTorch张量转换为PIL图像列表
        图像列表 = self.张量转PIL(图像)
        结果图像 = []
        
        # 执行选择的图像操作
        if 操作类型 == "水平翻转":
            for img in 图像列表:
                结果图像.append(img.transpose(Image.FLIP_LEFT_RIGHT))
                
        elif 操作类型 == "垂直翻转":
            for img in 图像列表:
                结果图像.append(img.transpose(Image.FLIP_TOP_BOTTOM))
                
        elif 操作类型 == "旋转180度":
            for img in 图像列表:
                结果图像.append(img.rotate(180))
                
        elif 操作类型 == "顺时针旋转90度":
            for img in 图像列表:
                结果图像.append(img.rotate(-90))
                
        elif 操作类型 == "逆时针旋转90度":
            for img in 图像列表:
                结果图像.append(img.rotate(90))
                
        elif 操作类型 == "图像分割":
            for img in 图像列表:
                分割子图 = self.分割图像(img, 横向分割数, 纵向分割数)
                结果图像.extend(分割子图)
                
        elif 操作类型 == "边缘检测":
            for img in 图像列表:
                # 转换为OpenCV格式
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # 执行Canny边缘检测
                edges = cv2.Canny(opencv_img, 100, 200)
                # 将单通道边缘图像转换为三通道
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                结果图像.append(Image.fromarray(edges_rgb))
                
        elif 操作类型 == "高斯模糊":
            for img in 图像列表:
                # 转换为OpenCV格式
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # 应用高斯模糊
                blurred = cv2.GaussianBlur(opencv_img, (0, 0), 模糊半径)
                # 转换回RGB
                blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                结果图像.append(Image.fromarray(blurred_rgb))
                
        elif 操作类型 == "锐化":
            for img in 图像列表:
                # 锐化核
                kernel = np.array([
                    [-锐化强度, -锐化强度, -锐化强度],
                    [-锐化强度, 1 + 8 * 锐化强度, -锐化强度],
                    [-锐化强度, -锐化强度, -锐化强度]
                ])
                # 转换为OpenCV格式
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # 应用锐化
                sharpened = cv2.filter2D(opencv_img, -1, kernel)
                # 转换回RGB
                sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                结果图像.append(Image.fromarray(sharpened_rgb))
                
        elif 操作类型 == "阈值二值化":
            for img in 图像列表:
                # 转换为灰度图
                gray_img = img.convert("L")
                # 应用阈值
                binary = gray_img.point(lambda p: 255 if p > 阈值 else 0)
                # 转换为RGB
                binary_rgb = binary.convert("RGB")
                结果图像.append(binary_rgb)
                
        elif 操作类型 == "颜色反转":
            for img in 图像列表:
                # 反转颜色
                inverted = Image.eval(img, lambda x: 255 - x)
                结果图像.append(inverted)
                
        elif 操作类型 == "灰度化":
            for img in 图像列表:
                # 转换为灰度图
                gray = img.convert("L")
                # 转换为RGB（三通道灰度）
                gray_rgb = gray.convert("RGB")
                结果图像.append(gray_rgb)
                
        elif 操作类型 == "透视变换":
            for img in 图像列表:
                # 转换为OpenCV格式
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w = opencv_img.shape[:2]
                
                # 定义原始点（图像四角）
                src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                # 定义目标点（创建透视效果）
                dst_points = np.float32([
                    [w*0.1, h*0.1], 
                    [w*0.9, h*0.2], 
                    [w*0.8, h*0.9], 
                    [w*0.2, h*0.8]
                ])
                
                # 计算透视变换矩阵
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                # 应用透视变换
                transformed = cv2.warpPerspective(opencv_img, matrix, (w, h))
                # 转换回RGB
                transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
                结果图像.append(Image.fromarray(transformed_rgb))
                
        elif 操作类型 == "图像融合":
            if 融合图像 is None:
                raise ValueError("图像融合操作需要提供融合图像")
                
            # 将融合图像转换为PIL列表
            融合图像列表 = self.张量转PIL(融合图像)
            
            for i in range(min(len(图像列表), len(融合图像列表))):
                img1 = 图像列表[i]
                img2 = 融合图像列表[i]
                
                # 确保两张图像尺寸相同
                if img1.size != img2.size:
                    img2 = img2.resize(img1.size)
                
                # 转换为数组
                array1 = np.array(img1).astype(np.float32)
                array2 = np.array(img2).astype(np.float32)
                
                # 图像融合
                blended = (1 - 融合权重) * array1 + 融合权重 * array2
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                结果图像.append(Image.fromarray(blended))
                
        elif 操作类型 == "风格化":
            for img in 图像列表:
                # 转换为OpenCV格式
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # 应用风格化滤镜
                stylized = cv2.stylization(opencv_img, sigma_s=60, sigma_r=0.45)
                # 转换回RGB
                stylized_rgb = cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)
                结果图像.append(Image.fromarray(stylized_rgb))
                
        # 将PIL图像转换回PyTorch张量
        输出张量 = self.PIL转张量(结果图像)
        return (输出张量,)
    
    def 张量转PIL(self, 张量):
        """将PyTorch张量转换为PIL图像列表"""
        图像列表 = []
        for i in range(张量.size(0)):
            img = 张量[i].numpy() * 255.0
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            图像列表.append(img)
        return 图像列表
    
    def PIL转张量(self, pil_list):
        """将PIL图像列表转换为PyTorch张量列表"""
        张量列表 = []
        for img in pil_list:
            # 转换为numpy数组
            img_array = np.array(img).astype(np.float32) / 255.0
            # 转换为PyTorch张量
            img_tensor = torch.from_numpy(img_array)[None,]
            张量列表.append(img_tensor)
        return 张量列表
    
    def 分割图像(self, img, 横向分割数, 纵向分割数):
        """将图像分割为指定数量的子图"""
        width, height = img.size
        子图列表 = []
        
        # 计算每个子图的宽度和高度
        子图宽度 = width // 横向分割数
        子图高度 = height // 纵向分割数
        
        # 按行优先顺序分割图像
        for row in range(纵向分割数):
            for col in range(横向分割数):
                # 计算子图边界
                left = col * 子图宽度
                upper = row * 子图高度
                right = left + 子图宽度
                lower = upper + 子图高度
                
                # 裁剪子图
                子图 = img.crop((left, upper, right, lower))
                子图列表.append(子图)
                
        return 子图列表

# 节点注册
NODE_CLASS_MAPPINGS = {
    "多功能图像编辑器": 多功能图像编辑器
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "多功能图像编辑器": "zoey🖼🖼🖼️ 多功能图像编辑器"
}