import os
import glob
import time
import shutil
from PIL import Image

class BatchImageCropper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "placeholder": "输入图片文件夹路径"}),
                "output_folder": ("STRING", {"default": "", "placeholder": "输出文件夹路径（可选）"}),
                "file_types": ("STRING", {"default": "jpg,png,jpeg,bmp,tiff,webp", "placeholder": "支持的文件类型（逗号分隔）"}),
                "left_crop": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 1, "display": "slider"}),
                "right_crop": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 1, "display": "slider"}),
                "top_crop": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 1, "display": "slider"}),
                "bottom_crop": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 1, "display": "slider"}),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
                "preserve_names": ("BOOLEAN", {"default": True}),
                "prefix": ("STRING", {"default": "cropped_"}),
                "suffix": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("output_folder", "processed_count", "error_count", "process_log")
    FUNCTION = "crop_images"
    CATEGORY = "Image Processing/Batch"
    OUTPUT_IS_LIST = (False, False, False, False)

    def crop_images(self, input_folder, output_folder, file_types, 
                   left_crop, right_crop, top_crop, bottom_crop,
                   overwrite=False, preserve_names=True, prefix="cropped_", suffix=""):
        # ========================
        # 1. 初始化和路径验证
        # ========================
        if not os.path.exists(input_folder):
            raise ValueError(f"❌❌ 输入路径不存在: {input_folder}")
        
        # 创建带时间戳的输出目录（如果未指定）
        if not output_folder:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_folder = os.path.join(os.path.dirname(input_folder), f"cropped_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        # ========================
        # 2. 收集图像文件
        # ========================
        valid_exts = [ext.strip().lower() for ext in file_types.split(',')]
        patterns = [f"*.{ext}" for ext in valid_exts] + [f"*.{ext.upper()}" for ext in valid_exts]
        
        file_list = []
        for pattern in patterns:
            file_list.extend(glob.glob(os.path.join(input_folder, pattern)))
        file_list = list(set(file_list))  # 去重
        
        if not file_list:
            return (output_folder, 0, 0, "⚠️ 未找到匹配的图像文件")
        
        # ========================
        # 3. 主处理循环
        # ========================
        processed_count = 0
        error_count = 0
        log_lines = []
        
        for file_path in file_list:
            try:
                filename = os.path.basename(file_path)
                file_ext = os.path.splitext(filename)[1]
                
                # 生成输出文件名
                if preserve_names:
                    new_filename = f"{prefix}{os.path.splitext(filename)[0]}{suffix}{file_ext}"
                else:
                    new_filename = f"{prefix}{processed_count+1:04d}{suffix}{file_ext}"
                
                output_path = os.path.join(output_folder, new_filename)
                
                # 处理文件名冲突
                if os.path.exists(output_path) and not overwrite:
                    conflict_count = 1
                    while os.path.exists(output_path):
                        new_filename = f"{prefix}{os.path.splitext(filename)[0]}{suffix}_{conflict_count}{file_ext}"
                        output_path = os.path.join(output_folder, new_filename)
                        conflict_count += 1
                
                # 打开并裁剪图像
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # 计算裁剪区域
                    left = left_crop
                    top = top_crop
                    right = width - right_crop
                    bottom = height - bottom_crop
                    
                    # 验证裁剪区域有效性
                    if left >= right or top >= bottom:
                        raise ValueError(f"裁剪区域无效: 左({left}) >= 右({right}) 或 上({top}) >= 下({bottom})")
                    if left < 0 or top < 0 or right > width or bottom > height:
                        raise ValueError(f"裁剪区域超出图像边界: 原始尺寸={width}x{height}，裁剪区域=[左:{left}, 右:{right}, 上:{top}, 下:{bottom}]")
                    
                    # 执行裁剪
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(output_path)
                
                processed_count += 1
                log_lines.append(f"✅ {filename} -> {new_filename} | 裁剪:左:{left_crop},右:{right_crop},上:{top_crop},下:{bottom_crop}")
                
            except Exception as e:
                error_count += 1
                log_lines.append(f"❌❌ {filename} - 错误: {str(e)}")
        
        # ========================
        # 4. 生成处理摘要
        # ========================
        process_log = "\n".join(log_lines)
        summary = (
            f"==================== 处理摘要 ====================\n"
            f"• 输入目录: {input_folder}\n"
            f"• 输出目录: {output_folder}\n"
            f"• 成功处理: {processed_count} 张图片\n"
            f"• 失败处理: {error_count} 张图片\n"
            f"• 裁剪参数: 左={left_crop}px, 右={right_crop}px, 上={top_crop}px, 下={bottom_crop}px\n"
            f"• 文件名处理: {'保留原名' if preserve_names else '生成序列名'}\n"
            f"• 文件名前缀: '{prefix}' | 后缀: '{suffix}'\n"
            f"• 文件覆盖: {'启用' if overwrite else '禁用'}\n"
            f"================================================\n\n"
            f"==================== 详细日志 ====================\n"
            f"{process_log}"
        )
        
        return (output_folder, processed_count, error_count, summary)

# 节点注册
NODE_CLASS_MAPPINGS = {"BatchImageCropper": BatchImageCropper}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchImageCropper": "zoey 图像批量裁剪器（全向自由裁剪）"}