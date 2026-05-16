# -*- coding: utf-8 -*-
"""
Zoey Rename Nodes - 重命名专用节点
包含：MultiFileBatchRenamer
"""

import os
import re
import glob
from typing import List, Dict, Tuple, Optional, Any

# ====================== 节点 7: MultiFileBatchRenamer ======================
class MultiFileBatchRenamer:
    """多文件批量重命名器 - 仅保留此节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {"default": "", "placeholder": "输入目录路径"}),
                "file_extension": ([".jpg", ".png", ".mp4", ".*"], {"default": ".*"}),
                "prefix": ("STRING", {"default": "img"}),
                "start_number": ("INT", {"default": 1, "min": 0, "step": 1}),
                "number_digits": ("INT", {"default": 4, "min": 1, "max": 10}),
                "suffix": ("STRING", {"default": "", "placeholder": "可选后缀"}),
                "find_text": ("STRING", {"default": "", "placeholder": "查找（留空则不替换）"}),
                "replace_text": ("STRING", {"default": "", "placeholder": "替换为（留空则删除）"}),
            },
            "optional": {
                "trigger": (["any"], {"forceInput": True, "label": "触发器（连接前一个节点的输出以确保顺序）"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_directory",)
    FUNCTION = "rename_files"
    CATEGORY = "Zoey Tool/文件工具"

    def rename_files(self, input_directory: str, file_extension: str, prefix: str, start_number: int, number_digits: int, suffix: str, find_text: str, replace_text: str, trigger=None) -> Tuple[str]:
        input_directory = os.path.normpath(input_directory)
        if not os.path.isdir(input_directory):
            raise ValueError(f"输入目录不存在: {input_directory}")

        # 获取所有文件
        pattern = f"*{file_extension}" if file_extension != ".*" else "*"
        files = glob.glob(os.path.join(input_directory, pattern))
        
        # 自然排序
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        files.sort(key=natural_sort_key)

        # 重命名逻辑
        counter = start_number
        for file_path in files:
            try:
                # 处理查找替换
                base_name = os.path.basename(file_path)
                if find_text and find_text in base_name:
                    base_name = base_name.replace(find_text, replace_text)

                # 构建新文件名
                num_str = str(counter).zfill(number_digits)
                new_name = f"{prefix}{num_str}"
                if suffix:
                    new_name += f"_{suffix}"
                new_name += f"_{base_name}" # 保留原文件名作为后缀（可选逻辑，可根据需求调整）
                
                # 防止重名
                ext = os.path.splitext(file_path)[1]
                new_path = os.path.join(input_directory, new_name + ext)
                
                # 重命名
                os.rename(file_path, new_path)
                counter += 1
            except Exception as e:
                print(f"重命名失败 {file_path}: {e}")
                continue

        return (input_directory,)

# ====================== 注册 ======================
NODE_CLASS_MAPPINGS = {
    "MultiFileBatchRenamer": MultiFileBatchRenamer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiFileBatchRenamer": "Zoey - 多文件批量重命名"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']